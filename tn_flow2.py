import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import corner
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from pycbc.waveform import get_td_waveform
from pycbc.filter import resample_to_delta_t
from sklearn.metrics import confusion_matrix

from typing import Tuple
import seaborn as sns
from pycbc.types import TimeSeries

torch.manual_seed(42)
np.random.seed(42)


def rotate_half(x):
    """
    Rotate last dimension in pairs: (..., 2i, 2i+1) -> (-2i+1, 2i)
    """
    last_dim = x.shape[-1]
    assert last_dim % 2 == 0
    x = x.view(*x.shape[:-1], last_dim // 2, 2)  # (..., D/2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    x_rot = torch.stack([-x2, x1], dim=-1)
    return x_rot.view(*x.shape[:-2], last_dim)

def apply_rotary_emb(freqs, x):
    """Apply rotary embedding: x * cos(freq) + rotate_half(x) * sin(freq)"""
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())


class RotaryEmbeddingND(nn.Module):
    def __init__(
        self,
        spatial_shape: Tuple[int, ...],
        head_dim: int,
        base: float = 10000.0,
        learned_freq: bool = False,
        use_xpos: bool = False,
        xpos_scale_base: float = 512.0,
        theta_rescale_factor: float = 1.0,
        cache_max_len: int = 8192
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.ndims = len(spatial_shape)
        assert head_dim % self.ndims == 0
        self.dim_per_axis = head_dim // self.ndims
        assert self.dim_per_axis % 2 == 0

        self.learned_freq = learned_freq
        self.use_xpos = use_xpos
        self.xpos_scale_base = xpos_scale_base
        self.theta_rescale_factor = theta_rescale_factor
        self.base = base
        self.cache_max_len = cache_max_len

        self.cached_lengths = set()  # keep track of cached sequence lengths

        # Flattened patch coordinates
        coords = torch.meshgrid(*[torch.arange(s) for s in spatial_shape], indexing="ij")
        coords = torch.stack([c.reshape(-1) for c in coords], dim=-1)  # (N_patches, ndims)
        self.register_buffer("patch_coords", coords, persistent=False)

        # Inverse frequencies per axis (with NTK rescaling)
        self.inv_freqs = nn.ParameterList()
        for _ in range(self.ndims):
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
            inv_freq = inv_freq * (theta_rescale_factor ** (self.dim_per_axis / (self.dim_per_axis - 2)))
            self.inv_freqs.append(nn.Parameter(inv_freq, requires_grad=learned_freq))

        # XPOS base scale per dimension
        if use_xpos:
            scale = (torch.arange(0, self.dim_per_axis, 2) + 0.4 * self.dim_per_axis) / (1.4 * self.dim_per_axis)
            self.register_buffer('scale', scale)

        # Frequency caching
        self.register_buffer('cached_freqs', torch.zeros(cache_max_len, head_dim))
        self.cached_freqs_len = 0

    @property
    def device(self):
        return self.patch_coords.device

    def _get_full_coords(self):
        """Include CLS token (position 0)"""
        N_patches = self.patch_coords.shape[0]
        N_total = 1 + N_patches
        pos_coords = torch.zeros((N_total, self.ndims), device=self.device, dtype=self.patch_coords.dtype)
        pos_coords[1:] = self.patch_coords
        return pos_coords

    def _get_full_coords_no_cls(self):
        """Return patch coordinates only, no CLS token"""
        return self.patch_coords # shape: (N_patches, ndims)

    def _compute_freqs(self, pos_coords):
        """
        Compute full rotary frequencies with vectorization, XPOS, and NTK scaling
        Returns: (N_total, head_dim)
        """
        freqs_axes = []
        for axis, inv_freq in enumerate(self.inv_freqs):
            theta = pos_coords[:, axis:axis+1].to(inv_freq.dtype) @ inv_freq.unsqueeze(0)  # (N_total, dim_per_axis/2)
            theta = theta.repeat(1, 2)  # rotation pairs
            freqs_axes.append(theta)
        freqs = torch.cat(freqs_axes, dim=-1)  # (N_total, head_dim)

        if self.use_xpos:
            # position-dependent XPOS scaling
            positions = torch.arange(freqs.shape[0], device=freqs.device)
            power = positions / self.xpos_scale_base  # grows with position
            scale = self.scale.repeat_interleave(2).repeat(self.ndims)  # head_dim
            freqs = freqs * (scale ** power.unsqueeze(-1))

        return freqs

    def _get_cached_or_compute_freqs(self, use_cls=True):
        N_patches = self.patch_coords.shape[0]
        #N_total = 1 + N_patches
        #N_total = 0 + N_patches
        N_total = (1 + N_patches) if use_cls else N_patches

        if N_total <= self.cached_freqs_len:
            return self.cached_freqs[:N_total]

        #pos_coords = self._get_full_coords()
        #pos_coords = self._get_full_coords_no_cls()
        pos_coords = self._get_full_coords() if use_cls else self._get_full_coords_no_cls()

        freqs = self._compute_freqs(pos_coords)

        if N_total <= self.cache_max_len:
            self.cached_freqs[:N_total] = freqs.detach()
            self.cached_freqs_len = N_total

        return freqs

    def _compute_freqs_dynamic(self, N):
        # Check cache first
        if N in self.cached_lengths:
            return self.cached_freqs[:N]

        # Compute freqs for this length
        pos_coords = torch.arange(N, device=self.cached_freqs.device).unsqueeze(-1)  # (N,1)
        freqs_axes = []
        for inv_freq in self.inv_freqs:
            theta = pos_coords.float() @ inv_freq.unsqueeze(0)
            theta = theta.repeat(1, 2)
            freqs_axes.append(theta)
        freqs = torch.cat(freqs_axes, dim=-1)  # (N, head_dim)

        # Cache if it fits
        if N <= self.cache_max_len:
            self.cached_freqs[:N] = freqs.detach()
            self.cached_lengths.add(N)

        return freqs

    def apply_to(self, x):
        B, n_heads, N, head_dim = x.shape
        freqs = self._compute_freqs_dynamic(N)  # compute or fetch cached
        freqs = freqs[None, None, :, :]         # (1,1,N,D)
        return apply_rotary_emb(freqs, x)


    def forward(self, q, k, use_cls=False):
        """
        q, k: (B, n_heads, N_total, head_dim)
        Returns rotated q, k
        """
        freqs = self._get_cached_or_compute_freqs(use_cls)[None, None, :, :]  # (1,1,N,D)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        return q, k
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, spatial_shape):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbeddingND(
            spatial_shape,
            self.head_dim,
            learned_freq=False,
            use_xpos=True,
            xpos_scale_base=spatial_shape[0] * 2,      # proper XPOS base
            theta_rescale_factor=1.0    # NTK scaling
        )

    def forward(self, x, use_cls=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k, use_cls)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
# =========================
# 1 Multi-BBH Toy Dataset
# =========================
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform

t_window = 1.0  # seconds

class MultiBBHToyDataset(Dataset):
    def __init__(self, batch_size=100, signal_length=2048, K=3, fs=2048,
                 f_lower=5, seed=42):

        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.batch_size = batch_size
        self.signal_length = signal_length
        self.K = K
        self.fs = fs
        self.f_lower = f_lower

        self.signals = torch.zeros(batch_size, signal_length, dtype=torch.float32)
        self.theta = []                       # True parameters [m1, m2, q, t0] per BBH

        t_margin = 5 # extra seconds before and after all signals
        noise_std = 0.1
        target_fs = 256
        target_dt = 1.0 / target_fs

        for i in range(batch_size):
            sample_params = []

            # Storage for shifted signals
            shifted_signals = []
            start_times = []
            end_times = []

            desired_merger_times = np.random.uniform(-1.0, 1.0, size=K)

            # --- Single loop: generate, resample, shift ---
            for k in range(K):
                desired_merger_time = desired_merger_times[k]

                m1 = np.random.uniform(100, 305)
                m2 = np.random.uniform(5, m1)  # m2 ≤ m1
                q = m2 / m1
                t0 = desired_merger_time#np.random.uniform(0, signal_length / fs)
                sample_params.append([m1, m2, q, t0])

                # Generate waveform
                hp, _ = get_td_waveform(approximant="IMRPhenomXPHM",
                                        mass1=m1, mass2=m2,
                                        f_lower=f_lower,
                                        delta_t=1.0/2048)
                # Resample
                hp = resample_to_delta_t(hp, target_dt)

                """
                # --- Clip waveform around its peak ---
                peak_idx = np.argmax(np.abs(hp.numpy()))
                window = int(t_window / hp.delta_t) // 2     # 20 seconds total window (-10 to +10)
                start = max(0, peak_idx - window)
                end   = min(len(hp), peak_idx + window)
                hp = hp[start:end]

                if len(hp) < 2*window:
                    pad = 2*window - len(hp)
                    hp = TimeSeries(np.pad(hp, (0, pad)), delta_t=hp.delta_t)
                """

                # Shift to desired merger time
                merger_index = np.argmax(np.abs(hp))
                peak_time_original = hp.sample_times[merger_index]
                time_shift = desired_merger_time - peak_time_original
                hp_shifted = TimeSeries(hp, delta_t=hp.delta_t, epoch=hp.start_time + time_shift)

                # Convert to tensor and normalize
                template_tensor = torch.tensor(hp_shifted, dtype=torch.float32)
                template_tensor = (template_tensor - template_tensor.mean()) / (template_tensor.std() + 1e-8)

                # Store for later placement
                shifted_signals.append((hp_shifted, template_tensor))
                start_times.append(hp_shifted.sample_times[0])
                end_times.append(hp_shifted.sample_times[-1])

            # --- Define global buffer ---
            global_start = min(start_times) - t_margin
            global_end   = max(end_times) + t_margin
            global_time = np.arange(global_start, global_end, target_dt)
            global_len = len(global_time)
            superposed_signal = torch.zeros(global_len, dtype=torch.float32)

            plt.figure(figsize=(14,6))

            # --- Place each waveform into global buffer and plot ---
            for j, (hp_shifted, template_tensor) in enumerate(shifted_signals):
                idx_start = int(round((hp_shifted.sample_times[0] - global_start) / target_dt))
                idx_end   = idx_start + len(template_tensor)
                idx_end = min(idx_end, global_len)
                tensor_end = idx_end - idx_start

                superposed_signal[idx_start:idx_end] += template_tensor[:tensor_end]
                if(i==0):
                    plt.plot(hp_shifted.sample_times, template_tensor.numpy(),
                            label=f'Signal {j+1}, merger {desired_merger_times[j]:.2f}s')
                    plt.xlim(-t_window, t_window)
                    plt.xlabel("Time [s]")
                    plt.ylabel("Amplitude")
                    plt.title(f"Time-domain waveform")
                    plt.axvline(desired_merger_times[j], color='r', linestyle='--', label='Merger Peak')


            if(i==0):
                plt.legend()
                plt.grid(True)
                plt.savefig("Time_domain_waveforms.png", dpi=300, bbox_inches='tight')
                plt.show()

            # --- Add noise and plot superposition ---
            noisy_signal = superposed_signal + torch.randn_like(superposed_signal) * 0.3 * superposed_signal.abs().max()
            if(i==0):
                plt.plot(global_time, superposed_signal.numpy(), color='k', linewidth=2, label='Superposed')
                #plt.plot(global_time, noisy_signal.numpy(), color='r', alpha=0.5, linewidth=2, label='Superposed + noise')

                plt.xlim(-t_window, t_window)
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.title("Multiple Signals with Different Merger Times + Superposed Noise")
                plt.legend()
                plt.grid(True)
                plt.savefig("Superposed_Noise.png", dpi=300, bbox_inches='tight')
                plt.show()


            # Find the peak of the superposed signal
            peak_idx = torch.argmax(torch.abs(noisy_signal))
            
            # Compute crop indices
            start_idx = peak_idx - signal_length // 2
            end_idx = peak_idx + signal_length // 2

            # Initialize a fixed-size tensor
            cropped_signal = torch.zeros(signal_length, dtype=noisy_signal.dtype)

            # Compute source indices (within the noisy_signal)
            src_start = max(0, start_idx)
            src_end   = min(len(noisy_signal), end_idx)

            # Compute target indices (within cropped_signal)
            tgt_start = max(0, -start_idx)                 # if start_idx < 0
            tgt_end   = tgt_start + (src_end - src_start)

            # Copy the relevant slice
            cropped_signal[tgt_start:tgt_end] = noisy_signal[src_start:src_end]

            # Normalize
            #final_signal = (cropped_signal - cropped_signal.mean()) / (cropped_signal.std() + 1e-8)

            # Time axis for plotting
            time_axis = np.linspace(-t_window, t_window, signal_length)

            self.signals[i] = cropped_signal
            #self.signals[i] = final_signal

            if(i==0):
                plt.plot(time_axis, cropped_signal.numpy(), color='k', linewidth=2, label='final_signal zero mean normalisation')
                #plt.plot(time_axis, final_signal.numpy(), color='k', linewidth=2, label='final_signal zero mean normalisation')

                plt.xlim(-t_window, t_window)
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.title("final_signal zero mean normalisation Times + Superposed Noise")
                plt.legend()
                plt.grid(True)
                plt.savefig("final_signal_norm.png", dpi=300, bbox_inches='tight')
                #plt.show()

            self.theta.append(sample_params)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        x = self.signals[idx]               # [L]
        theta = np.array(self.theta[idx], dtype=np.float32)   # convert list -> array      # [K,4]  (m1, m2, q, t0)

        # ---- Normalize parameters ----
        # masses: m1 in [100,310], m2 in [5,310]
        theta[:, 0] = (theta[:, 0] - 5) / 305   # m1_norm ∈ [0,1]
        theta[:, 1] = (theta[:, 1] - 5) / 305     # m2_norm ∈ [0,1]

        # mass ratio q already in [0,1] → no change
        theta[:, 2] = theta[:, 2]

        # merger time t0 ∈ [0, T_obs = signal_length/fs]
        theta[:, 3] = (theta[:, 3] + t_window) / (2 * t_window)

        return x, torch.tensor(theta, dtype=torch.float32)


# =========================
# 2 Transformer Model
# =========================
# =========================
# Patch embedding (Conv1D) and Transformer encoder block
# =========================
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, signal_length, patch_size, n_channels=1):
        super().__init__()
        assert signal_length % patch_size == 0, "patch_size must divide signal_length"
        self.conv = nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, L]
        x = self.conv(x)             # [B, d_model, n_patches]
        x = x.transpose(1, 2)        # [B, n_patches, d_model]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, spatial_shape, mlp_ratio=4.):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, spatial_shape)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, n_patches, d_model]
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


# =========================
# Bayesian Transformer (encoder-only)
# =========================
class BayesianTransformer(nn.Module):
    def __init__(self, n_params=4, d_model=128, signal_length=512, patch_size=16,
                 n_channels=1, n_heads=4, n_layers=4):
        super().__init__()
        self.n_params = n_params
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = signal_length // patch_size
        self.spatial_shape = (self.n_patches,)
        self.patch_embed = PatchEmbedding(d_model, signal_length, patch_size, n_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))  # learnable pos emb

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, self.spatial_shape)
            for _ in range(n_layers)
        ])

        # pooling and heads
        self.pool = nn.AdaptiveAvgPool1d(1)  # or use mean over patches
        self.mean_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_params)
        )
        self.logvar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_params)
        )

    def forward(self, x):
        """
        x: [B, C, L]
        returns:
          mean: [B, n_params]
          var: [B, n_params]  (positive)
        """
        B = x.size(0)
        h = self.patch_embed(x)                  # [B, n_patches, d_model]
        h = h + self.pos_embed                   # broadcast

        for layer in self.encoder_layers:
            h = layer(h)

        # pool over patches
        h_mean = h.mean(dim=1)                   # [B, d_model]

        mean = self.mean_head(h_mean)            # [B, n_params]
        logvar = self.logvar_head(h_mean)        # [B, n_params]
        var = F.softplus(logvar) + 1e-6          # ensure positivity and numerical stability

        return mean, var

def sample_posterior(mean, var, n_samples=100, n_signals=3):
    """
    Sample from Gaussian posterior while enforcing physical constraints:
      - mass ratio q ∈ [0,1]
      - merger time t_merger ∈ [0, T_obs]

    mean: [B, n_params]
    var:  [B, n_params]
    n_samples: number of draws per batch element
    T_obs: observation duration for t_merger
    n_signals: number of BBHs in the signal

    Returns: [n_samples*B, n_params]
    """
    B, P = mean.shape
    eps = torch.randn(n_samples, B, P, device=mean.device)

    # Sample
    samples = mean.unsqueeze(0) + eps * torch.sqrt(var).unsqueeze(0)
    samples = samples.reshape(-1, P)

    return samples.cpu().numpy()



# =========================
# 3 Spectral clustering
# =========================
def cluster_posterior(samples, n_clusters=2):
    """
    samples: np.ndarray [n_samples, n_params]
    returns list of clusters arrays and labels
    """
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    affinity='nearest_neighbors',
                                    assign_labels='kmeans',
                                    random_state=42,
                                    n_jobs=-1)
    labels = clustering.fit_predict(samples)
    clustered = [samples[labels == k] for k in range(n_clusters)]
    return clustered, labels

# =========================
# 4 Hungarian algorithm to reorder clusters
# =========================
def get_reference_samples(theta_all, n_batch_size, n_signals):
    """
    Automatically split theta_all[:n_batch_size] into n_signals reference groups.
    """
    ref_samples = []
    for i in range(n_signals):
        # Select rows corresponding to signal i
        ref_samples.append(theta_all[:n_batch_size].numpy()[i::n_signals])
    return ref_samples

def reorder_clusters_to_reference(clustered_samples, reference_samples_per_signal):
    """
    Reorder clusters to match reference signals using Hungarian algorithm.

    clustered_samples : list of np.ndarray [n_samples_in_cluster, n_params]
    reference_samples_per_signal : list of np.ndarray [n_samples, n_params]
    """
    n_clusters = len(clustered_samples)
    n_signals = len(reference_samples_per_signal)
    if n_clusters != n_signals:
        raise ValueError("Number of clusters must equal number of reference signals.")

    cluster_medians = np.array([np.median(c, axis=0) for c in clustered_samples])
    reference_medians = np.array([np.median(ref, axis=0) for ref in reference_samples_per_signal])

    cost_matrix = np.zeros((n_clusters, n_signals))
    for i in range(n_clusters):
        for j in range(n_signals):
            cost_matrix[i, j] = np.sum(np.abs(cluster_medians[i] - reference_medians[j]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    reordered_samples = [clustered_samples[i] for i in col_ind]
    return reordered_samples

# =========================
# 5 Training loop
# =========================
signal_length = 2048
batch_size    = 10
n_epochs      = 150
n_signals     = 3
n_params      = n_signals * 4
n_draws=100

ds = MultiBBHToyDataset(batch_size=batch_size,
                        signal_length=signal_length,
                        K=n_signals,
                        fs=2048,
                        f_lower=5,
                        seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BayesianTransformer(
    n_params=n_params,
    d_model=128,
    signal_length=signal_length,
    patch_size=16,
    n_channels=1,
    n_heads=4,
    n_layers=4,
).to(device)

opt = Adam(model.parameters(), lr=3e-4)

# -------------------------
# Store metrics
# -------------------------
train_losses = []
val_losses   = []

# A simple 20% validation split from your dataset
val_size = int(0.2 * len(ds))
train_size = len(ds) - val_size
print("val_size :", val_size)
print("train_size :", train_size)

train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=train_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=val_size, shuffle=False)
       
# =========================
# TRAINING LOOP
# =========================
for epoch in range(n_epochs):
    # ====== TRAIN ======
    model.train()
    running_loss = 0
    val_running_loss = 0
    for x, theta in train_loader:
        B = x.shape[0]
        x = x.unsqueeze(1).to(device)  # [B,1,L]
        theta = theta.reshape(theta.shape[0],-1).to(device)  # [B,K*4]
        mu, var = model(x)
        nll = 0.5 * (((theta - mu)**2)/var + torch.log(var)).sum(dim=1).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
        running_loss += nll.item()
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # ====== VALIDATION ======
    model.eval()
    all_pred_samples = []
    all_true_labels = []

    with torch.no_grad():
        for x, theta in val_loader:
            B = x.shape[0]
            x = x.unsqueeze(1).to(device)
            theta_arr = theta.reshape(B, n_signals, 4).cpu().numpy()
            theta_flat = theta.reshape(theta.shape[0],-1).to(device)  # [B,K*4]

            mu, var = model(x)

            val_nll = 0.5 * (((theta_flat - mu)**2) / var + torch.log(var)).sum(dim=1).mean()

            val_running_loss += val_nll.item()

            samples = sample_posterior(mu, var, n_samples=n_draws, n_signals=n_signals)

            # Flatten samples per BBH
            for k in range(n_signals):
                samples_k = samples[:, k*4:(k+1)*4]
                all_pred_samples.append(samples_k)
                all_true_labels.append(np.full(samples_k.shape[0], k))

    epoch_val_loss = val_running_loss / len(val_loader)

    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} | "f"Train NLL={epoch_train_loss:.4f} | Val NLL={epoch_val_loss:.4f}")

    # Concatenate all BBHs
    all_pred_samples = np.concatenate(all_pred_samples)
    all_true_labels = np.concatenate(all_true_labels)

    # ----- Cluster posterior samples -----
    clustered_samples, _ = cluster_posterior(all_pred_samples, n_clusters=n_signals)

    # ----- Hungarian reorder -----
    # all_true_labels_per_bbh: list of arrays, one per BBH
    true_samples_per_bbh = []
    for k in range(n_signals):
        # select the theta values for BBH k
        true_samples_per_bbh.append(theta_arr[:, k, :])  # shape [B, 4]

    # Hungarian reorder using true BBH samples as reference
    clustered_reordered = reorder_clusters_to_reference(clustered_samples, true_samples_per_bbh)


    # ----- Assign predicted labels -----
    pred_labels = np.zeros_like(all_true_labels)
    start_idx = 0
    for cluster_idx, cluster in enumerate(clustered_reordered):
        n_cluster_samples = cluster.shape[0]
        pred_labels[start_idx:start_idx+n_cluster_samples] = cluster_idx
        start_idx += n_cluster_samples

# =========================
# PLOT NLL
# =========================
epochs_range = range(1, n_epochs+1)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label="Train NLL")
plt.plot(epochs_range, val_losses, label="Val NLL")
plt.xlabel("Epochs"); plt.ylabel("NLL"); plt.title("Training NLL")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig("Nll.png", dpi=300, bbox_inches='tight')
#plt.show()

# -----------------------------
# Compute confusion matrix for the last epoch
# -----------------------------
cm = confusion_matrix(all_true_labels, pred_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Plot confusion matrix counts
# Create annotation strings combining count and normalized value
labels_annotation = np.empty_like(cm).astype(str)
n_rows, n_cols = cm.shape
for i in range(n_rows):
    for j in range(n_cols):
        labels_annotation[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"

labels = [f"BBH_{i+1}" for i in range(n_signals)]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=labels_annotation, fmt='', cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Cluster")
plt.ylabel("True BBH")
plt.title("Posterior-aware Confusion Matrix (Counts + Normalized)")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

#plt.show()

n_draws=5000

# =========================
# 6 Posterior & Clustering
# =========================
x_obs = torch.stack([ds[i][0] for i in range(len(ds))]).unsqueeze(1).to(device)
theta_all = torch.tensor(ds.theta).reshape(len(ds),-1)
theta_batch = theta_all[:batch_size].numpy()   # [B, params]

with torch.no_grad():
    mean,var = model(x_obs)
samples = sample_posterior(mean,var,n_samples=n_draws, n_signals=n_signals)

clustered,_ = cluster_posterior(samples,n_clusters=n_signals)

# Hungarian reorder
reference_samples_per_signal = get_reference_samples(theta_all, n_batch_size=batch_size, n_signals=n_signals)
clustered_reordered = reorder_clusters_to_reference(clustered, reference_samples_per_signal)

true_values = []
for ref in reference_samples_per_signal:
    med_ref = np.median(ref, axis=0)
    true_values.append(med_ref)
true_values = np.concatenate(true_values) # shape [n_params]

# =========================
# 7 Corner Plot
# =========================
param_names = ["m1", "m2", "q", "t"]
labels_names = []

for k in range(n_signals):  # loop over each BBH
    for pname in param_names:
        labels_names.append(f"{pname}_{k+1}")

colors = ["red", "blue", "green", "orange", "purple"]

# convert to numpy
clustered_np = [
    c if isinstance(c, np.ndarray) else c.detach().numpy()
    for c in clustered_reordered
]

# Loop over BBHs
for k_bbh in range(n_signals):
    m1_idx = 4*k_bbh     # because labels are [m1_1, m2_1, q_1, t_1, m1_2, ...]
    m2_idx = 4*k_bbh +1
    t_idx = 4*k_bbh + 2
    q_idx = 4*k_bbh + 3

    # Denormalize true values
    true_values[m1_idx] = true_values[m1_idx]*305 + 5
    true_values[m2_idx] = true_values[m2_idx]*305 + 5
    true_values[t_idx] = true_values[t_idx] * (2*t_window) - t_window


    # Denormalize cluster samples
    for cluster in clustered_np:
        cluster[:, m1_idx] = cluster[:, m1_idx]*305 + 5
        cluster[:, m2_idx] = cluster[:, m2_idx]*305 + 5
        
        cluster[:, m1_idx] = np.clip(cluster[:, m1_idx], 5, 310)
        cluster[:, m2_idx] = np.clip(cluster[:, m2_idx], 5, 310)
        
        cluster[:, t_idx] = cluster[:, t_idx] * (2*t_window) - t_window

        cluster[:, q_idx] = np.clip(cluster[:, q_idx], 0, 1.)



# base fig for cluster 0
fig = corner.corner(
    clustered_np[0],
    labels=labels_names,
    color=colors[0],
    show_titles=True,
    plot_datapoints=True,
    fill_contours=False
)

# add other clusters
for k in range(1, len(clustered_np)):
    corner.corner(
        clustered_np[k], fig=fig,
        color=colors[k],
        plot_datapoints=True,
        fill_contours=False
    )

n_params = clustered_np[0].shape[1]
# overlay medians/stds
axes = np.array(fig.axes).reshape((n_params, n_params))

for k, cluster in enumerate(clustered_np):

    medians = np.median(cluster, axis=0)
    stds = np.std(cluster, axis=0)
    color = colors[k]

    for i in range(n_params):

        ax = axes[i, i]
        ax.axvline(true_values[i], color=color, linestyle="-", lw=2)
        ax.axvline(medians[i], color=color, linestyle=":", lw=2)
        ax.axvline(medians[i] + stds[i], color=color, linestyle=":", lw=1)
        ax.axvline(medians[i] - stds[i], color=color, linestyle=":", lw=1)

        for j in range(i):
            ax2 = axes[i, j]
            ax2.axvline(true_values[j], color=color, linestyle="-", lw=1)
            ax2.axhline(true_values[i], color=color, linestyle="-", lw=1)

plt.savefig("corner_plot.png", dpi=300, bbox_inches='tight')
#plt.show()
