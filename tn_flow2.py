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
from sklearn.metrics import confusion_matrix

from typing import Tuple
import seaborn as sns
from scipy.stats import wasserstein_distance

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

class MultiBBHToyDataset(Dataset):
    def __init__(self, num_samples=100, signal_length=2048, K=3, fs=1024,
                 f_lower=5, seed=42, store_sample_idx=None):
        """
        store_sample_idx: int or None
            If an integer index is provided, stores the individual BBH waveforms
            of that sample in self.stored_sample_individuals for inspection.
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.signal_length = signal_length
        self.K = K
        self.fs = fs
        self.f_lower = f_lower

        self.signals = torch.zeros(num_samples, signal_length, dtype=torch.float32)
        self.theta = []                       # True parameters [m1, m2, q, t0] per BBH
        self.individual_waveforms = []        # Store all individual BBH waveforms
        self.stored_sample_individuals = None # Optional storage for a chosen sample
        self.stored_sample_idx = store_sample_idx

        for i in range(num_samples):
            sample_signal = torch.zeros(signal_length, dtype=torch.float32)
            sample_params = []
            sample_individuals = []

            for k in range(K):
                m1 = np.random.uniform(20, 60)
                m2 = np.random.uniform(5, m1)
                q = m2 / m1
                t0 = np.random.uniform(0, signal_length / fs)
                sample_params.append([m1, m2, q, t0])

                # Generate waveform once
                hp, _ = get_td_waveform(approximant='IMRPhenomXPHM',
                                        mass1=m1, mass2=m2,
                                        delta_t=1.0/fs,
                                        f_lower=f_lower)
                h = torch.tensor(hp.data, dtype=torch.float32)

                # Pad or truncate
                if len(h) < signal_length:
                    h = torch.cat([torch.zeros(signal_length - len(h)), h])
                else:
                    h = h[-signal_length:]

                # Apply random scaling
                h_scaled = h * np.random.uniform(0.5, 1.0)

                sample_signal += h_scaled
                sample_individuals.append(h_scaled)

            # Add noise and normalize
            noise_scale = 0.05 * sample_signal.abs().max()
            sample_signal += noise_scale * torch.randn_like(sample_signal)
            sample_signal = (sample_signal - sample_signal.mean()) / (sample_signal.std() + 1e-20)

            self.signals[i] = sample_signal
            self.theta.append(sample_params)
            self.individual_waveforms.append(sample_individuals)

            # If this is the sample to inspect, store its individual waveforms
            if store_sample_idx is not None and i == store_sample_idx:
                self.stored_sample_individuals = sample_individuals

        self.theta = np.array(self.theta, dtype=np.float32)  # shape: [N, K, 4]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.signals[idx], torch.tensor(self.theta[idx], dtype=torch.float32)

    def plot_stored_sample(self):
        """
        Plot the individual BBH signals, superposed waveform, and noisy signal
        of the stored sample, if available.
        """
        if self.stored_sample_individuals is None or self.stored_sample_idx is None:
            print("No stored sample. Please pass store_sample_idx when initializing the dataset.")
            return

        idx = self.stored_sample_idx
        sample_signal = self.signals[idx]
        individual_signals = self.stored_sample_individuals
        sample_params = self.theta[idx]

        time = np.arange(self.signal_length) / self.fs
        superposed_signal = torch.stack(individual_signals).sum(dim=0).numpy()

        # Scale individual signals to match the noisy + normalized signal
        scaled_individuals = [h / superposed_signal.max() * sample_signal.abs().max() for h in individual_signals]

        plt.figure(figsize=(14,6))
        for k, h in enumerate(scaled_individuals):
            plt.plot(time, h.numpy(), label=f'BBH {k+1}')
        plt.plot(time, superposed_signal, color='k', linestyle='--', label='Superposed signal')
        plt.plot(time, sample_signal.numpy(), color='r', alpha=0.5, label='Noisy + normalized signal')
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Sample {idx}: Multi-BBH Signals")
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Sample {idx} BBH parameters [m1, m2, q, t0] per BBH:\n", sample_params)


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

def sample_posterior(mean, var, n_samples=100, T_obs=16.0, n_signals=3):
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

    # Optional: transform mean to stay positive
    mean = mean.clone()
    for k in range(n_signals):
        q_idx = k*4 + 2
        t_idx = k*4 + 3
        mean[:, q_idx] = F.softplus(mean[:, q_idx])           # ensure q>0
        mean[:, t_idx] = F.softplus(mean[:, t_idx])           # ensure t>0

    # Sample
    samples = mean.unsqueeze(0) + eps * torch.sqrt(var).unsqueeze(0)
    samples = samples.reshape(-1, P)

    # Clamp to physical ranges
    for k in range(n_signals):
        q_idx = k*4 + 2
        t_idx = k*4 + 3
        samples[:, q_idx] = torch.clamp(samples[:, q_idx], 0.0, 1.0)
        samples[:, t_idx] = torch.clamp(samples[:, t_idx], 0.0, T_obs)

    return samples


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
                                    random_state=42)
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
num_samples   = 1
signal_length = 2048
batch_size    = 1
n_epochs      = 400
n_signals     = 3
n_params      = n_signals * 4
n_draws=2000

ds = MultiBBHToyDataset(num_samples=num_samples,
                        signal_length=signal_length,
                        K=n_signals,
                        fs=2048,
                        f_lower=5,
                        seed=42,
                        store_sample_idx=0)

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
train_mae    = []
val_mae      = []

# A simple 20% validation split from your dataset
val_size = int(0.2 * len(ds))
train_size = len(ds) - val_size
print("val_size :", val_size)
print("train_size :", train_size)

train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

ds.plot_stored_sample()
# -------------------------
# Training Loop
# -------------------------

# METRICS STORAGE
train_losses = []
val_accuracies = []

# =========================
# TRAINING LOOP
# =========================
for epoch in range(n_epochs):
    # ====== TRAIN ======
    model.train()
    running_loss = 0
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

            mu, var = model(x)
            samples = sample_posterior(mu, var, n_samples=n_draws,
                                       T_obs=16, n_signals=n_signals).cpu().numpy()

            # Flatten samples per BBH
            for k in range(n_signals):
                samples_k = samples[:, k*4:(k+1)*4]
                all_pred_samples.append(samples_k)
                all_true_labels.append(np.full(samples_k.shape[0], k))

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

    # ----- Compute accuracy -----
    accuracy = 100 * np.mean(pred_labels == all_true_labels)
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{n_epochs} | Train NLL={epoch_train_loss:.4f} | Val Accuracy={accuracy:.2f}%")

# =========================
# PLOT NLL AND ACCURACY
# =========================
epochs_range = range(1, n_epochs+1)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label="Train NLL")
plt.xlabel("Epochs"); plt.ylabel("NLL"); plt.title("Training NLL")
plt.grid(True); plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)"); plt.title("Posterior-aware Accuracy")
plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()

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
plt.show()


# =========================
# 6 Posterior & Clustering
# =========================
x_obs = torch.stack([ds[i][0] for i in range(len(ds))]).unsqueeze(1).to(device)
theta_all = torch.tensor(ds.theta).reshape(len(ds),-1)
theta_batch = theta_all[:batch_size].numpy()   # [B, params]

with torch.no_grad():
    mean,var = model(x_obs)
samples = sample_posterior(mean,var,n_samples=n_draws).cpu().numpy()

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

plt.show()

#
# =========================
# 8 P-P plot
# =========================

def qq_plot(samples, truth, label="QQ"):
    samples = np.asarray(samples)
    truth   = np.asarray(truth)

    # Number of quantile points to compare
    m = min(len(samples), len(truth))

    # Quantile levels between 0 and 1
    q = np.linspace(0, 1, m)

    # Compute quantiles
    x = np.quantile(samples, q)   # posterior quantiles
    y = np.quantile(truth,   q)   # truth quantiles

    # Normalize both to [0,1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Plot dots + line
    plt.plot(x_norm, y_norm, 'o-', markersize=4, label=label)


N = theta_all.shape[0]
n_params = theta_all.shape[1]

# Draw posterior samples
posterior_samples_list = []
with torch.no_grad():
    for i in range(N):
        mean_i = mean[i:i+1]
        var_i  = var[i:i+1]
        samples_i = sample_posterior(mean_i, var_i, n_samples=n_draws)
        posterior_samples_list.append(samples_i.cpu().numpy())   # shape [n_draws, n_params]

plt.figure(figsize=(8,8))

for j in range(n_params):

    # flatten all posterior samples for this parameter
    posterior_flat = np.concatenate([posterior_samples_list[i][:, j]
                                     for i in range(N)])
    truth = theta_all[:, j].cpu().numpy()

    W = wasserstein_distance(posterior_flat, truth)

    # Label with metric
    label = f"{labels_names[j]} (W={W:.3f})"

    qq_plot(posterior_flat, truth, label=label)

# diagonal line (0–1 in normalized space)
plt.plot([0,1], [0,1], 'k--', label="y=x")

plt.title("QQ Plot: Posterior vs True Distribution (normalized)")
plt.xlabel("Posterior quantiles (normalized)")
plt.ylabel("True quantiles (normalized)")
plt.legend()
plt.grid()
plt.show()

# Violin Plot

# Assume:
# posterior_samples_list: list of N arrays [n_samples, n_params]
# theta_all: [N, n_params]
# labels_names: ["m1_1","m2_1","q_1","t_1",..., "t_3"]

# 1. Flatten posterior samples for each parameter across all events
n_params = theta_all.shape[1]
posterior_flat = [
    np.concatenate([posterior_samples_list[i][:, j] for i in range(len(posterior_samples_list))])
    for j in range(n_params)
]

# 2. Prepare data for seaborn
data_for_violin = []
param_labels = []
for j in range(n_params):
    data_for_violin.extend(posterior_flat[j])
    param_labels.extend([labels_names[j]] * len(posterior_flat[j]))

data_for_violin = np.array(data_for_violin)
param_labels = np.array(param_labels)

# 3. Plot violin plot
plt.figure(figsize=(16,6))
sns.violinplot(x=param_labels, y=data_for_violin, inner="quartile", palette="Set3")

# 4. Overlay true values
true_flat = theta_all.reshape(-1, n_params).numpy() if isinstance(theta_all, torch.Tensor) else theta_all
for j in range(n_params):
    plt.scatter([j]*true_flat.shape[0], true_flat[:, j], color='red', s=20, label="True value" if j==0 else "")

plt.xticks(rotation=45)
plt.xlabel("Parameters")
plt.ylabel("Posterior Samples")
plt.title("Violin Plot of Posterior Samples with True Values for 12 BBH Parameters")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
