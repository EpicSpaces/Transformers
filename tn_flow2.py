import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import corner
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_1samp
from MySelfAttention import SelfAttention

torch.manual_seed(42)
np.random.seed(42)

# =========================
# Dataset: continuous-θ signals
# =========================
class RandomSignalDataset(Dataset):
    """
    Each sample:
      - theta: [4] = [A, f, phi, alpha]
      - signal: [1, signal_length]
    The 'templates' field is optional; here we produce a small set of template waveforms
    that could be used with cross-attention (not used by default).
    """
    def __init__(self, num_samples=10000, signal_length=512, n_channels=1,
                 n_templates=4, noise_std=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.n_channels = n_channels
        self.n_templates = n_templates
        self.noise_std = noise_std

        t = torch.linspace(0, 1, signal_length)

        self.signals = torch.zeros(num_samples, n_channels, signal_length)
        self.thetas = torch.zeros(num_samples, 4)  # continuous targets
        self.templates = torch.zeros(num_samples, n_channels, signal_length, n_templates)

        for i in range(num_samples):
            # sample parameters
            A = 0.2 + 1.0 * torch.rand(1)          # amplitude in [0.2,1.2]
            f = 2.0 + 6.0 * torch.rand(1)          # frequency in [2,8]
            phi = 2 * np.pi * torch.rand(1)        # phase
            alpha = 0.5 * torch.rand(1)            # decay

            theta = torch.cat([A, f, phi, alpha])
            self.thetas[i] = theta

            # generate templates (bank) - small variations
            for j in range(n_templates):
                fj = f + 0.2 * (j - n_templates/2) + 0.05 * torch.randn(1)
                phij = phi + 0.1 * torch.randn(1)
                Aj = A * (1.0 + 0.05 * (j - n_templates/2))
                self.templates[i, 0, :, j] = (Aj * torch.sin(2 * np.pi * fj * t + phij)
                                               * torch.exp(-alpha * t))

            # Choose one "true" template (or use the analytic signal)
            # Here we use the analytic signal (same generator) + noise
            self.signals[i, 0] = (A * torch.sin(2 * np.pi * f * t + phi) * torch.exp(-alpha * t)
                                  + noise_std * torch.randn(signal_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return: (signal [C,L], theta [4], templates [C,L,n_templates])
        return self.signals[idx], self.thetas[idx], self.templates[idx]


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

# =========================
# 4️⃣ Posterior sampling
# =========================
def sample_posterior(mean, var, n_samples=100):
    """
    mean: [B, n_params]
    var:  [B, n_params]
    returns: [n_samples*B, n_params]
    """
    B, P = mean.shape
    eps = torch.randn(n_samples, B, P, device=mean.device)
    samples = mean.unsqueeze(0) + eps * torch.sqrt(var).unsqueeze(0)
    samples = samples.reshape(-1, P)
    return samples

# =========================
# 5️⃣ Spectral clustering
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
# 5b Hungarian algorithm to reorder clusters
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


# ===============================================================
# Posterior checks: clustering + corner + P-P plot (KS)
# ===============================================================
def posterior_checks(model, x_obs, theta_all, n_params,
                     n_signals=2, n_samples=100, batch_size=128):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -------------------------
    # 1) Pick a batch
    # -------------------------
    x_batch = x_obs[:batch_size].to(device)        # [B, signal]
    theta_batch = theta_all[:batch_size].numpy()   # [B, params]

    # -------------------------
    # 2) Posterior samples
    # -------------------------
    with torch.no_grad():
        mean, var = model(x_batch)
        posterior_samples = sample_posterior(mean, var, n_samples=n_samples)
        # posterior_samples: [n_samples*B, n_params]

    posterior_samples = posterior_samples.cpu()
    
    # -------------------------
    # 3) Cluster (e.g. 2 signals)
    # -------------------------
    clustered_samples, labels = cluster_posterior(
        posterior_samples, n_clusters=n_signals
    )

    # -------------------------
    # 4) Hungarian reorder
    # -------------------------
    reference_samples_per_signal = get_reference_samples(
        theta_all, n_batch_size=batch_size, n_signals=n_signals
    )

    clustered_samples = reorder_clusters_to_reference(
        clustered_samples, reference_samples_per_signal
    )

    # -------------------------
    # 5) Corner plot per cluster
    # -------------------------
    
    labels_names = [f"theta{i+1}" for i in range(n_params)]
    colors = ["red", "blue", "green", "orange", "purple"]

    # convert to numpy
    clustered_np = [
        c if isinstance(c, np.ndarray) else c.detach().numpy()
        for c in clustered_samples
    ]

    # "true" lines: use average of batch
    true_values = theta_batch.mean(axis=0)

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

    # overlay medians/stds
    axes = np.array(fig.axes).reshape((n_params, n_params))
    for k, cluster in enumerate(clustered_np):
        med = np.median(cluster, axis=0)
        std = np.std(cluster, axis=0)
        col = colors[k]

        for i in range(n_params):
            ax = axes[i, i]
            ax.axvline(true_values[i], color=col, linestyle="-", lw=2)
            ax.axvline(med[i], color=col, linestyle=":", lw=2)
            ax.axvline(med[i] + std[i], color=col, linestyle=":", lw=1)
            ax.axvline(med[i] - std[i], color=col, linestyle=":", lw=1)

    plt.show()

    # -------------------------
    # 6) Posterior predictive P–P plot (KS)
    # -------------------------
    N = x_obs.shape[0]
    p_values = np.zeros((N, n_params))
    true_params = theta_all.numpy()

    posterior_samples_list = []

    # draw posterior for each sample
    for i in range(N):
        mean_i, var_i = model(x_obs[i:i+1].to(device))
        samples_i = sample_posterior(mean_i, var_i, n_samples=n_samples)
        posterior_samples_list.append(samples_i.detach().cpu().numpy())

    # compute p-values
    for i in range(N):
        samples = posterior_samples_list[i]
        for j in range(n_params):
            p_values[i, j] = np.mean(samples[:, j] < true_params[i, j])

    # P-P plot
    plt.figure(figsize=(6, 6))
    ks_pvals = []

    for j in range(n_params):
        sorted_p = np.sort(p_values[:, j])
        cdf = np.arange(1, N + 1) / N
        ks_val = ks_1samp(p_values[:, j], lambda x: x).pvalue
        ks_pvals.append(ks_val)
        plt.plot(sorted_p, cdf, marker='o', linestyle='-', label=f"θ{j+1} (KS={ks_val:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("p")
    plt.ylabel("CDF")
    plt.title("Posterior predictive P–P plot")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("KS p-values per parameter:", ks_pvals)
    print("Combined (mean) KS p-value:", float(np.mean(ks_pvals)))

# =========================
# Run training + checks if script executed
# =========================

# =========================
# Training loop (example)
# =========================

# hyperparams
num_samples = 1000
signal_length = 512
batch_size = 128
n_epochs = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = RandomSignalDataset(num_samples=num_samples, signal_length=signal_length, n_templates=6, noise_std=0.12)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

model = BayesianTransformer(n_params=4, d_model=128, signal_length=signal_length,
                            patch_size=16, n_channels=1, n_heads=4, n_layers=4).to(device)

opt = Adam(model.parameters(), lr=3e-4)

for epoch in range(n_epochs):
    running_loss = 0.0
    model.train()
    for (x, theta, _) in loader:
        x = x.to(device)              # [B, 1, L]
        theta = theta.to(device)      # [B, 4]

        mean, var = model(x)          # [B,4] each
        # negative log likelihood (diagonal Gaussian)
        nll_per_sample = 0.5 * (((theta - mean) ** 2) / var + torch.log(var)).sum(dim=1)
        loss = nll_per_sample.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{n_epochs} | NLL: {avg_loss:.4f}")


# Extract all signals and thetas from dataset
x_obs = torch.stack([ds[i][0] for i in range(len(ds))])    # [N, 1, signal_length]
theta_all = torch.stack([ds[i][1] for i in range(len(ds))])  # [N, n_params]
n_params = theta_all.shape[1]

# Run posterior checks
posterior_checks(
    model,
    x_obs=x_obs,
    theta_all=theta_all,
    n_params=n_params,
    n_signals=2,      # number of clusters/signals
    n_samples=100,
    batch_size=128
)
