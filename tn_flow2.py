import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import corner
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_1samp
import numpy as np
from MySelfAttention import SelfAttention

torch.manual_seed(42)
np.random.seed(42)

# =========================
# 1️⃣ Generate correlated data
# =========================
N = 1000
n_params = 4  # number of theta parameters

theta1 = torch.rand(N, 1) * 2 - 1
theta2 = -1.0 * theta1 + 0.2 * torch.randn(N, 1)
theta3 = 1.0 * theta2 + 0.2 * torch.randn(N, 1)
theta4 = -1.0 * theta3 + 0.2 * torch.randn(N, 1)

theta_all = torch.cat([theta1, theta2, theta3, theta4], dim=1)  # [N,4]

# Observation: noisy measurement
x_obs = theta_all + 0.1 * torch.randn_like(theta_all)  # [N,4]


# =========================
# 4️⃣ Transformer Encoder Layer with custom Self-Attention
# =========================
class TransformerEncoderLayerCustom(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, spatial_shape):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, spatial_shape)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

# =========================
# 5️⃣ Bayesian Transformer with Rotary
# =========================
class BayesianTransformer(nn.Module):
    def __init__(self, n_params, d_model=64, n_heads=2, n_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(1, d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerCustom(d_model, n_heads, dim_feedforward=128, spatial_shape=(n_params,))
            for _ in range(n_layers)
        ])
        self.mean_fc = nn.Linear(d_model, 1)
        self.logvar_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, N, 1]
        h = F.relu(self.input_fc(x))
        for layer in self.encoder_layers:
            h = layer(h)
        mean = self.mean_fc(h).squeeze(-1)
        logvar = self.logvar_fc(h).squeeze(-1)
        var = F.softplus(logvar) + 1e-6
        return mean, var

# =========================
# 4️⃣ Posterior sampling
# =========================
def sample_posterior(mean, var, n_samples=500):
    eps = torch.randn(n_samples, mean.size(0), mean.size(1))
    samples = mean.unsqueeze(0) + eps * torch.sqrt(var).unsqueeze(0)
    samples = samples.reshape(-1, mean.size(1))
    return samples

# =========================
# 5️⃣ Spectral clustering
# =========================
def cluster_posterior(samples, n_clusters=2):
    if torch.is_tensor(samples):
        samples = samples.detach().numpy()
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        assign_labels='kmeans',
        random_state=42
    )
    labels = clustering.fit_predict(samples)
    clustered_samples = [samples[labels == k] for k in range(n_clusters)]
    return clustered_samples, labels

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

# =========================
# 6️⃣ Training loop
# =========================
model = BayesianTransformer(n_params=n_params)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 128
epochs = 10

for epoch in range(epochs):
    perm = torch.randperm(N)
    running_loss = 0.0
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        x_batch = x_obs[idx]
        theta_batch = theta_all[idx]

        mean, var = model(x_batch)
        nll = 0.5 * (((theta_batch - mean) ** 2) / var + torch.log(var)).sum(dim=1)
        loss = nll.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} | NLL: {running_loss / (N/batch_size):.4f}")

# =========================
# 7️⃣ Sample posterior and cluster
# =========================
n_samples = 100
n_batch_size = 100
n_signals = 2

mean, var = model(x_obs[:n_batch_size])
posterior_samples = sample_posterior(mean, var, n_samples=n_samples)

clustered_samples, labels = cluster_posterior(posterior_samples, n_clusters=n_signals)

reference_samples_per_signal = get_reference_samples(theta_all, n_batch_size, n_signals)

clustered_samples = reorder_clusters_to_reference(clustered_samples, reference_samples_per_signal=reference_samples_per_signal)

# =========================
# 8️⃣ Corner plot
# =========================
labels_names = [f"theta{i+1}" for i in range(n_params)]
colors = ["red", "blue", "green", "orange", "purple"]

clustered_samples_np = [c if isinstance(c, np.ndarray) else c.detach().numpy() for c in clustered_samples]
true_values = theta_all[:500].mean(dim=0).numpy()

fig = corner.corner(
    clustered_samples_np[0],
    labels=labels_names,
    color=colors[0],
    show_titles=True,
    plot_datapoints=True,
    fill_contours=False
)

for i, cluster in enumerate(clustered_samples_np[1:], start=1):
    corner.corner(cluster, fig=fig, color=colors[i], plot_datapoints=True, fill_contours=False)

axes = np.array(fig.axes).reshape((n_params, n_params))
for k, cluster in enumerate(clustered_samples_np):
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

# =========================
# 9️⃣ Posterior predictive P-P plot with KS
# =========================
posterior_samples_list = []
for i in range(N):
    mean_i, var_i = model(x_obs[i:i+1])
    samples_i = sample_posterior(mean_i, var_i, n_samples=n_samples)
    posterior_samples_list.append(samples_i)

p_values = np.zeros((N, n_params))
true_params = theta_all.numpy()

for i in range(N):
    samples = posterior_samples_list[i]
    if torch.is_tensor(samples):
        samples = samples.detach().numpy()
    for j in range(n_params):
        p_values[i, j] = np.mean(samples[:, j] < true_params[i, j])

plt.figure(figsize=(6,6))
for j in range(n_params):
    sorted_p = np.sort(p_values[:, j])
    cdf = np.arange(1, N+1)/N
    ks_val = ks_1samp(p_values[:, j], lambda x: x).pvalue
    plt.plot(sorted_p, cdf, marker='o', linestyle='-', label=f'theta{j+1} (KS={ks_val:.3f})')

plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel('p')
plt.ylabel('CDF')
plt.title('Posterior predictive check (P-P plot)')
plt.legend()
plt.grid(True)
plt.show()

ks_pvalues = [ks_1samp(p_values[:,j], lambda x: x).pvalue for j in range(n_params)]
combined_p = np.mean(ks_pvalues)
print("KS p-values per parameter:", ks_pvalues)
print("Combined p-value:", combined_p)
