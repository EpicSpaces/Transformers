import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.filter import resample_to_delta_t

from torch.nn.functional import interpolate
import argparse
import json

from MySelfAttention import SelfAttention
from MyCrossAttention import CrossAttention
"""
parser = argparse.ArgumentParser(description="Load base_dir / Load filter config from JSON.")
parser.add_argument('--b', type=str, required=True, help="Path to gwfs")
parser.add_argument('--j', type=str, required=True, help="Path to config.json")
args = parser.parse_args()

base_dir = args.b

# Detectors
detectors = ['E1']#, 'E2', 'E3'] # E1  E2 E3

# Masses for the template
m = 36  # Solar masses


filter = {}

# Load filter configuration from JSON
with open("confs/"+args.j, "r") as f:
    filter = json.load(f)

# Now you can access it like a regular dict
print("filter : ", filter)

# Dictionary to store results
results = {}

# Initialize the chunks dictionary
chunks = {}
"""
# templates
banks = [{'name': 'template1',"m1":1.4, "m2":1.4, "f_lower":20.0, "distance": 8295, "coa_phase":6.18, "inclination" : 0.44, "spin1z":0.05, "spin2z":0.05},
         #{'name': 'template2', "m1":50, "m2":50, "f_lower":20.0},
         #{'name': 'template3', "m1":75, "m2":75, "f_lower":20.0},
         #{'name': 'template4', "m1":100, "m2":100, "f_lower":20.0},
         #{'name': 'template5', "m1":120, "m2":120, "f_lower":20.0},
         #{'name': 'template6', "m1":140, "m2":140, "f_lower":20.0},
         #{'name': 'template7', "m1":160, "m2":160, "f_lower":20.0},
         #{'name': 'template8', "m1":180, "m2":180, "f_lower":20.0},
         #{'name': 'template9', "m1":200, "m2":200, "f_lower":20.0},
         #{'name': 'template10', "m1":180, "m2": 110, "f_lower":20.0},

         #{'name': 'template11', "m1": 36,  "m2": 50, "f_lower":20.0},
         #{'name': 'template12', "m1": 50,  "m2": 75, "f_lower":20.0},
         #{'name': 'template13', "m1": 75,  "m2": 100, "f_lower":20.0},
         #{'name': 'template14', "m1": 100, "m2": 150, "f_lower":20.0},
         #{'name': 'template15', "m1": 120, "m2": 60, "f_lower":20.0},
         #{'name': 'template16', "m1": 90,  "m2": 30, "f_lower":20.0},
         #{'name': 'template17', "m1": 140, "m2": 100, "f_lower":20.0},
         #{'name': 'template18', "m1": 160, "m2": 120, "f_lower":20.0},
         #{'name': 'template19', "m1": 200, "m2": 150, "f_lower":20.0},
         #{'name': 'template20', "m1": 250, "m2": 250, "f_lower":20.0},

         #{'name': 'template21', "m1": 50,  "m2": 36, "f_lower":20.0},
         #{'name': 'template22', "m1": 100,  "m2": 75, "f_lower":20.0},
        ]
class RandomSignalDataset(Dataset):
    def __init__(self, num_samples=10000, signal_length0=2048, n_channels=1, noise=0.0):
        super().__init__()
        # Pre-generate all templates with cyclic shift and resampling
        template_waveforms = []
        self.signal_length=0
        for template_ in banks:
            hp, hc = get_td_waveform(approximant='IMRPhenomNSBH',#IMRPhenomD_NRTidal # IMRPhenomNSBH
                                    mass1=template_['m1'], mass2=template_['m2'],
                                    delta_t=1/2048,
                                    f_lower=template_['f_lower'])
            # Resize/crop to signal_length

            target_fs = 256  # Hz
            target_dt = 1.0 / target_fs

            # Resample
            hp_resampled = resample_to_delta_t(hp, target_dt)

            print("Original length:", len(hp))
            print("Resampled length:", len(hp_resampled))

            if len(hp_resampled) < signal_length0:
              pad_total = signal_length0 - len(hp_resampled)
              pad_left = pad_total // 2
              pad_right = pad_total - pad_left
              hp_resampled = torch.cat([
                  torch.zeros(pad_left),
                  torch.tensor(hp_resampled.numpy(), dtype=torch.float32),
                  torch.zeros(pad_right)
              ])

            #if len(hp_resampled) < signal_length0:
                # pad with zeros
            #    hp_resampled = torch.cat([torch.zeros(signal_length0 - len(hp_resampled)), torch.tensor(hp_resampled.numpy(), dtype=torch.float32)])
            elif len(hp_resampled) > signal_length0:
                # truncate
                hp_resampled = torch.tensor(hp_resampled.numpy(), dtype=torch.float32)[-signal_length0:]
            else:
                hp_resampled = torch.tensor(hp_resampled.numpy(), dtype=torch.float32)

            hp=hp_resampled
            self.signal_length=len(hp_resampled)

            #if len(hp) < self.signal_length:
            #    hp.resize(self.signal_length)
            #else:
            #    hp = hp[:self.signal_length]

            # Convert to tensor
            template_tensor = torch.tensor(hp.numpy(), dtype=torch.float32)
            #template_tensor = template_tensor / template_tensor.abs().max()
            template_tensor = (template_tensor - template_tensor.mean()) / hp_resampled.std()

            # Optional cyclic shift
            #shift = torch.randint(0, self.signal_length, (1,)).item()
            #template_tensor = torch.cat([template_tensor[-shift:], template_tensor[:-shift]])

            template_waveforms.append(template_tensor)


        self.signals = torch.zeros(num_samples, n_channels, self.signal_length)
        self.templates = torch.zeros(num_samples, n_channels, self.signal_length, len(banks))
        self.labels = torch.zeros(num_samples, dtype=torch.long)  # NEW: store template index

        n_templates = len(banks)

        # Generate noisy signals
        for i in range(num_samples):
            for j in range(n_templates):
                L_old = template_waveforms[j].shape[0]

                if L_old >= self.signal_length:
                    self.templates[i, 0, :, j] = template_waveforms[j][:self.signal_length]  # truncate
                else:
                    # pad with zeros at start
                    padding = self.signal_length - L_old
                    self.templates[i, 0, :, j] = torch.cat([torch.zeros(padding), template_waveforms[j]])

                #self.templates[i, 0, :, j] = template_waveforms[j]

            # Pick a random template as the "source" of this noisy signal
            chosen_template = torch.randint(0, n_templates, (1,)).item()
            self.labels[i] = chosen_template

            template = self.templates[i, 0, :, chosen_template]  # [L] tensor

            # Uniform noise in [-1, 1] matching template length
            noise_tensor = noise * (2 * torch.rand_like(template) - 1)
            # Add noise
            noise_level = 0.05 * template.abs().max()  # 5% of peak amplitude
            self.signals[i, 0] = template + noise_level * torch.randn_like(template)



    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.templates[idx], self.labels[idx], self.signal_length

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, signal_length, patch_size, n_channels):
        super().__init__()
        self.linear_project = nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_project(x)  # (B, d_model, n_patches)
        return x.transpose(1, 2)    # (B, n_patches, d_model)



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, spatial_shape, r_mlp=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, spatial_shape)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SignalTransformer(nn.Module):
    def __init__(self, d_model, signal_length, patch_size, n_channels, n_heads, n_layers, n_templates):
        super().__init__()
        assert signal_length % patch_size == 0
        self.n_templates = n_templates
        self.spatial_shape = (signal_length // patch_size,)
        self.n_patches = self.spatial_shape[0]
        self.patch_size = patch_size
        self.n_channels = n_channels

        # Embedding layers
        self.patch_embed = PatchEmbedding(d_model, signal_length, patch_size, n_channels)
        self.patch_embed_templates = PatchEmbedding(d_model, signal_length, patch_size, n_channels)

        # Transformer encoder & cross-attention
        self.encoder = nn.Sequential(*[
            TransformerEncoder(d_model, n_heads, self.spatial_shape)
            for _ in range(n_layers)
        ])
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            spatial_shape_q=(self.n_patches,),
            spatial_shape_kv=(self.n_patches,)
        )

        # Reconstruction head
        self.reconstruct = nn.Linear(d_model, self.patch_size * self.n_channels)

        # **NEW: Classification head**
        self.classifier = nn.Sequential(
            nn.Linear(self.n_templates * n_heads, 128),
            nn.GELU(),
            nn.Linear(128, n_templates)
        )

        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x, templates):
        """
        Returns:
            x_out         : (B, n_patches, d_model)
            match_score   : (B, n_heads, n_patches, n_patches*T)
            recon_signal  : (B, C, L)
            class_logits  : (B, n_templates) — predicted template class
        """
        B, C, L = x.shape
        assert L == self.n_patches * self.patch_size
        _, _, _, T = templates.shape

        # Signal path
        x_tokens = self.patch_embed(x)
        x_tokens = self.encoder(x_tokens)
        x_norm = self.ln_cross(x_tokens)

        # Templates → tokens
        temps_btcl = templates.permute(0, 3, 1, 2).reshape(B*T, C, L)
        temp_tokens = self.patch_embed_templates(temps_btcl)
        temp_tokens = temp_tokens.reshape(B, T*self.n_patches, -1)

        # Cross-attention
        x_cross, match_score = self.cross_attn(x_norm, temp_tokens)
        x_out = self.ln_out(x_cross)

        # Reconstruction
        blocks = self.reconstruct(x_out)
        blocks = blocks.view(B, self.n_patches, self.n_channels, self.patch_size)
        recon_signal = blocks.permute(0, 2, 1, 3).reshape(B, self.n_channels, self.n_patches*self.patch_size)

        # **Template classification**
        # match_score: (B, n_heads, N_q, N_k = n_patches*T)
        # Step 1: average over query patches → (B, n_heads, N_k)
        attn_per_k = match_score.mean(dim=2)
        # Step 2: reshape into (B, n_heads, T, n_patches)
        attn_templates = attn_per_k.view(B, match_score.size(1), T, self.n_patches)
        # Step 3: sum per template → (B, n_heads, T)
        template_scores = attn_templates.sum(dim=-1)
        # Step 4: flatten heads → (B, n_heads*T)
        class_features = template_scores.flatten(start_dim=1)
        # Step 5: classifier → logits
        class_logits = self.classifier(class_features)

        return x_out, match_score, recon_signal, class_logits



# Training setup
d_model = 64          # Increased model size for better learning
signal_length0 = 2048
patch_size = 32
n_channels = 1
n_heads = 8
n_layers = 3
batch_size = 128
epochs = 40
lr = 0.005



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")




# data
num_train = 10000
num_test  = 2000

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

# Dataset
train_ds = RandomSignalDataset(num_samples=10000, signal_length0=signal_length0)
test_ds  = RandomSignalDataset(num_samples=2000,  signal_length0=signal_length0)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)



# Pick a sample index
idx = 40
noisy_signal, templates, label, signal_length = train_ds[idx]  # shapes: [1,512], [1,512,n_templates], label scalar

transform = T.Compose([
    T.Resize(signal_length),
    T.ToTensor(),
])
# Remove channel dimension
noisy_signal = noisy_signal[0]        # [L]
templates = templates[0]              # [L, n_templates]

# Create time axis (relative time in seconds)
# Assuming delta_t = 1/2048 as in waveform generation
delta_t = 1/2048

# Plot noisy signal
plt.figure(figsize=(12,4))
plt.plot(noisy_signal.numpy(), label='Noisy Signal', linewidth=2)

# Plot all templates
n_templates = templates.shape[1]
for j in range(n_templates):
    plt.plot(templates[:, j].numpy(), linestyle='--', label=f'Template {j}')

# Highlight the chosen template
#plt.plot(templates[:, label.item()].numpy(), linestyle='-', linewidth=2, color='red', label='Chosen Template')

plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title(f'Sample {idx} - Chosen Template {label.item()}')
plt.grid(True)
plt.legend()
plt.show()




# Model
model = SignalTransformer(d_model, signal_length, patch_size, n_channels, n_heads, n_layers, len(banks)).to(device)
optimizer = Adam(model.parameters(), lr=3e-4)
recon_loss_fn = torch.nn.MSELoss()
cls_loss_fn = torch.nn.CrossEntropyLoss()


def plot_signal_with_template_attention(signals, recon_signal, templates, match_score, sample_idx=0, head=0, patch_size=32, n_templates=2):
    """
    Plots waveform + per-template attention heatmaps for a single sample
    Also computes and displays the predicted template index.
    """
    signal_noisy = signals[sample_idx, 0].detach().cpu().numpy()
    signal_recon = recon_signal[sample_idx, 0].detach().cpu().numpy()
    signal_target = templates[sample_idx, 0, :, 0].detach().cpu().numpy()  # true template
    attn = match_score[sample_idx, head].detach().cpu().numpy()             # (n_patches, n_patches*T)

    # -----------------------
    # Compute predicted template
    # -----------------------
    n_patches_q, n_patches_k = attn.shape
    n_patches_per_template = n_patches_k // n_templates
    # reshape to (signal_patches, templates, template_patches) and sum over template patches
    attn_sum_per_template = attn.reshape(n_patches_q, n_templates, n_patches_per_template).sum(axis=2)
    # sum over signal patches to get a total attention per template
    template_votes = attn_sum_per_template.sum(axis=0)  # shape: (n_templates,)
    pred_template_idx = np.argmax(template_votes)       # predicted template

    # -----------------------
    # Plotting
    # -----------------------
    plt.figure(figsize=(14, 4 + 2*n_templates))

    # --- Top: waveform ---
    plt.subplot(n_templates+1,1,1)
    plt.plot(signal_noisy, label="Noisy input", alpha=0.6)
    plt.plot(signal_recon, label="Reconstructed", alpha=0.8)
    #plt.plot(signal_target, label="True template", linestyle='--')
    plt.title(f"Waveforms (Sample {sample_idx}) | Predicted template index: {pred_template_idx}")
    plt.ylabel("Amplitude")
    plt.legend()

    # --- Bottom: per-template attention heatmaps ---
    for t in range(n_templates):
        plt.subplot(n_templates+1,1,t+2)
        block = attn[:, t*n_patches_per_template:(t+1)*n_patches_per_template]
        im = plt.imshow(block, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Cross-attention to Template {t} (Head {head})")
        plt.xlabel("Template patches")
        plt.ylabel("Signal patches")

    plt.tight_layout()
    plt.show()


    N_zoom = 256
    coalescence_idx = 800  # center of your zoom region
    half_window = N_zoom // 2

    start_idx = max(coalescence_idx - half_window, 0)
    end_idx = min(coalescence_idx + half_window, len(signal_noisy))

    signal_noisy_zoom = signal_noisy[start_idx:end_idx]
    signal_recon_zoom = signal_recon[start_idx:end_idx]
    signal_target_zoom = signal_target[start_idx:end_idx]


    delta_t = 1/2048  # or your signal sampling
    t_zoom = np.arange(-N_zoom, 0) * delta_t

    n_patches_q, n_patches_k = attn.shape
    patches_per_point = n_patches_q / len(signal_noisy)  # approximate
    start_patch = int(n_patches_q - N_zoom * patches_per_point)
    attn_zoom = attn[start_patch:, :]

    plt.figure(figsize=(12, 3 + 2*n_templates))

    # Waveform zoom
    plt.subplot(n_templates+1, 1, 1)
    plt.plot(signal_noisy_zoom, label="Noisy input", alpha=0.6)
    plt.plot(signal_recon_zoom, label="Reconstructed", alpha=0.8)
    plt.title(f"Waveforms (Sample {sample_idx}) - Coalescence region zoom")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()







# -----------------------------
# Training Loop
# -----------------------------
train_recon_losses = []
train_cls_losses = []
test_recon_losses = []
test_cls_losses = []
test_accuracies = []


for epoch in range(epochs):
    model.train()
    total_recon_loss = 0.0
    total_cls_loss = 0.0

    for noisy_signal, temps, labels, signal_length in train_loader:
        B, C, L, T = temps.shape
        noisy_signal, temps, labels = noisy_signal.to(device), temps.to(device), labels.to(device)
        optimizer.zero_grad()

        _, match_score, recon, logits = model(noisy_signal, temps)

        # Correct template per sample
        temps_reshape = temps.view(B, C*L, T)
        labels_expand = labels.view(B, 1, 1).expand(B, C*L, 1)
        target = torch.gather(temps_reshape, dim=2, index=labels_expand).view(B, C, L)

        recon_loss = recon_loss_fn(recon, target)
        cls_loss = cls_loss_fn(logits, labels)
        loss = recon_loss + 0.5 * cls_loss  # balance weights

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_cls_loss += cls_loss.item()

    train_recon_losses.append(total_recon_loss / len(train_loader))
    train_cls_losses.append(total_cls_loss / len(train_loader))

    # -----------------
    # Evaluation (inline)
    # -----------------
    model.eval()
    total_recon_loss = 0.0
    total_cls_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for noisy_signal, temps, labels, signal_length in test_loader:
            B, C, L, T = temps.shape
            noisy_signal, temps, labels = noisy_signal.to(device), temps.to(device), labels.to(device)

            _, match_score, recon, logits = model(noisy_signal, temps)

            if(epoch==9):
                # Plot first 2 samples from this batch
                for sample_idx in range(min(2, noisy_signal.size(0))):
                    plot_signal_with_template_attention(
                        noisy_signal, recon, temps, match_score,
                        sample_idx=sample_idx, head=0,
                        patch_size=patch_size, n_templates=len(banks)
                    )
                #break  # only plot first batch

            # Correct template per sample
            temps_reshape = temps.view(B, C*L, T)
            labels_expand = labels.view(B, 1, 1).expand(B, C*L, 1)
            target = torch.gather(temps_reshape, dim=2, index=labels_expand).view(B, C, L)

            recon_loss = recon_loss_fn(recon, target)
            cls_loss = cls_loss_fn(logits, labels)

            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_recon_losses.append(total_recon_loss / len(test_loader))
    test_cls_losses.append(total_cls_loss / len(test_loader))
    test_accuracies.append(100.0 * correct / total)

    print(f"Epoch {epoch+1:02d} | "
          f"Train MSE={train_recon_losses[-1]:.4f}, TrainClsLoss={train_cls_losses[-1]:.4f} | "
          f"Test MSE={test_recon_losses[-1]:.4f}, TestClsLoss={test_cls_losses[-1]:.4f}, "
          f"Accuracy={test_accuracies[-1]:.2f}%")


epochs_range = range(1, epochs+1)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, train_recon_losses, label="Train MSE")
plt.plot(epochs_range, test_recon_losses, label="Test MSE")
plt.xlabel("Epochs"); plt.ylabel("Reconstruction Loss (MSE)")
plt.title("Signal Reconstruction Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, test_accuracies, label="Classification Accuracy (%)", color="tab:orange")
plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
plt.title("Template Classification Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
