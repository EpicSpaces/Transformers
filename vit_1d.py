import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
from typing import Tuple
from torch.utils.data import Dataset

from MySelfAttention import SelfAttention

class RandomSignalDataset(Dataset):
    def __init__(self, num_samples=10000, signal_length=500, n_channels=1, n_classes=10):
        super().__init__()
        self.signals = torch.zeros(num_samples, n_channels, signal_length)
        self.labels = torch.zeros(num_samples, dtype=torch.long)

        t = torch.linspace(0, 1, signal_length)

        for i in range(num_samples):
            label = torch.randint(0, n_classes, (1,)).item()
            freq = 2 + label + 0.1 * torch.randn(1)  # class-dependent frequency + small random shift
            phase = 2 * torch.pi * torch.rand(1)      # random phase
            noise=0.1
            self.signals[i, 0] = torch.sin(2 * torch.pi * freq * t + phase) + noise * torch.randn(signal_length)
            self.labels[i] = label

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


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
        x = x + self.attn(self.ln1(x), True)
        x = x + self.mlp(self.ln2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, signal_length, patch_size, n_channels, n_heads, n_layers):
        super().__init__()
        assert signal_length % patch_size == 0
        self.spatial_shape = (signal_length // patch_size,)

        self.n_patches = self.spatial_shape[0]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.patch_embed = PatchEmbedding(d_model, signal_length, patch_size, n_channels)
        
        self.encoder = nn.Sequential(*[
            TransformerEncoder(d_model, n_heads, self.spatial_shape)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)  # (B, n_patches, d_model)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, d_model)
        x = self.encoder(x)
        return self.classifier(x[:, 0])  # CLS token output

# Training setup
d_model = 64          # Increased model size for better learning
n_classes = 10
signal_length = 512
patch_size = 32
n_channels = 1
n_heads = 8
n_layers = 3
batch_size = 128
epochs = 100
lr = 0.005


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

transform = T.Compose([
    T.Resize(signal_length),
    T.ToTensor(),
])


# Parameters
num_train = 10000
num_test = 2000

train_set = RandomSignalDataset(num_samples=num_train, signal_length=signal_length,
                                n_channels=n_channels, n_classes=n_classes)
test_set = RandomSignalDataset(num_samples=num_test, signal_length=signal_length,
                               n_channels=n_channels, n_classes=n_classes)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = VisionTransformer(d_model, n_classes, signal_length, patch_size, n_channels, n_heads, n_layers).to(device)
optimizer = Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_losses, test_accuracies = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_losses[-1]:.4f} | Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
