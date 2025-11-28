import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
import math
from typing import Tuple
from MySelfAttention import SelfAttention

"""
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, n_channels, image_size):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.n_channels = n_channels

        Ph, Pw = patch_size
        H, W = image_size
        assert H % Ph == 0 and W % Pw == 0, "Image dimensions must be divisible by patch size"

        self.num_patches = (H // Ph) * (W // Pw)
        self.linear = nn.Linear(n_channels * Ph * Pw, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        Ph, Pw = self.patch_size

        x = x.unfold(2, Ph, Ph).unfold(3, Pw, Pw)  # (B, C, H//Ph, W//Pw, Ph, Pw)
        x = x.permute(0, 2, 3, 1, 4, 5)            # (B, H//Ph, W//Pw, C, Ph, Pw)
        x = x.flatten(1, 2)                        # (B, N_patches, C, Ph, Pw)
        x = x.flatten(2)                           # (B, N_patches, C*Ph*Pw)
        x = self.linear(x)                         # (B, N_patches, d_model)
        return x
"""

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.linear_project = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_project(x)  # (B, d_model, H_patch, W_patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, d_model)
        return x


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
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0

        self.spatial_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.n_patches = self.spatial_shape[0] * self.spatial_shape[1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.patch_embed = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        #self.patch_embed = PatchEmbedding(d_model, patch_size, n_channels, img_size)

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
img_size = (32, 32)
patch_size = (16, 16)
n_channels = 1
n_heads = 8
n_layers = 3
batch_size = 128
epochs = 10
lr = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

transform = T.Compose([
    T.Resize(img_size),
    T.ToTensor(),
])

train_set = MNIST(root="./../datasets", train=True, download=True, transform=transform)
test_set = MNIST(root="./../datasets", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
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
