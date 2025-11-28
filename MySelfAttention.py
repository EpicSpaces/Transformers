import torch
import torch.nn as nn
from typing import Tuple

from MyRotaryEmbeddingND import RotaryEmbeddingND

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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
