import torch
import torch.nn as nn
from typing import Tuple

from MyRotaryEmbeddingND import RotaryEmbeddingND

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, spatial_shape_q, spatial_shape_kv):
        """
        d_model      : embedding dimension
        n_heads      : number of attention heads
        spatial_shape_q : spatial shape (or length) of the query sequence
        spatial_shape_kv: spatial shape (or length) of the key/value sequence
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # rotary embeddings for Q and K
        self.rope_q = RotaryEmbeddingND(
            spatial_shape_q,
            self.head_dim,
            learned_freq=True,
            use_xpos=True,
            xpos_scale_base=spatial_shape_q[0] * 2,
            theta_rescale_factor=1.0
        )

        self.rope_kv = RotaryEmbeddingND(
            spatial_shape_kv,
            self.head_dim,
            learned_freq=True,
            use_xpos=True,
            xpos_scale_base=spatial_shape_kv[0] * 2,
            theta_rescale_factor=1.0
        )

    def forward(self, q_seq, kv_seq):
        """
        q_seq : (B, N_q, d_model) query sequence (e.g., noisy signal)
        kv_seq: (B, N_k, d_model) key/value sequence (e.g., templates)
        """
        B, N_q, C = q_seq.shape
        N_k = kv_seq.shape[1]

        # Linear projections
        q = self.q_proj(q_seq).reshape(B, N_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_seq).reshape(B, N_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_seq).reshape(B, N_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply rotary embeddings
        #q, k = self.rope_q(q, q), self.rope_kv(k, k)  # q and k each get their RoPE

        q = self.rope_q.apply_to(q)
        k = self.rope_kv.apply_to(k)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        return self.out_proj(out), attn  # return output + attention weights
