import torch
import torch.nn as nn
from typing import Tuple

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
