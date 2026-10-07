"""One per-frame spatial read from XY features into the depth stream."""
import torch
from torch import nn


class XYDepthInteraction(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads:
            raise ValueError('Feature dimension must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.z_norm = nn.LayerNorm(dim)
        self.xy_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        # Initially exactly the E5 residual path. Q/K/V receive gradients after
        # the output projection begins learning; do not add a second zero gate.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z, xy):
        b, t, j, c = z.shape
        q = self.q(self.z_norm(z)).reshape(b * t, j, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.xy_norm(xy)).reshape(b * t, j, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((q * self.head_dim ** -0.5) @ k.transpose(-2, -1)).softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(b, t, j, c)
        return z + self.proj(context)
