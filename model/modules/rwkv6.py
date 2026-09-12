import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKV6TimeMix(nn.Module):
    """Pure PyTorch RWKV-6/Finch-style temporal mixer.

    Input / output: [N, T, C]. This reference implementation keeps the
    data-dependent time mixing, dynamic decay and matrix-valued recurrent state.
    It intentionally avoids a custom CUDA WKV kernel for architecture validation.
    """

    def __init__(self, dim, layer_id, num_layers, head_size=32,
                 mix_rank=32, decay_rank=64):
        super().__init__()
        if dim % head_size != 0:
            raise ValueError(f"dim={dim} must be divisible by head_size={head_size}")

        self.dim = dim
        self.head_size = head_size
        self.num_heads = dim // head_size

        ratio_0_to_1 = layer_id / max(num_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_id / max(num_layers, 1)
        ddd = (torch.arange(dim, dtype=torch.float32) / dim).view(1, 1, dim)

        base = torch.pow(ddd, ratio_1_to_almost0)
        self.time_maa_x = nn.Parameter(1.0 - base)
        self.time_maa_w = nn.Parameter(1.0 - base.clone())
        self.time_maa_k = nn.Parameter(1.0 - base.clone())
        self.time_maa_v = nn.Parameter(1.0 - (base.clone() + 0.3 * ratio_0_to_1))
        self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
        self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        # Low-rank, data-dependent offsets for w, k, v, r, g.
        self.time_maa_w1 = nn.Parameter(torch.zeros(dim, mix_rank * 5))
        self.time_maa_w2 = nn.Parameter(torch.empty(5, mix_rank, dim))
        nn.init.uniform_(self.time_maa_w2, -0.01, 0.01)

        idx = torch.arange(dim, dtype=torch.float32)
        decay_speed = -6.0 + 5.0 * torch.pow(
            idx / max(dim - 1, 1), 0.7 + 1.3 * ratio_0_to_1
        )
        self.time_decay = nn.Parameter(decay_speed.view(1, 1, dim))
        self.time_decay_w1 = nn.Parameter(torch.zeros(dim, decay_rank))
        self.time_decay_w2 = nn.Parameter(torch.empty(decay_rank, dim))
        nn.init.uniform_(self.time_decay_w2, -0.01, 0.01)

        zigzag = (((idx + 1) % 3) - 1) * 0.1
        time_first = ratio_0_to_1 * (1.0 - idx / max(dim - 1, 1)) + zigzag
        self.time_first = nn.Parameter(time_first.view(self.num_heads, self.head_size))

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.group_norm = nn.GroupNorm(self.num_heads, dim, eps=1e-5)

    @staticmethod
    def _shift_diff(x):
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return prev - x

    def _wkv6_reference(self, r, k, v, w):
        """Reference recurrent WKV6 operator using a Python loop over time."""
        n, t, d = r.shape
        h, s = self.num_heads, self.head_size

        r = r.view(n, t, h, s).float()
        k = k.view(n, t, h, s).float()
        v = v.view(n, t, h, s).float()
        w = w.view(n, t, h, s).float()

        u = self.time_first.float().view(1, h, s, 1)
        state = torch.zeros(n, h, s, s, device=r.device, dtype=torch.float32)
        outputs = []

        for i in range(t):
            rt = r[:, i]
            kt = k[:, i]
            vt = v[:, i]
            wt = w[:, i]

            kv = kt.unsqueeze(-1) * vt.unsqueeze(-2)
            current = state + u * kv
            yt = torch.matmul(rt.unsqueeze(-2), current).squeeze(-2)
            outputs.append(yt)

            decay = torch.exp(-torch.exp(wt)).unsqueeze(-1)
            state = state * decay + kv

        return torch.stack(outputs, dim=1).reshape(n, t, d)

    def forward(self, x):
        n, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"expected feature dim {self.dim}, got {d}")

        xx = self._shift_diff(x)
        xxx = x + xx * self.time_maa_x

        dynamic = torch.tanh(xxx @ self.time_maa_w1).view(n, t, 5, -1)
        dynamic_out = torch.stack(
            [dynamic[:, :, i] @ self.time_maa_w2[i] for i in range(5)], dim=2
        )
        mw, mk, mv, mr, mg = dynamic_out.unbind(dim=2)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        dynamic_decay = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + dynamic_decay

        y = self._wkv6_reference(r, k, v, w).to(x.dtype)
        y = self.group_norm(y.reshape(n * t, d)).reshape(n, t, d)
        return self.output(y * g)


class RWKV6TimeMixTemporalExpert(nn.Module):
    """Lightweight pose temporal expert using RWKV-6 TimeMix only.

    The shared backbone feature is projected into a smaller expert dimension,
    modeled temporally, then projected back:

        [B,T,J,C] -> C->E -> RWKV6 TimeMix(E) -> E->C -> [B,T,J,C]

    There is intentionally no RWKV ChannelMix here because the outer
    TemporalMoEBlock already has a shared AGFormer MLP after expert fusion.
    """

    def __init__(self, dim, layer_id, num_layers, expert_dim=64,
                 head_size=32, mix_rank=16, decay_rank=32):
        super().__init__()
        if expert_dim % head_size != 0:
            raise ValueError(
                f"expert_dim={expert_dim} must be divisible by head_size={head_size}"
            )

        self.dim = dim
        self.expert_dim = expert_dim
        self.down_proj = nn.Linear(dim, expert_dim, bias=False)
        self.norm = nn.LayerNorm(expert_dim)
        self.time_mix = RWKV6TimeMix(
            dim=expert_dim,
            layer_id=layer_id,
            num_layers=num_layers,
            head_size=head_size,
            mix_rank=mix_rank,
            decay_rank=decay_rank,
        )
        self.up_proj = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x):
        b, t, j, c = x.shape
        seq = x.permute(0, 2, 1, 3).contiguous().view(b * j, t, c)
        seq = self.down_proj(seq)
        seq = self.time_mix(self.norm(seq))
        seq = self.up_proj(seq)
        return seq.view(b, j, t, c).permute(0, 2, 1, 3).contiguous()


# Full RWKV-6 block is retained for later ablations, but is not used by the
# lightweight MoE configuration below.
class RWKV6ChannelMix(nn.Module):
    def __init__(self, dim, layer_id, num_layers, ffn_mult=3.5):
        super().__init__()
        ratio = 1.0 - layer_id / max(num_layers, 1)
        ddd = (torch.arange(dim, dtype=torch.float32) / dim).view(1, 1, dim)
        mix = 1.0 - torch.pow(ddd, ratio ** 3)
        self.time_maa_k = nn.Parameter(mix.clone())
        self.time_maa_r = nn.Parameter(mix.clone())

        hidden_dim = max(32, int(dim * ffn_mult) // 32 * 32)
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        xx = prev - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        k = F.relu(self.key(xk)).square()
        return torch.sigmoid(self.receptance(xr)) * self.value(k)


class RWKV6Block(nn.Module):
    def __init__(self, dim, layer_id, num_layers, head_size=32,
                 ffn_mult=3.5, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.time_mix = RWKV6TimeMix(
            dim=dim,
            layer_id=layer_id,
            num_layers=num_layers,
            head_size=head_size,
        )
        self.channel_mix = RWKV6ChannelMix(
            dim=dim,
            layer_id=layer_id,
            num_layers=num_layers,
            ffn_mult=ffn_mult,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.time_mix(self.norm1(x)))
        x = x + self.dropout(self.channel_mix(self.norm2(x)))
        return x


class RWKV6TemporalExpert(nn.Module):
    """Full RWKV-6 pose wrapper retained for ablations."""

    def __init__(self, dim, layer_id, num_layers, head_size=32,
                 ffn_mult=3.5, dropout=0.0):
        super().__init__()
        self.block = RWKV6Block(
            dim=dim,
            layer_id=layer_id,
            num_layers=num_layers,
            head_size=head_size,
            ffn_mult=ffn_mult,
            dropout=dropout,
        )

    def forward(self, x):
        b, t, j, c = x.shape
        seq = x.permute(0, 2, 1, 3).contiguous().view(b * j, t, c)
        seq = self.block(seq)
        return seq.view(b, j, t, c).permute(0, 2, 1, 3).contiguous()
