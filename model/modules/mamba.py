import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_selective_scan_fn
except ImportError:
    _cuda_selective_scan_fn = None


def _selective_scan_cpu_reference(u, delta, A, B, C, D, z=None,
                                  delta_bias=None, delta_softplus=True):
    """Small differentiable selective-scan reference used for CPU smoke tests.

    Shapes follow mamba_ssm.selective_scan_fn for the variant used in this repo:
      u, delta: [B, D, L]
      A:        [D, N]
      B, C:     [B, N, L]
      D:        [D]

    GPU training still uses mamba_ssm's optimized selective scan when available.
    """
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1).to(delta.dtype)
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    d_state = A.shape[-1]
    state = torch.zeros(batch, dim, d_state, device=u.device, dtype=torch.float32)
    outputs = []

    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    D_f = D.float()

    for i in range(seqlen):
        dt = delta_f[:, :, i].unsqueeze(-1)
        delta_a = torch.exp(dt * A_f.unsqueeze(0))
        delta_b_u = dt * B_f[:, :, i].unsqueeze(1) * u_f[:, :, i].unsqueeze(-1)
        state = delta_a * state + delta_b_u
        y = (state * C_f[:, :, i].unsqueeze(1)).sum(dim=-1)
        y = y + D_f.unsqueeze(0) * u_f[:, :, i]
        outputs.append(y)

    out = torch.stack(outputs, dim=-1).to(u.dtype)
    if z is not None:
        out = out * F.silu(z)
    return out


def selective_scan_dispatch(u, delta, A, B, C, D, z=None,
                            delta_bias=None, delta_softplus=True):
    """Use CUDA selective scan for GPU training, CPU reference otherwise."""
    if u.device.type == 'cuda':
        if _cuda_selective_scan_fn is None:
            raise ImportError(
                "CUDA input requires mamba-ssm. Install a compatible mamba-ssm build."
            )
        return _cuda_selective_scan_fn(
            u, delta, A, B, C, D, z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
        )

    return _selective_scan_cpu_reference(
        u, delta, A, B, C, D, z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )


class MAM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            mode='spatial',
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mode = mode
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        batch, time, joints, channels = hidden_states.shape

        if self.mode == 'spatial':
            hidden_states = rearrange(hidden_states, 'b t j c -> (b t) j c')
            seqlen = joints
        elif self.mode == 'temporal':
            hidden_states = rearrange(hidden_states, 'b t j c -> (b j) t c')
            seqlen = time
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(
            x, self.conv1d_x.weight, self.conv1d_x.bias,
            padding='same', groups=self.d_inner // 2
        ))
        z = F.silu(F.conv1d(
            z, self.conv1d_z.weight, self.conv1d_z.bias,
            padding='same', groups=self.d_inner // 2
        ))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B_param = rearrange(
            B_param, "(b l) dstate -> b dstate l", l=seqlen
        ).contiguous()
        C_param = rearrange(
            C_param, "(b l) dstate -> b dstate l", l=seqlen
        ).contiguous()

        y = selective_scan_dispatch(
            x, dt, A, B_param, C_param, self.D.float(), z=None,
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
        )

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        if self.mode == 'spatial':
            out = rearrange(out, '(b t) j c -> b t j c', b=batch, t=time)
        else:
            out = rearrange(out, '(b j) t c -> b t j c', b=batch, j=joints)

        return out
