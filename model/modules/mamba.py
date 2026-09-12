import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from model.modules.scan_backend import selective_scan_fn


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
            temporal_msm=False,
            dt_bias_mode="legacy_double",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mode = mode
        if dt_bias_mode not in ("legacy_double", "single"):
            raise ValueError("dt_bias_mode must be legacy_double or single")
        self.dt_bias_mode = dt_bias_mode
        self.temporal_msm = bool(temporal_msm) and mode == "temporal"
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Projection layers
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        # Initialize dt_proj weights
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt_proj bias motionagformer-b-h36m.pth.tr
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # State parameters
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # 1D Convolutions
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

        # SX-MSM-01: causal [previous, current] convolution on the MAM input.
        # Parameter rather than Conv1d avoids consuming baseline initialization RNG.
        if self.temporal_msm:
            self.msm_dt_weight = nn.Parameter(torch.zeros(
                self.d_inner // 2, self.d_model, 2, **factory_kwargs
            ))

    def motion_dt_logits(self, u):
        """[batch, time, C] -> [batch, SSM channels, time]; left zero padding."""
        return F.conv1d(F.pad(u.transpose(1, 2), (1, 0)), self.msm_dt_weight)

    def forward(self, hidden_states):
        B, T, J, C = hidden_states.shape

        # 模式分支处理
        if self.mode == 'spatial':
            # 空间模式: 处理每个时间步内的关节关系
            hidden_states = rearrange(hidden_states, 'b t j c -> (b t) j c')
            seqlen = J
        elif self.mode == 'temporal':
            # 时间模式: 处理每个关节的时间序列
            hidden_states = rearrange(hidden_states, 'b t j c -> (b j) t c')
            seqlen = T
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # 投影处理
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        # 卷积处理
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(x, self.conv1d_x.weight, self.conv1d_x.bias, padding='same', groups=self.d_inner // 2))
        z = F.silu(F.conv1d(z, self.conv1d_z.weight, self.conv1d_z.bias, padding='same', groups=self.d_inner // 2))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Legacy adds dt_proj.bias here AND inside selective_scan_fn.
        # Keep that behavior by default so MSM is the only experiment variable.
        projected_dt = (self.dt_proj(dt) if self.dt_bias_mode == "legacy_double"
                        else F.linear(dt, self.dt_proj.weight))
        dt = rearrange(projected_dt, "(b l) d -> b d l", l=seqlen)
        if self.temporal_msm:
            dt = dt + self.motion_dt_logits(hidden_states)
        B_param = rearrange(B_param, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C_param = rearrange(C_param, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # print(x.shape[-1])
        # 选择性扫描
        y = selective_scan_fn(x, dt, A, B_param, C_param, self.D.float(), z=None,
                              delta_bias=self.dt_proj.bias.float(), delta_softplus=True)

        # 输出处理
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        # 恢复原始维度
        if self.mode == 'spatial':
            out = rearrange(out, '(b t) j c -> b t j c', b=B, t=T)
        elif self.mode == 'temporal':
            out = rearrange(out, '(b j) t c -> b t j c', b=B, j=J)

        return out




    # def forward(self, hidden_states):
    #         # 输入形状: [B, T, J, C]
    #         B, T, J, C = hidden_states.shape
    #         x = hidden_states.reshape(B, T * J, C)  # 合并时空维度
    #
    #         xz = self.in_proj(x)
    #         xz = rearrange(xz, "b l d -> b d l")
    #         x, z = xz.chunk(2, dim=1)
    #
    #         A = -torch.exp(self.A_log.float())
    #         x = F.silu(F.conv1d(x, self.conv1d_x.weight, self.conv1d_x.bias, padding='same', groups=self.d_inner // 2))
    #         z = F.silu(F.conv1d(z, self.conv1d_z.weight, self.conv1d_z.bias, padding='same', groups=self.d_inner // 2))
    #
    #         x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
    #         dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #         dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=T * J)
    #         B_param = rearrange(B_param, "(b l) dstate -> b dstate l", l=T * J).contiguous()
    #         C_param = rearrange(C_param, "(b l) dstate -> b dstate l", l=T * J).contiguous()
    #
    #         y = selective_scan_fn(x, dt, A, B_param, C_param, self.D.float(), z=None,
    #                               delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
    #
    #         # 合并输出并恢复形状
    #         y = torch.cat([y, z], dim=1)
    #         y = rearrange(y, "b d l -> b l d")
    #         out = self.out_proj(y)
    #
    #         # 恢复4D输出
    #         return out.reshape(B, T, J, -1)  # [B, T, J, C_out]



    # def forward(self, hidden_states):
    #     _, seqlen, _ = hidden_states.shape
    #     xz = self.in_proj(hidden_states)
    #     xz = rearrange(xz, "b l d -> b d l")
    #     x, z = xz.chunk(2, dim=1)
    #
    #     # Selective scan operations
    #     A = -torch.exp(self.A_log.float())
    #     x = F.silu(F.conv1d(
    #         input=x,
    #         weight=self.conv1d_x.weight,
    #         bias=self.conv1d_x.bias,
    #         padding='same',
    #         groups=self.d_inner // 2
    #     ))
    #     z = F.silu(F.conv1d(
    #         input=z,
    #         weight=self.conv1d_z.weight,
    #         bias=self.conv1d_z.bias,
    #         padding='same',
    #         groups=self.d_inner // 2
    #     ))
    #
    #     # Project and split
    #     x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
    #     dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #     dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
    #     B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    #     C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    #
    #     # Selective scan
    #     y = selective_scan_fn(
    #         x, dt, A, B, C, self.D.float(),
    #         z=None,
    #         delta_bias=self.dt_proj.bias.float(),
    #         delta_softplus=True,
    #         return_last_state=None
    #     )

        # Merge and project
        # y = torch.cat([y, z], dim=1)
        # y = rearrange(y, "b d l -> b l d")
        # out = self.out_proj(y)
        # return out