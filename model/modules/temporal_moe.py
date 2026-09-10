import torch
import torch.nn as nn

from model.modules.mamba import MAM
from model.modules.rwkv6 import RWKV6TemporalExpert
from model.modules.mlp import MLP
from timm.models.layers import DropPath


class TemporalMoE(nn.Module):
    """Dense soft temporal MoE with Mamba and RWKV-6 experts.

    Input / output: [B, T, J, C]
    Router output: [B, T, J, 2]

    Both experts always process the complete temporal sequence. The router only
    controls the fusion weights, so frame-wise weights do not break temporal
    continuity inside either expert.
    """

    def __init__(self, dim, layer_id, num_layers, router_hidden_ratio=0.5,
                 rwkv_head_size=32, rwkv_ffn_mult=3.5, dropout=0.0):
        super().__init__()

        router_hidden = max(16, int(dim * router_hidden_ratio))

        self.mamba_expert = MAM(
            d_model=dim,
            d_state=8,
            d_conv=3,
            expand=1,
            mode='temporal',
        )
        self.rwkv_expert = RWKV6TemporalExpert(
            dim=dim,
            layer_id=layer_id,
            num_layers=num_layers,
            head_size=rwkv_head_size,
            ffn_mult=rwkv_ffn_mult,
            dropout=dropout,
        )
        self.router = nn.Sequential(
            nn.Linear(dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 2),
        )

        # Start from an unbiased 0.5 / 0.5 mixture.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.last_router_weights = None

    def forward(self, x):
        y_mamba = self.mamba_expert(x)
        y_rwkv = self.rwkv_expert(x)

        logits = self.router(x)
        weights = torch.softmax(logits, dim=-1)
        self.last_router_weights = weights.detach()

        w_mamba = weights[..., 0:1]
        w_rwkv = weights[..., 1:2]
        return w_mamba * y_mamba + w_rwkv * y_rwkv


class TemporalMoEBlock(nn.Module):
    """AGFormer-style residual block whose mixer is TemporalMoE."""

    def __init__(self, dim, layer_id, num_layers, mlp_ratio=4., act_layer=nn.GELU,
                 drop=0., drop_path=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5, router_hidden_ratio=0.5,
                 rwkv_head_size=32, rwkv_ffn_mult=3.5):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.mixer = TemporalMoE(
            dim=dim,
            layer_id=layer_id,
            num_layers=num_layers,
            router_hidden_ratio=router_hidden_ratio,
            rwkv_head_size=rwkv_head_size,
            rwkv_ffn_mult=rwkv_ffn_mult,
            dropout=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
