from collections import OrderedDict

import torch
from torch import nn

from model.MotionAGFormer import AGFormerBlock, MotionAGFormerBlock
from model.modules.graph import GCN
from model.modules.temporal_moe import TemporalMoEBlock


class MotionAGFormerMoEBlock(nn.Module):
    """Spatial Mamba followed by lightweight Mamba + RWKV-6 temporal MoE."""

    is_temporal_moe = True

    def __init__(self, dim, layer_id, num_layers,
                 mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0.,
                 drop_path=0., num_heads=8, use_layer_scale=True,
                 qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_temporal_similarity=True, neighbour_num=4, n_frames=243,
                 router_hidden_ratio=0.25, rwkv_dim=64,
                 rwkv_head_size=32, rwkv_mix_rank=16,
                 rwkv_decay_rank=32):
        super().__init__()

        self.spatial = AGFormerBlock(
            dim=dim,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            mode='spatial',
            mixer_type='mamba',
            use_temporal_similarity=use_temporal_similarity,
            neighbour_num=neighbour_num,
            n_frames=n_frames,
        )

        self.temporal = TemporalMoEBlock(
            dim=dim,
            layer_id=layer_id,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            drop=drop,
            drop_path=drop_path,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            router_hidden_ratio=router_hidden_ratio,
            rwkv_dim=rwkv_dim,
            rwkv_head_size=rwkv_head_size,
            rwkv_mix_rank=rwkv_mix_rank,
            rwkv_decay_rank=rwkv_decay_rank,
        )

    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        return x


def _block_type_for_layer(i, n_layers):
    """Mirror the original sx-zxf main-branch Mamba/Attention schedule."""
    if i < (n_layers * 1 / 3) - 1:
        return "mamba"
    if (n_layers * 1 / 3) - 1 < i < (n_layers * 2 / 3) - 1:
        return "attention"
    if (n_layers * 2 / 3) - 1 < i < (n_layers * 5 / 6) - 1:
        return "mamba"
    return "attention"


def create_moe_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU,
                      attn_drop=0., drop_rate=0., drop_path_rate=0.,
                      num_heads=8, use_layer_scale=True, qkv_bias=False,
                      qkv_scale=None, layer_scale_init_value=1e-5,
                      use_temporal_similarity=True, temporal_connection_len=1,
                      neighbour_num=4, n_frames=243,
                      router_hidden_ratio=0.25, rwkv_dim=64,
                      rwkv_head_size=32, rwkv_mix_rank=16,
                      rwkv_decay_rank=32):
    """Create baseline-compatible layers.

    Original Mamba blocks become lightweight temporal MoE blocks.
    Original Attention blocks are kept unchanged.
    """
    layers = []
    for i in range(n_layers):
        block_type = _block_type_for_layer(i, n_layers)
        if block_type == "mamba":
            layers.append(
                MotionAGFormerMoEBlock(
                    dim=dim,
                    layer_id=i,
                    num_layers=n_layers,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    attn_drop=attn_drop,
                    drop=drop_rate,
                    drop_path=drop_path_rate,
                    num_heads=num_heads,
                    use_layer_scale=use_layer_scale,
                    qkv_bias=qkv_bias,
                    qk_scale=qkv_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    use_temporal_similarity=use_temporal_similarity,
                    neighbour_num=neighbour_num,
                    n_frames=n_frames,
                    router_hidden_ratio=router_hidden_ratio,
                    rwkv_dim=rwkv_dim,
                    rwkv_head_size=rwkv_head_size,
                    rwkv_mix_rank=rwkv_mix_rank,
                    rwkv_decay_rank=rwkv_decay_rank,
                )
            )
        else:
            # Keep the original Spatial Attention -> Temporal Attention block.
            layers.append(
                MotionAGFormerBlock(
                    dim=dim,
                    mod="attention",
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    attn_drop=attn_drop,
                    drop=drop_rate,
                    drop_path=drop_path_rate,
                    num_heads=num_heads,
                    use_layer_scale=use_layer_scale,
                    qkv_bias=qkv_bias,
                    qk_scale=qkv_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    use_adaptive_fusion=False,
                    hierarchical=False,
                    use_temporal_similarity=use_temporal_similarity,
                    temporal_connection_len=temporal_connection_len,
                    use_tcn=False,
                    graph_only=False,
                    neighbour_num=neighbour_num,
                    n_frames=n_frames,
                )
            )
    return nn.ModuleList(layers)


class MotionAGFormerMoE(nn.Module):
    """Parameter-efficient MotionAGFormer temporal MoE variant.

    The baseline architecture is retained. Only blocks that were Mamba blocks in
    the original schedule receive a second lightweight RWKV-6 TimeMix expert.
    Attention blocks remain exactly Spatial Attention -> Temporal Attention.
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3,
                 mlp_ratio=4, act_layer=nn.GELU, attn_drop=0., drop=0.,
                 drop_path=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5, num_heads=4, qkv_bias=False,
                 qkv_scale=None, num_joints=17, use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243,
                 router_hidden_ratio=0.25, rwkv_dim=64,
                 rwkv_head_size=32, rwkv_mix_rank=16,
                 rwkv_decay_rank=32, **kwargs):
        super().__init__()

        self.n_frames = n_frames
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))

        self.gcn_s = GCN(
            dim=dim_feat,
            num_nodes=17,
            neighbour_num=neighbour_num,
            mode='spatial',
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
        )
        self.gcn_t = GCN(
            dim=dim_feat,
            num_nodes=n_frames,
            neighbour_num=neighbour_num,
            mode='temporal',
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
        )

        self.layers = create_moe_layers(
            dim=dim_feat,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            attn_drop=attn_drop,
            drop_rate=drop,
            drop_path_rate=drop_path,
            num_heads=num_heads,
            use_layer_scale=use_layer_scale,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            layer_scale_init_value=layer_scale_init_value,
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
            neighbour_num=neighbour_num,
            n_frames=n_frames,
            router_hidden_ratio=router_hidden_ratio,
            rwkv_dim=rwkv_dim,
            rwkv_head_size=rwkv_head_size,
            rwkv_mix_rank=rwkv_mix_rank,
            rwkv_decay_rank=rwkv_decay_rank,
        )

        self.moe_layer_indices = [
            i for i, layer in enumerate(self.layers)
            if getattr(layer, 'is_temporal_moe', False)
        ]

        self.norm = nn.LayerNorm(dim_feat)
        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh()),
        ]))
        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False, return_router=False):
        x = self.joints_embed(x)
        x = x + self.gcn_s(x)
        x = x + self.gcn_t(x)
        x = x + self.pos_embed

        router_weights = []
        for layer in self.layers:
            x = layer(x)
            if return_router and getattr(layer, 'is_temporal_moe', False):
                router_weights.append(layer.temporal.mixer.last_router_weights)

        x = self.norm(x)
        x = self.rep_logit(x)

        if return_rep:
            if return_router:
                return x, router_weights
            return x

        x = self.head(x)
        if return_router:
            return x, router_weights
        return x


def _test():
    b, t, j, c = 1, 27, 17, 3
    x = torch.randn(b, t, j, c)
    model = MotionAGFormerMoE(
        n_layers=6,
        dim_in=3,
        dim_feat=128,
        dim_rep=512,
        n_frames=t,
        num_joints=j,
        rwkv_dim=64,
        rwkv_head_size=32,
        rwkv_mix_rank=16,
        rwkv_decay_rank=32,
    )
    y, routing = model(x, return_router=True)
    assert y.shape == (b, t, j, 3)
    assert all(w.shape == (b, t, j, 2) for w in routing)
    print('output:', y.shape)
    print('moe layers:', model.moe_layer_indices)
    print('router tensors:', len(routing))
    print('params:', sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    _test()
