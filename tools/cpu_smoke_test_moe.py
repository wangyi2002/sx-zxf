import argparse

import torch

from model.MotionAGFormer import MotionAGFormer
from model.MotionAGFormer_MoE import MotionAGFormerMoE


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_moe(n_frames):
    return MotionAGFormerMoE(
        n_layers=30,
        dim_in=3,
        dim_feat=128,
        dim_rep=512,
        dim_out=3,
        mlp_ratio=4,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        num_heads=8,
        qkv_bias=False,
        qkv_scale=None,
        num_joints=17,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        neighbour_num=2,
        n_frames=n_frames,
        router_hidden_ratio=0.25,
        rwkv_dim=64,
        rwkv_head_size=32,
        rwkv_mix_rank=16,
        rwkv_decay_rank=32,
    )


def build_baseline(n_frames):
    return MotionAGFormer(
        n_layers=30,
        dim_in=3,
        dim_feat=128,
        dim_rep=512,
        dim_out=3,
        mlp_ratio=4,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_adaptive_fusion=False,
        num_heads=8,
        qkv_bias=False,
        qkv_scale=None,
        hierarchical=False,
        num_joints=17,
        use_temporal_similarity=True,
        temporal_connection_len=1,
        use_tcn=False,
        graph_only=False,
        neighbour_num=2,
        n_frames=n_frames,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-frames', type=int, default=9)
    args = parser.parse_args()

    # Exact parameter count for the production H36M configuration.
    prod_moe = build_moe(n_frames=243)
    prod_base = build_baseline(n_frames=243)
    moe_total, moe_trainable = count_params(prod_moe)
    base_total, base_trainable = count_params(prod_base)

    print(f'[PARAMS] MoE total/trainable: {moe_total:,} / {moe_trainable:,}')
    print(f'[PARAMS] Base total/trainable: {base_total:,} / {base_trainable:,}')
    print(f'[PARAMS] Increase: {moe_total - base_total:,} '
          f'({(moe_total / base_total - 1) * 100:.2f}%)')
    print(f'[MOE] layer indices: {prod_moe.moe_layer_indices}')

    # Full 30-layer CPU forward with a short sequence for a quick smoke test.
    smoke_frames = args.smoke_frames
    model = build_moe(n_frames=smoke_frames).cpu().eval()
    x = torch.randn(1, smoke_frames, 17, 3)

    with torch.no_grad():
        y, router_weights = model(x, return_router=True)

    expected_output = (1, smoke_frames, 17, 3)
    expected_router = (1, smoke_frames, 17, 2)
    assert tuple(y.shape) == expected_output, (y.shape, expected_output)
    assert len(router_weights) == len(model.moe_layer_indices)
    assert all(tuple(w.shape) == expected_router for w in router_weights)
    assert torch.isfinite(y).all()

    print(f'[CPU PASS] output shape: {tuple(y.shape)}')
    print(f'[CPU PASS] router tensors: {len(router_weights)}')
    print(f'[CPU PASS] router shape: {tuple(router_weights[0].shape)}')
    print('[CPU PASS] all outputs finite')


if __name__ == '__main__':
    main()
