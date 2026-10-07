"""E6 synthetic correctness checks; CPU uses explicit selective-scan reference."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--steps', type=int, default=3)
    opts = parser.parse_args()
    if min(opts.batch_size, opts.steps) < 1 or opts.frames < 3:
        parser.error('Require batch-size/steps >= 1 and frames >= 3')
    if opts.device == 'cpu':
        from tools.cpu_scan_reference import enable_cpu_reference
        enable_cpu_reference()
        torch.set_num_threads(2)
    elif not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    from utils.learning import load_model
    from utils.tools import get_config
    from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity
    cfg = get_config('configs/h36m/MotionAGFormer-xy-guided-depth.yaml')
    cfg.n_frames = opts.frames
    e5_cfg = deepcopy(cfg)
    e5_cfg.use_xy_depth_interaction = False
    torch.manual_seed(0)
    e5 = load_model(e5_cfg).eval()
    torch.manual_seed(0)
    e6 = load_model(cfg).eval()
    for key, value in e5.state_dict().items():
        torch.testing.assert_close(e6.state_dict()[key], value, rtol=0, atol=0)
    restored = load_model(cfg)
    restored.load_state_dict(e6.state_dict(), strict=True)
    del restored
    x = torch.randn(opts.batch_size, opts.frames, 17, 3, device=opts.device)
    x[..., 2] = torch.rand_like(x[..., 2])
    e5.to(opts.device)
    with torch.no_grad():
        expected = e5(x)
    e5.cpu()
    e6.to(opts.device)
    with torch.no_grad():
        actual = e6(x)
    assert actual.shape == x.shape and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    counts = [sum(p.numel() for p in m.parameters()) for m in (e5, e6)]
    result = dict(device=opts.device, input_shape=list(x.shape), e5_params=counts[0],
                  e6_params=counts[1], added_params=counts[1]-counts[0],
                  init_max_abs_diff=(actual-expected).abs().max().item())
    del expected, actual
    e6.cpu()
    grad_cfg = deepcopy(cfg)
    grad_cfg.n_frames = min(opts.frames, 9) if opts.device == 'cpu' else opts.frames
    model = load_model(grad_cfg).to(opts.device).train()
    gx = x[:, :grad_cfg.n_frames]
    target = torch.randn_like(gx)
    target = target - target[..., :1, :]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        pred = model(gx)
        loss = loss_mpjpe(pred, target) + cfg.lambda_scale*n_mpjpe(pred, target) + cfg.lambda_3d_velocity*loss_velocity(pred, target)
        loss.backward()
        interaction = model.xy_depth_interaction
        assert interaction.proj.weight.grad.abs().sum() > 0
        if step == 1:
            for name, p in interaction.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
    # Test the learned (nonzero) interaction rather than only its zero-init state.
    model.zero_grad(set_to_none=True)
    model(gx)[..., 2:].square().mean().backward()
    assert model.layers[-2].spatial.norm1.weight.grad.abs().sum() > 0
    assert model.joints_embed.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model(gx)[..., :2].square().mean().backward()
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.z_layers.parameters())
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.xy_depth_interaction.parameters())
    # XY dependency and frame independence of the interaction itself.
    block = model.xy_depth_interaction.eval()
    z = torch.randn(1, 3, 17, cfg.dim_feat, device=opts.device)
    xy = torch.randn_like(z)
    changed = xy.clone()
    changed[:, 0] += torch.randn_like(changed[:, 0])
    with torch.no_grad():
        a, b = block(z, xy), block(z, changed)
    assert (a[:, 0]-b[:, 0]).abs().max() > 0
    torch.testing.assert_close(a[:, 1:], b[:, 1:], rtol=0, atol=0)
    result.update(strict_roundtrip=True, two_optimizer_steps=True,
                  learned_xy_to_z_gradient=True, no_z_to_xy_forward_dependency=True,
                  per_frame_interaction=True, gradient_frames=grad_cfg.n_frames)
    del model, optimizer, pred, loss, block, interaction, a, b, z, xy, changed, target, gx
    if opts.device == 'cuda':
        def benchmark(m):
            m.to('cuda').train()
            opt = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            target = torch.randn_like(x)
            target -= target[..., :1, :].clone()
            def step():
                opt.zero_grad(set_to_none=True)
                p = m(x)
                loss = loss_mpjpe(p,target)+cfg.lambda_scale*n_mpjpe(p,target)+cfg.lambda_3d_velocity*loss_velocity(p,target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip_norm,error_if_nonfinite=True)
                opt.step()
            step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(opts.steps):
                step()
            torch.cuda.synchronize()
            out = dict(seconds_per_step=(time.perf_counter()-start)/opts.steps,
                       peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
            opt.zero_grad(set_to_none=True)
            del opt
            m.cpu()
            torch.cuda.empty_cache()
            return out
        result['e5_short_run'] = benchmark(e5)
        result['e6_short_run'] = benchmark(e6)
    else:
        result['gpu_vram_speed'] = 'not measured'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
