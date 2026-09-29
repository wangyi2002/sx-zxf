"""Synthetic checks only. CPU uses the explicit scan reference, CUDA the real kernel.

Run before full training; never loads data or writes checkpoints.
"""
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
    if min(opts.batch_size, opts.frames, opts.steps) < 1 or opts.frames < 3:
        parser.error('Require batch-size/steps >= 1 and frames >= 3')
    if opts.device == 'cpu':
        from tools.cpu_scan_reference import enable_cpu_reference
        enable_cpu_reference()
        torch.set_num_threads(2)
    elif not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    from utils.learning import load_model
    from utils.tools import get_config
    from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity

    config = get_config('configs/h36m/MotionAGFormer-xy-z-split.yaml')
    config.n_frames = opts.frames
    base_config = deepcopy(config)
    base_config.use_xy_z_split = False
    torch.manual_seed(0)
    base = load_model(base_config).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(0)
    split = load_model(config).eval()
    assert torch.equal(rng, torch.get_rng_state()), 'Split changed initialization RNG'
    base_count = sum(p.numel() for p in base.parameters())
    split_count = sum(p.numel() for p in split.parameters())
    assert base_count == split.baseline_parameter_count
    for source, copied in zip(base.layers[-2:].parameters(), split.z_layers.parameters()):
        assert torch.equal(source, copied)
        assert source.data_ptr() != copied.data_ptr()
    for xy, z in zip(split.layers[-2:].parameters(), split.z_layers.parameters()):
        assert xy.data_ptr() != z.data_ptr()
    restored = load_model(config)
    restored.load_state_dict(split.state_dict(), strict=True)
    del restored
    try:
        split.load_state_dict(base.state_dict(), strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Baseline checkpoint must not silently load into split model')
    # Failed strict load can copy matching tensors; these are identical at init.
    base, split = base.to(opts.device), split.to(opts.device)
    x = torch.randn(opts.batch_size, opts.frames, 17, 3, device=opts.device)
    x[..., 2] = torch.rand_like(x[..., 2])
    with torch.no_grad():
        expected, actual = base(x), split(x)
    assert actual.shape == (opts.batch_size, opts.frames, 17, 3)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    max_diff = (actual - expected).abs().max().item()

    # Coordinate-specific backward paths must reach the shared prefix, but not
    # the other task's suffix. Use a short clip on CPU to bound reference-scan RAM.
    grad_x = x if opts.device == 'cuda' else x[:, :9]
    # gcn_t has a fixed frame dimension; create a small-T model for CPU backward.
    if opts.device == 'cpu' and opts.frames > 9:
        grad_config = deepcopy(config)
        grad_config.n_frames = 9
        grad_model = load_model(grad_config).train()
    else:
        grad_model = split.train()
    for axis in ('xy', 'z'):
        grad_model.zero_grad(set_to_none=True)
        output = grad_model(grad_x)
        loss = output[..., :2].square().mean() if axis == 'xy' else output[..., 2:].square().mean()
        loss.backward()
        selected = grad_model.layers[-2:] if axis == 'xy' else grad_model.z_layers
        other = grad_model.z_layers if axis == 'xy' else grad_model.layers[-2:]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected.parameters())
        assert any(p.grad.abs().sum() > 0 for p in selected.parameters())
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in other.parameters())
        assert grad_model.joints_embed.weight.grad.abs().sum() > 0

    # Exercise the baseline's active loss terms, clipping and an optimizer step.
    optimizer = torch.optim.AdamW(grad_model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    target = torch.randn(*grad_x.shape[:-1], 3, device=opts.device)
    target = target - target[..., :1, :]
    pred = grad_model(grad_x)
    training_loss = (loss_mpjpe(pred, target) + config.lambda_scale * n_mpjpe(pred, target)
                     + config.lambda_3d_velocity * loss_velocity(pred, target))
    training_loss.backward()
    # The original GCN declares batch_norm but does not call it in forward.
    # Preserve that baseline behavior; every new split parameter must be active.
    unused = {name for name, p in grad_model.named_parameters() if p.grad is None}
    assert unused == {'gcn_s.batch_norm.weight', 'gcn_s.batch_norm.bias',
                      'gcn_t.batch_norm.weight', 'gcn_t.batch_norm.bias'}, unused
    assert all(torch.isfinite(p.grad).all() for p in grad_model.parameters()
               if p.grad is not None)
    before = grad_model.z_head.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(grad_model.parameters(), config.grad_clip_norm,
                                   error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before, grad_model.z_head.weight)
    del optimizer, pred, training_loss, target, before

    def short_run(model):
        model.train()
        target = torch.randn_like(actual)
        target = target - target[..., :1, :]
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        def step():
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = (loss_mpjpe(pred, target) + config.lambda_scale * n_mpjpe(pred, target)
                    + config.lambda_3d_velocity * loss_velocity(pred, target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
        step()  # warm-up and allocate Adam state
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(opts.steps):
            step()
        torch.cuda.synchronize()
        return dict(seconds_per_step=(time.perf_counter() - start) / opts.steps,
                    peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20)

    result = dict(device=opts.device, input_shape=list(x.shape), output_shape=list(actual.shape),
                  baseline_params=base_count, total_params=split_count,
                  added_params=split_count-base_count, init_max_abs_diff=max_diff,
                  init_rng_unchanged=True, independent_parameters=True,
                  xy_z_gradient_isolation=True, shared_prefix_gradient=True,
                  strict_checkpoint_roundtrip=True,
                  original_active_loss_optimizer_step=True,
                  gradient_frames=grad_x.shape[1])
    if opts.device == 'cuda':
        # Keep only the tested model on GPU for comparable memory measurements.
        grad_model = None
        split.zero_grad(set_to_none=True)
        base.cpu()
        torch.cuda.empty_cache()
        result['split_short_run'] = short_run(split)
        split.cpu()
        base.cuda()
        torch.cuda.empty_cache()
        result['baseline_short_run'] = short_run(base)
    else:
        result['gpu_vram_speed'] = 'not measured; CPU reference is not a GPU benchmark'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
