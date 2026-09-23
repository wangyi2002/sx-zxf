"""Synthetic short SRDA checks/benchmark, never starts dataset training.

GPU: python tools/smoke_srda.py --device cuda --batch-size 4 --steps 5
CPU: python tools/smoke_srda.py --device cpu --batch-size 1 --steps 2
CPU uses the existing explicit scan reference; its timings are not GPU estimates.
"""
import argparse
import ast
import copy
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/h36m/MotionAGFormer-srda.yaml')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', default='depth_outputs/srda_smoke.json')
    opts = parser.parse_args()
    if opts.steps < 1 or opts.batch_size < 1:
        parser.error('steps and batch-size must be positive')
    if opts.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; run on the training GPU or explicitly select --device cpu')
    if opts.device == 'cpu':
        from tools.cpu_scan_reference import enable_cpu_reference
        enable_cpu_reference()
        torch.set_num_threads(2)
    from utils.learning import load_model, AverageMeter
    from utils.tools import get_config
    from loss import pose3d

    config = get_config(opts.config)
    if not config.use_skeleton_relative_depth_adapter or getattr(config, 'use_relative_depth_loss', False):
        parser.error('Use the SRDA-on, relative-depth-loss-off experiment config')
    base_config = copy.deepcopy(config)
    base_config.use_skeleton_relative_depth_adapter = False
    torch.manual_seed(0)
    baseline = load_model(base_config)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(0)
    model = load_model(config)
    assert torch.equal(baseline_rng, torch.get_rng_state()), 'Adapter changed baseline RNG stream'
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    counts = dict(baseline=sum(p.numel() for p in baseline.parameters()),
                  adapter=sum(p.numel() for p in model.depth_adapter.parameters()),
                  total=sum(p.numel() for p in model.parameters()))
    assert counts['total'] - counts['baseline'] == counts['adapter'] == 4161
    # Strict off-model checkpoint compatibility.
    load_model(base_config).load_state_dict(baseline.state_dict(), strict=True)

    device = torch.device(opts.device)
    inputs = torch.randn(opts.batch_size, config.n_frames, 17, 3, device=device)
    inputs[..., 2] = torch.rand_like(inputs[..., 2])
    baseline.to(device).eval()
    with torch.no_grad():
        base_pose = baseline(inputs)
    baseline.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model.to(device).eval()
    shapes = {}
    def shape_hook(name):
        def hook(module, args, output):
            shapes[name] = list(output.shape)
        return hook
    hooks = [model.norm.register_forward_hook(shape_hook('features')),
             model.depth_adapter.mlp[0].register_forward_pre_hook(
                 lambda module, args: shapes.update(edge_features=list(args[0].shape)))]
    with torch.no_grad():
        pose, stats = model(inputs, return_depth_stats=True)
    for hook in hooks:
        hook.remove()
    assert torch.equal(base_pose, pose), 'Zero-init output differs from baseline'
    assert stats[0, :2].abs().max().item() == 0
    assert torch.isfinite(pose).all()
    shapes.update(bone_residual=[opts.batch_size, config.n_frames, 16],
                  joint_residual=[opts.batch_size, config.n_frames, 17], pose_final=list(pose.shape))
    model.cpu()
    del base_pose, pose, stats

    # Execute the real production epoch body, without importing wandb or data.
    source = Path(__file__).resolve().parents[1] / 'train.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'train_one_epoch')
    namespace = dict(vars(pose3d), torch=torch, tqdm=lambda x: x)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    names = ['3d_pose', '3d_scale', '3d_velocity', 'lv', 'lg', 'angle', 'angle_velocity',
             'total', 'grad_norm', 'grad_clip_fraction', 'depth_adapter_bone_residual_abs_mean',
             'depth_adapter_joint_residual_abs_mean']
    target = torch.randn_like(inputs)
    timings = {}
    # CPU full-reference backward is expensive; CPU benchmark is forward-only.
    for label, net, cfg in [('baseline', baseline, base_config), ('srda', model, config)]:
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        meters = {name: AverageMeter() for name in names}
        def step():
            if device.type == 'cuda':
                namespace['train_one_epoch'](cfg, net, [(inputs, target)], optimizer, device, meters)
            else:
                net.eval()
                with torch.no_grad():
                    result = net(inputs)
                    assert torch.isfinite(result).all()
        step()  # warm-up; CUDA also allocates optimizer state
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(opts.steps):
            step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings[label] = dict(seconds_per_iteration=(time.perf_counter() - start) / opts.steps,
                             peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                             peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if device.type == 'cuda' else None)
        if label == 'srda' and device.type == 'cuda':
            for p in net.depth_adapter.parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            assert meters['depth_adapter_bone_residual_abs_mean'].avg > 0
        net.cpu()
        del optimizer
    result = dict(parameters=counts, shapes=shapes, zero_init_equal=True,
                  baseline_rng_preserved=True, device=opts.device,
                  mode='synthetic training, original loss' if device.type == 'cuda' else 'CPU reference forward only',
                  timings=timings, steps=opts.steps, batch_size=opts.batch_size)
    path = Path(opts.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
