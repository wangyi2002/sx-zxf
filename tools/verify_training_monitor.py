"""Run the actual train_one_epoch with synthetic data and real pose losses.

Loads the function AST from train.py to avoid importing unused dataset and
W&B service dependencies. The model, losses, optimizer, clipping and monitor
are real. This is not a dataset evaluation or a W&B upload test.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm

import loss.pose3d as pose_losses
from utils.learning import AverageMeter, load_model
from utils.training_monitor import TrainingMonitor, clip_and_monitor_gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--frames', type=int, default=243)
    opts = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    args = EasyDict(yaml.safe_load((ROOT / 'configs/h36m/MotionAGFormer-msm-dt.yaml').read_text()))
    args.n_frames = opts.frames
    model = load_model(args).to(opts.device)
    before_keys = set(model.state_dict())
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    tree = ast.parse((ROOT / 'train.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_one_epoch')
    scope = dict(vars(pose_losses), torch=torch, tqdm=tqdm, TrainingMonitor=TrainingMonitor,
                 clip_and_monitor_gradients=clip_and_monitor_gradients)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / 'train.py'), 'exec'), scope)
    losses = {key: AverageMeter() for key in ['3d_pose', '3d_scale', '3d_velocity', 'lv', 'lg', 'angle', 'angle_velocity', 'total']}
    x = torch.randn(1, args.n_frames, 17, 3) * 0.1
    target = torch.randn_like(x) * 0.1
    stats = scope['train_one_epoch'](args, model, [(x, target)], optimizer, opts.device, losses)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert before_keys == set(model.state_dict())
    assert all(getattr(m, 'dt_observer', None) is None for m in model.modules())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    post_norm = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm().item()
    assert post_norm <= args.grad_clip_norm + 1e-5
    assert sum(key.endswith('/dt_mean') for key in stats) == 13
    assert stats['grad/msm_norm_mean'] > 0
    assert 'pose/scale_denominator_min' in stats
    print(json.dumps({'device': opts.device, 'input_shape': list(x.shape), 'params': param_count,
                      'pose_loss': losses['total'].avg, 'diagnostic_count': len(stats),
                      'gradient_norm_after_clip': post_norm,
                      **{k: v for k, v in stats.items() if k.startswith('grad/')},
                      'finite_optimizer_step': True, 'state_dict_unchanged': True,
                      'wandb_upload_tested': False}, indent=2))


if __name__ == '__main__':
    main()
