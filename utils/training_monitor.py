"""Detached training diagnostics, including DataParallel replica callbacks.

Only scalar aggregates survive a batch. No activations/graphs are retained.
DT/pose statistics are sampled; gradient clipping covers every optimizer step.
"""

import math
import threading

import torch
import torch.nn.functional as F

from loss.pose3d import get_limb_lens
from model.modules.mamba import MAM


class TrainingMonitor:
    def __init__(self, every=100):
        self.every = int(every)
        if self.every < 0:
            raise ValueError("monitor_every must be >= 0 (0 disables DT/pose sampling)")
        self.active = False
        self.values = {}
        self.lock = threading.Lock()
        self.attached = []

    def add(self, key, value, reduction="mean", weight=1):
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite training diagnostic: {key}={value}")
        with self.lock:
            if key not in self.values:
                self.values[key] = [0.0, 0, float("inf"), -float("inf"), reduction]
            stat = self.values[key]
            stat[0] += value * weight
            stat[1] += weight
            stat[2] = min(stat[2], value)
            stat[3] = max(stat[3], value)

    def summary(self):
        with self.lock:
            return {key: (v[2] if v[4] == "min" else v[3] if v[4] == "max"
                          else v[0] / v[1]) for key, v in self.values.items()}

    def attach(self, model):
        if self.every == 0:
            return
        for name, module in model.named_modules():
            if isinstance(module, MAM) and module.mode == "temporal":
                # A plain callback is shared by DataParallel replicas; updates
                # below are locked and never rely on replica attribute writes.
                key = name.removeprefix("module.").replace(".", "_")
                old = getattr(module, "dt_observer", None)
                self.attached.append((module, old))
                module.dt_observer = lambda dt, residual, bias, key=key: self.dt(key, dt, residual, bias)

    def close(self):
        self.active = False
        for module, old in self.attached:
            module.dt_observer = old
        self.attached.clear()

    def begin_batch(self, step):
        self.active = self.every > 0 and step % self.every == 0

    @torch.no_grad()
    def dt(self, name, logits, residual, bias):
        if not self.active:
            return
        logits = logits.detach().float()
        bias = bias.detach().float()[None, :, None]
        effective = F.softplus(logits + bias)
        prefix = f"msm/{name}/"
        self.add(prefix + "dt_mean", effective.mean().item(), weight=effective.numel())
        self.add(prefix + "dt_min", effective.min().item(), "min")
        self.add(prefix + "dt_max", effective.max().item(), "max")
        # Deterministic bounded sample for p99; means/min/max above use all values.
        flat = effective.flatten()
        stride = max(1, math.ceil(flat.numel() / 4096))
        self.add(prefix + "dt_p99_sampled", torch.quantile(flat[::stride], 0.99).item(), "max")
        self.add(prefix + "dt_below_1e-5_fraction", (effective < 1e-5).float().mean().item(), weight=flat.numel())
        if residual is not None:
            residual = residual.detach().float()
            base = logits - residual + bias
            residual_rms = residual.square().mean().sqrt()
            base_rms = base.square().mean().sqrt()
            self.add(prefix + "residual_rms", residual_rms.item(), weight=flat.numel())
            self.add(prefix + "residual_to_base_logit_rms", (residual_rms / base_rms.clamp_min(1e-12)).item(), "max")
            base_dt = F.softplus(base).mean()
            self.add(prefix + "dt_to_baseline_mean_ratio", (effective.mean() / base_dt.clamp_min(1e-12)).item(), "max")

    @torch.no_grad()
    def pose(self, pred, target):
        if not self.active:
            return
        pred, target = pred.detach().float(), target.detach().float()
        pred_limb, target_limb = get_limb_lens(pred), get_limb_lens(target)
        self.add("pose/pred_limb_mean", pred_limb.mean().item(), weight=pred_limb.numel())
        self.add("pose/target_limb_mean", target_limb.mean().item(), weight=target_limb.numel())
        ratio = pred_limb.mean() / target_limb.mean().clamp_min(1e-12)
        self.add("pose/limb_ratio_mean", ratio.item())
        self.add("pose/limb_ratio_min", ratio.item(), "min")
        self.add("pose/pred_rms", pred.square().mean().sqrt().item())
        # Exact denominator used by n_mpjpe, BEFORE any diagnostic clamp.
        denominator = pred.square().sum(-1).mean(-1)
        self.add("pose/scale_denominator_min", denominator.min().item(), "min")
        self.add("pose/scale_denominator_mean", denominator.mean().item(), weight=denominator.numel())
        if pred.shape[1] > 1:
            self.add("pose/pred_velocity_rms", pred.diff(dim=1).square().mean().sqrt().item())
            self.add("pose/target_velocity_rms", target.diff(dim=1).square().mean().sqrt().item())


def clip_and_monitor_gradients(model, max_norm, monitor):
    """Call after backward, before optimizer.step (after unscale if AMP is added).

0 disables clipping but still measures the global norm. Nonfinite gradients
raise before updating weights; clipping is not a NaN repair mechanism.
"""
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("grad_clip_norm must be finite and >= 0")
    parameters = [p for p in model.parameters() if p.grad is not None]
    msm_grads = [p.grad.detach().float().norm() for n, p in model.named_parameters()
                 if n.endswith("msm_dt_weight") and p.grad is not None]
    if msm_grads:
        msm_norm = torch.stack(msm_grads).norm().item()
        monitor.add("grad/msm_norm_mean", msm_norm)
        monitor.add("grad/msm_norm_max", msm_norm, "max")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm if max_norm else float("inf"),
                                         error_if_nonfinite=True).item()
    monitor.add("grad/global_norm_mean", norm)
    monitor.add("grad/global_norm_max", norm, "max")
    monitor.add("grad/clipped_steps_fraction", float(max_norm > 0 and norm > max_norm))
    monitor.add("grad/clip_threshold", max_norm)
    return norm
