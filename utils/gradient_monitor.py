"""Global gradient clipping and epoch aggregates, identical for baseline/SSI."""

import math

import torch


class GradientMonitor:
    def __init__(self, max_norm=0.0):
        self.max_norm = float(max_norm)
        if not math.isfinite(self.max_norm) or self.max_norm < 0:
            raise ValueError("grad_clip_norm must be finite and >= 0")
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.clipped = 0

    def clip(self, model):
        # Backward has completed (including DataParallel gradient reduction).
        # If AMP is introduced later, unscale before calling this method.
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm if self.max_norm else float("inf"),
            norm_type=2.0, error_if_nonfinite=True,
        ).item()
        self.count += 1
        self.total += norm
        self.maximum = max(self.maximum, norm)
        self.clipped += int(self.max_norm > 0 and norm > self.max_norm)

    def summary(self):
        return {
            "grad/global_norm_mean": self.total / max(1, self.count),
            "grad/global_norm_max": self.maximum,
            "grad/clipped_steps_fraction": self.clipped / max(1, self.count),
            "grad/clip_threshold": self.max_norm,
        }
