"""Matched-size joint-feature Z adapter control for SRDA."""
import math

import torch
from torch import nn

from data.const import H36M_BONES


class JointDepthAdapter(nn.Module):
    def __init__(self, dim, ratio=0.25):
        super().__init__()
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError('depth_adapter_ratio must be finite and in (0, 1]')
        hidden = max(1, int(dim * ratio))
        # Skeleton indices are used ONLY for detached monitoring, not prediction.
        edges = torch.tensor(H36M_BONES, dtype=torch.long)
        self.register_buffer('parent_indices', edges[:, 0].clone())
        self.register_buffer('child_indices', edges[:, 1].clone())
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features):
        # [B,T,17,C] -> [B,T,17]; same MLP for all joints, root index=0.
        raw = self.mlp(features).squeeze(-1)
        joint_residual = raw - raw[..., :1]
        # No feature difference or tree accumulation in the prediction path.
        with torch.no_grad():
            bone_residual = (joint_residual.index_select(-1, self.child_indices)
                             - joint_residual.index_select(-1, self.parent_indices))
        return bone_residual, joint_residual
