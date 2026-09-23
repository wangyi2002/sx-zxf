"""Final-feature skeleton-relative Z adapter; no auxiliary loss."""
import math

import torch
from torch import nn

from data.const import H36M_BONES


def skeleton_path_matrix(edges, num_joints=17, root=0):
    """A[j,e]=1 when directed edge e lies on root -> j. Init-time only."""
    parents = {}
    for e, (parent, child) in enumerate(edges):
        if not (0 <= parent < num_joints and 0 <= child < num_joints):
            raise ValueError('Skeleton index out of range')
        if child == root or child in parents:
            raise ValueError('Each non-root joint must have exactly one parent')
        parents[child] = (parent, e)
    if len(edges) != num_joints - 1 or len(parents) != num_joints - 1:
        raise ValueError('Expected a rooted spanning tree')
    matrix = torch.zeros(num_joints, len(edges))
    for joint in range(num_joints):
        visited = set()
        current = joint
        while current != root:
            if current in visited or current not in parents:
                raise ValueError('Skeleton must be connected and acyclic')
            visited.add(current)
            current, edge = parents[current]
            matrix[joint, edge] = 1
    return matrix


class SkeletonRelativeDepthAdapter(nn.Module):
    def __init__(self, dim, ratio=0.25):
        super().__init__()
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError('depth_adapter_ratio must be finite and in (0, 1]')
        hidden = max(1, int(dim * ratio))
        edges = torch.tensor(H36M_BONES, dtype=torch.long)
        self.register_buffer('parent_indices', edges[:, 0].clone())
        self.register_buffer('child_indices', edges[:, 1].clone())
        self.register_buffer('path_matrix', skeleton_path_matrix(H36M_BONES))
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # First backward updates the last layer. Earlier layers receive branch
        # gradients after the last layer becomes nonzero; F is never detached.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features):
        # [B,T,17,C] -> [B,T,16,C] -> [B,T,16] -> [B,T,17]
        edge_features = (features.index_select(-2, self.child_indices)
                         - features.index_select(-2, self.parent_indices))
        bone_residual = self.mlp(edge_features).squeeze(-1)
        joint_residual = bone_residual @ self.path_matrix.to(dtype=bone_residual.dtype).T
        return bone_residual, joint_residual
