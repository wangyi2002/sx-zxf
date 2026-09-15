"""SSI feature-only experiment: fixed H36M skeleton aggregation, no parameters."""

import torch
from torch import nn


# Same 17-joint ordering as loss.pose3d.get_limb_lens.
H36M_EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)


class FixedSkeletonFeatureFusion(nn.Module):
    """x'_j = x_j + sum_k P[j,k] x_k, applied to [B*T, C, J].

P = D^(-1/2) (A + I) D^(-1/2). No softmax: non-edges stay zero.
The fixed graph is shared across frames/channels and identical across layers.
It is regenerated from the skeleton, so is not stored in checkpoints.
"""

    def __init__(self, device=None):
        super().__init__()
        adjacency = torch.eye(17, device=device, dtype=torch.float32)
        for j, k in H36M_EDGES:
            adjacency[j, k] = adjacency[k, j] = 1.0
        inv_sqrt_degree = adjacency.sum(-1).rsqrt()
        graph = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
        self.register_buffer("graph", graph, persistent=False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 17:
            raise ValueError("SSI expects [batch*frames, channels, 17 joints]")
        # Rows are destinations, columns are source neighbors. No batch/time mixing.
        return x + torch.matmul(x, self.graph.to(dtype=x.dtype).T)
