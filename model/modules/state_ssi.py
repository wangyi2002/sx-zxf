"""Fixed graph fusion of ALL spatial states before C readout.

Recurrence is unchanged: h_j=a_j*h_{j-1}+b_j*x_j.
Readout is C_j*(h_j + sum_k P[j,k]*h_k) + D*x_j.
Fused states are NOT fed back into recurrence. Noncausal across spatial joints.
This replaces the scan call only for enabled spatial modules, using differentiable
PyTorch operations on CPU/CUDA, not a patch to the installed mamba_ssm package.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _scan(u, delta, A, B, C, D, bias, graph):
    dtype = u.dtype
    work = torch.float64 if dtype == torch.float64 else torch.float32
    u, delta, A, B, C, D, bias, graph = [
        v.to(work) for v in (u, delta, A, B, C, D, bias, graph)]
    delta = F.softplus(delta + bias[None, :, None])
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[-1])
    states = []
    for j in range(u.shape[-1]):
        step = delta[:, :, j, None]
        state = torch.exp(step * A[None]) * state + step * B[:, None, :, j] * u[:, :, j, None]
        states.append(state)
    h = torch.stack(states, dim=-1)  # [batch*frames, channels, state, joints]
    fused = h + torch.matmul(h, graph.T)
    return ((fused * C[:, None]).sum(2) + D[None, :, None] * u).to(dtype)


def state_ssi_scan(u, delta, A, B, C, D, bias, graph, chunk_size=32):
    if u.shape[-1] != 17 or graph.shape != (17, 17):
        raise ValueError('State SSI requires 17 spatial joints')
    outputs = []
    for start in range(0, u.shape[0], chunk_size):
        sl = slice(start, start + chunk_size)
        args = (u[sl], delta[sl], A, B[sl], C[sl], D, bias, graph)
        # Recompute state tensors during backward to bound retained activations.
        if torch.is_grad_enabled() and any(v.requires_grad for v in args):
            out = checkpoint(_scan, *args, use_reentrant=False)
        else:
            out = _scan(*args)
        outputs.append(out)
    return torch.cat(outputs, dim=0)
