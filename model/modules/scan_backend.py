"""CUDA uses the original Mamba kernel; CPU uses a differentiable recurrence.

The CPU path is for correctness/smoke checks, not training performance. It
implements the real SSM (no identity/mock replacement), limited to the real,
ungrouped B/C tensors used by this repository's MAM.
"""

import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False):
    if u.ndim != 3 or delta.shape != u.shape or u.shape[-1] == 0:
        raise ValueError("Expected matching nonempty u/delta [batch, channels, time]")
    if B.ndim != 3 or C.shape != B.shape:
        raise ValueError("CPU reference supports variable B/C [batch, state, time]")
    output_dtype = u.dtype
    # Match the kernel's float32 accumulation for low precision; keep float64
    # available for numerical gradient checks.
    work_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u, delta, A, B, C = [v.to(work_dtype) for v in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.to(work_dtype)[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[-1])
    outputs = []
    for t in range(u.shape[-1]):
        step = delta[:, :, t, None]
        state = (torch.exp(step * A[None]) * state
                 + step * B[:, None, :, t] * u[:, :, t, None])
        outputs.append((state * C[:, None, :, t]).sum(-1))
    out = torch.stack(outputs, dim=-1)
    if D is not None:
        out = out + u * D.to(work_dtype)[None, :, None]
    if z is not None:
        out = out * F.silu(z.to(work_dtype))
    return out.to(output_dtype)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None,
                      delta_bias=None, delta_softplus=False):
    if u.device.type == "cpu":
        return selective_scan_ref(u, delta, A, B, C, D, z,
                                  delta_bias, delta_softplus)
    if u.device.type != "cuda":
        raise ValueError("MAM scan supports CPU verification or CUDA training")
    # Lazy import keeps CPU verification independent of CUDA extensions.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as cuda_scan
    return cuda_scan(u, delta, A, B, C, D, z=z,
                     delta_bias=delta_bias, delta_softplus=delta_softplus)
