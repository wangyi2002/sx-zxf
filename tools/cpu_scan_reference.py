"""Explicit CPU smoke-test fallback; never imported by training or diagnosis.

Implements the real, ungrouped variable-B/C selective scan used in this repo.
This is a sequential reference for shape/finite checks, not a speed benchmark.
"""
import sys
import types

import torch
import torch.nn.functional as F


def selective_scan_cpu(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False, return_last_state=False):
    if u.device.type != 'cpu':
        raise RuntimeError('Smoke-test reference is CPU-only')
    if A.is_complex() or B.ndim != 3 or C.ndim != 3:
        raise NotImplementedError('Only the real variable-B/C scan used by MAM is supported')
    dtype = u.dtype
    u, delta, A, B, C = [x.float() for x in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = torch.zeros(u.shape[0], u.shape[1], A.shape[1], dtype=torch.float32)
    outputs = []
    for t in range(u.shape[-1]):
        dt = delta[:, :, t, None]
        state = torch.exp(dt * A[None]) * state + dt * B[:, None, :, t] * u[:, :, t, None]
        outputs.append((state * C[:, None, :, t]).sum(dim=-1))
    result = torch.stack(outputs, dim=-1)
    if D is not None:
        result = result + u * D.float()[None, :, None]
    if z is not None:
        result = result * F.silu(z)
    result = result.to(dtype)
    return (result, state) if return_last_state else result


def enable_cpu_reference():
    # Process-local import adapter. No repository model file or GPU path changes.
    if 'model.modules.mamba' in sys.modules:
        raise RuntimeError('Enable the CPU reference before importing the model')
    for name in ('mamba_ssm', 'mamba_ssm.ops'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    interface = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    interface.selective_scan_fn = selective_scan_cpu
    sys.modules[interface.__name__] = interface
