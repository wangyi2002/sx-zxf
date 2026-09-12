"""Verify the real full model, parameter delta, and one synthetic optimizer step.

CPU uses an explicit SSM recurrence. --device cuda also compares the installed
CUDA scan's outputs/gradients against that recurrence before model verification.
No dataset/checkpoint required; this is not an accuracy or throughput benchmark.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from easydict import EasyDict

from model.modules.mamba import MAM
from model.modules.scan_backend import selective_scan_fn, selective_scan_ref
from utils.learning import load_model


def check_cuda_scan():
    for length in (17, 243):
        torch.manual_seed(4)
        values = [torch.randn(2, 8, length, device="cuda"),
                  torch.randn(2, 8, length, device="cuda"),
                  -torch.rand(8, 8, device="cuda"),
                  torch.randn(2, 8, length, device="cuda"),
                  torch.randn(2, 8, length, device="cuda"),
                  torch.randn(8, device="cuda"),
                  torch.randn(8, device="cuda")]
        fused_inputs = [x.requires_grad_() for x in values]
        ref_inputs = [x.detach().clone().requires_grad_() for x in values]

        def run(fn, args):
            return fn(*args[:6], delta_bias=args[6], delta_softplus=True)

        fused = run(selective_scan_fn, fused_inputs)
        reference = run(selective_scan_ref, ref_inputs)
        torch.testing.assert_close(fused, reference, rtol=3e-3, atol=3e-3)
        upstream = torch.randn_like(fused)
        ga = torch.autograd.grad(fused, fused_inputs, upstream)
        gb = torch.autograd.grad(reference, ref_inputs, upstream)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=5e-3, atol=5e-3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--config", default="configs/h36m/MotionAGFormer-msm-dt.yaml")
    parser.add_argument("--frames", type=int, help="Optional shorter sequence for smoke checks")
    opts = parser.parse_args()
    torch.set_num_threads(1)
    if opts.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        check_cuda_scan()
    with open(opts.config) as f:
        args = EasyDict(yaml.safe_load(f))
    if opts.frames is not None:
        args.n_frames = opts.frames
    args.temporal_msm = True
    base_args = copy.deepcopy(args)
    base_args.temporal_msm = False
    torch.manual_seed(0)
    baseline = load_model(base_args).to(opts.device).eval()
    torch.manual_seed(0)
    model = load_model(args).to(opts.device).eval()
    old_state = baseline.state_dict()
    for name, value in old_state.items():
        assert torch.equal(value, model.state_dict()[name]), name
    # Loading an original checkpoint must miss ONLY the new zero-initialized weights.
    loaded = model.load_state_dict(old_state, strict=False)
    assert not loaded.unexpected_keys
    assert loaded.missing_keys and all(n.endswith("msm_dt_weight") for n in loaded.missing_keys)
    spatial = [m for m in model.modules() if isinstance(m, MAM) and m.mode == "spatial"]
    temporal = [m for m in model.modules() if isinstance(m, MAM) and m.mode == "temporal"]
    assert all(not m.temporal_msm for m in spatial)
    assert all(m.temporal_msm for m in temporal)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    expected_delta = sum(m.msm_dt_weight.numel() for m in temporal)
    assert model_params - baseline_params == expected_delta
    x = torch.randn(1, args.n_frames, args.num_joints, args.dim_in, device=opts.device)
    with torch.no_grad():
        before = model(x)
        reference = baseline(x)
        torch.testing.assert_close(before, reference, rtol=0, atol=0)
    del baseline, old_state, reference
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    target = torch.randn_like(before)
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    assert out.shape == (1, args.n_frames, args.num_joints, args.dim_out)
    assert torch.isfinite(out).all()
    loss = (out - target).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(m.msm_dt_weight.grad.abs().sum() > 0 for m in temporal)
    optimizer.step()
    assert all(m.msm_dt_weight.abs().sum() > 0 for m in temporal)
    with torch.no_grad():
        after = model(x)
        assert torch.isfinite(after).all()
        assert (after - before).abs().max() > 0
    print(json.dumps({"device": opts.device, "torch": torch.__version__,
                      "input_shape": list(x.shape), "output_shape": list(out.shape),
                      "baseline_params": baseline_params, "msm_params": model_params,
                      "added_params": expected_delta, "temporal_msm_modules": len(temporal),
                      "zero_init_max_error": 0.0, "synthetic_mse": loss.item(),
                      "all_msm_gradients_nonzero": True, "optimizer_step": "passed",
                      "cuda_scan_parity": "passed" if opts.device == "cuda" else "not run"},
                     indent=2))


if __name__ == "__main__":
    main()
