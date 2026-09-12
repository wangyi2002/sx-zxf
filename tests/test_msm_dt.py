"""Run: python -m unittest discover -s tests -p 'test_msm_dt.py' -v"""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.modules.mamba import MAM
from model.modules.scan_backend import selective_scan_ref


def make_mam(enabled=False, mode="temporal", bias="legacy_double"):
    return MAM(16, d_state=8, d_conv=3, expand=1, mode=mode,
               temporal_msm=enabled, dt_bias_mode=bias)


class MSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_init_preserves_weights_output_and_input_gradient(self):
        for bias in ("legacy_double", "single"):
            torch.manual_seed(12)
            baseline = make_mam(bias=bias)
            torch.manual_seed(12)
            msm = make_mam(True, bias=bias)
            for key, value in baseline.state_dict().items():
                self.assertTrue(torch.equal(value, msm.state_dict()[key]), key)
            for length in (1, 7):
                u = torch.randn(2, length, 17, 16, requires_grad=True)
                v = u.detach().clone().requires_grad_()
                a, b = baseline(u), msm(v)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                ga = torch.autograd.grad(a.square().sum(), u)[0]
                gb = torch.autograd.grad(b.square().sum(), v)[0]
                torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_two_taps_boundaries_and_batch_isolation(self):
        mam = make_mam(True)
        with torch.no_grad():
            mam.msm_dt_weight.normal_()
        u = torch.randn(2, 7, 16)
        out = mam.motion_dt_logits(u).transpose(1, 2)
        current = F.linear(u, mam.msm_dt_weight[..., 1])
        previous = F.pad(F.linear(u[:, :-1], mam.msm_dt_weight[..., 0]), (0, 0, 1, 0))
        torch.testing.assert_close(out, current + previous)
        torch.testing.assert_close(out[:, 0], current[:, 0])
        changed = u.clone()
        changed[:, 4:] += 100
        changed[1] *= -3
        torch.testing.assert_close(mam.motion_dt_logits(changed)[0, :, :4],
                                   out[0, :4].T)

    def test_new_weight_learns_from_zero(self):
        mam = make_mam(True)
        u = torch.randn(2, 7, 17, 16)
        before = mam(u).detach()
        optimizer = torch.optim.AdamW(mam.parameters(), lr=1e-3)
        mam(u).square().mean().backward()
        grad = mam.msm_dt_weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertGreater(mam.msm_dt_weight.abs().sum().item(), 0)
        self.assertGreater((mam(u) - before).abs().max().item(), 0)

    def test_spatial_has_no_msm_parameters(self):
        mam = make_mam(True, mode="spatial")
        self.assertFalse(mam.temporal_msm)
        self.assertNotIn("msm_dt_weight", mam.state_dict())

    def test_bias_mode_changes_exactly_one_bias(self):
        legacy = make_mam()
        single = make_mam(bias="single")
        single.load_state_dict(legacy.state_dict())
        u = torch.randn(1, 7, 17, 16)
        captured = []

        def scan(*args, **kwargs):
            captured.append(args[1].detach() + kwargs["delta_bias"][None, :, None])
            return selective_scan_ref(*args, **kwargs)

        with patch("model.modules.mamba.selective_scan_fn", side_effect=scan):
            legacy(u)
            single(u)
        expected = legacy.dt_proj.bias[None, :, None].expand_as(captured[0])
        torch.testing.assert_close(captured[0] - captured[1], expected)
        with self.assertRaises(ValueError):
            make_mam(bias="typo")

    def test_reference_scan_matches_closed_form_and_has_correct_gradients(self):
        u = torch.ones(1, 2, 5, dtype=torch.float64)
        dt = torch.full_like(u, 0.2)
        A = torch.full((2, 1), -0.7, dtype=torch.float64)
        B = torch.full((1, 1, 5), 2.0, dtype=torch.float64)
        C = torch.full_like(B, 3.0)
        D = torch.full((2,), 0.4, dtype=torch.float64)
        actual = selective_scan_ref(u, dt, A, B, C, D)
        decay = torch.exp(torch.tensor(-0.14, dtype=torch.float64))
        expected = torch.stack([1.2 * sum(decay ** k for k in range(t + 1)) + 0.4
                                for t in range(5)]).expand_as(actual)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, d: selective_scan_ref(x, d, A, B, C, D, delta_softplus=True),
            (u.requires_grad_(), dt.requires_grad_())))


if __name__ == "__main__":
    unittest.main()
