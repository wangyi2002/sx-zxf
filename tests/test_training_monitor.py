import copy
import math
import unittest

import torch

from model.modules.mamba import MAM
from utils.training_monitor import TrainingMonitor, clip_and_monitor_gradients


class MonitorTests(unittest.TestCase):
    def test_global_clip_and_disable(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[3., 4.]])
        monitor = TrainingMonitor(0)
        self.assertEqual(clip_and_monitor_gradients(model, 1, monitor), 5)
        self.assertLessEqual(model.weight.grad.norm().item(), 1)
        self.assertEqual(monitor.summary()['grad/clipped_steps_fraction'], 1)
        model.weight.grad = torch.tensor([[3., 4.]])
        clip_and_monitor_gradients(model, 0, TrainingMonitor(0))
        torch.testing.assert_close(model.weight.grad, torch.tensor([[3., 4.]]))

    def test_nonfinite_gradients_raise(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[float('nan'), 1.]])
        with self.assertRaises(RuntimeError):
            clip_and_monitor_gradients(model, 1, TrainingMonitor(0))

    def test_observation_preserves_forward_backward_and_state(self):
        torch.set_num_threads(1)
        model = MAM(16, d_state=8, expand=1, d_conv=3, mode='temporal', temporal_msm=True)
        with torch.no_grad():
            model.msm_dt_weight.normal_(std=0.01)
        other = copy.deepcopy(model)
        monitor = TrainingMonitor(2)
        monitor.attach(model)
        monitor.begin_batch(0)
        u = torch.randn(1, 5, 17, 16)
        a, b = model(u), other(u)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        self.assertEqual(model.state_dict().keys(), other.state_dict().keys())
        self.assertTrue(monitor.summary())
        self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in monitor.summary().values()))
        previous = copy.deepcopy(monitor.values)
        monitor.begin_batch(1)
        model(u)
        self.assertEqual(previous, monitor.values)
        monitor.close()
        self.assertIsNone(model.dt_observer)

    def test_effective_dt_accounts_for_scan_bias(self):
        monitor = TrainingMonitor(1)
        monitor.begin_batch(0)
        monitor.dt('layer', torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(2))
        stats = monitor.summary()
        self.assertAlmostEqual(stats['msm/layer/dt_mean'], torch.nn.functional.softplus(torch.tensor(1.)).item(), places=6)
        self.assertEqual(stats['msm/layer/dt_to_baseline_mean_ratio'], 1)
        self.assertEqual(stats['msm/layer/residual_rms'], 0)

    def test_collapse_diagnostics_and_aggregation(self):
        monitor = TrainingMonitor(1)
        monitor.begin_batch(0)
        monitor.pose(torch.zeros(1, 5, 17, 3), torch.randn(1, 5, 17, 3))
        stats = monitor.summary()
        self.assertEqual(stats['pose/limb_ratio_min'], 0)
        self.assertEqual(stats['pose/scale_denominator_min'], 0)
        monitor.add('max', 2, 'max')
        monitor.add('max', 1, 'max')
        monitor.add('mean', 2, weight=3)
        monitor.add('mean', 6, weight=1)
        self.assertEqual(monitor.summary()['max'], 2)
        self.assertEqual(monitor.summary()['mean'], 3)


if __name__ == '__main__':
    unittest.main()
