"""Small-backbone CPU integration, with the explicit reference scan only."""
import copy
import unittest

import torch


class SRDAIntegrationTests(unittest.TestCase):
    def test_toggle_initialization_and_backbone_branch_gradient(self):
        from tools.cpu_scan_reference import enable_cpu_reference
        import sys
        if 'model.modules.mamba' not in sys.modules:
            enable_cpu_reference()
        from utils.tools import get_config
        from utils.learning import load_model
        torch.set_num_threads(2)
        config = get_config('configs/h36m/MotionAGFormer-srda.yaml')
        # Same production backbone/adapter path, smaller dimensions for CPU backward.
        config.n_layers, config.n_frames, config.dim_feat, config.dim_rep = 1, 9, 16, 32
        base_config = copy.deepcopy(config)
        base_config.use_skeleton_relative_depth_adapter = False
        torch.manual_seed(3)
        baseline = load_model(base_config).eval()
        base_rng = torch.get_rng_state().clone()
        torch.manual_seed(3)
        model = load_model(config).eval()
        self.assertTrue(torch.equal(base_rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        x = torch.randn(2, 9, 17, 3)
        initial_head = {}
        def initial_hook(module, inputs, output):
            initial_head['pose'] = output
        initial_handle = model.head.register_forward_hook(initial_hook)
        with torch.no_grad():
            pose, stats = model(x, return_depth_stats=True)
            # Same-forward zero correction is exact. Separate model executions
            # may differ at floating-point roundoff on some CPU backends.
            self.assertTrue(torch.equal(initial_head['pose'], pose))
            torch.testing.assert_close(baseline(x), pose, rtol=1e-6, atol=1e-7)
            self.assertEqual(stats[0, :2].abs().sum().item(), 0.)
            self.assertEqual(tuple(stats.shape), (1, 4))
        initial_handle.remove()

        captured = {}
        def head_hook(module, inputs, output):
            captured['base_pose'] = output
        hook = model.head.register_forward_hook(head_hook)
        optimizer = torch.optim.SGD(model.depth_adapter.parameters(), lr=.1)
        target = torch.randn(2, 9, 17)
        for step in range(2):
            model.zero_grad(set_to_none=True)
            pose = model(x)
            # Cancel the direct pose-head path using the LIVE head output, so
            # the backbone gradient below can only come through the SRDA path.
            residual = pose[..., 2] - captured['base_pose'][..., 2]
            self.assertTrue(torch.equal(pose[..., :2], captured['base_pose'][..., :2]))
            self.assertEqual(residual[..., 0].abs().max().item(), 0.)
            (residual - target).square().mean().backward()
            grad = model.joints_embed.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            if step == 0:
                self.assertEqual(grad.abs().sum().item(), 0.)
            else:
                self.assertGreater(grad.abs().sum().item(), 0.)
                for parameter in model.depth_adapter.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.)
            optimizer.step()
        hook.remove()
        # Even after learning nonzero adapter weights, off recovers the base path.
        model.depth_adapter = None
        with torch.no_grad():
            torch.testing.assert_close(model(x), baseline(x), rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
