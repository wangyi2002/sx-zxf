import unittest

import torch

from data.const import H36M_BONES
from model.modules.srda import SkeletonRelativeDepthAdapter, skeleton_path_matrix
from utils.depth_diagnostics import BONES


class SRDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_paths_and_root(self):
        self.assertIs(BONES, H36M_BONES)
        adapter = SkeletonRelativeDepthAdapter(128)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 4161)
        bone = torch.arange(1., 17.).reshape(1, 1, 16)
        joint = bone @ adapter.path_matrix.T
        self.assertEqual(joint[..., 0].item(), 0.)
        self.assertEqual(joint[..., 3].item(), 1 + 2 + 3)
        self.assertEqual(joint[..., 16].item(), 7 + 8 + 14 + 15 + 16)
        recovered = joint[..., adapter.child_indices] - joint[..., adapter.parent_indices]
        torch.testing.assert_close(recovered, bone, rtol=0, atol=0)
        self.assertEqual(set(dict(adapter.named_buffers())),
                         {'path_matrix', 'parent_indices', 'child_indices'})

    def test_shape_zero_init_and_gradient_after_update(self):
        adapter = SkeletonRelativeDepthAdapter(128)
        features = torch.randn(4, 243, 17, 128, requires_grad=True)
        edge = (features[..., adapter.child_indices, :]
                - features[..., adapter.parent_indices, :])
        self.assertEqual(tuple(edge.shape), (4, 243, 16, 128))
        bone, joint = adapter(features)
        self.assertEqual(tuple(bone.shape), (4, 243, 16))
        self.assertEqual(tuple(joint.shape), (4, 243, 17))
        self.assertEqual(bone.abs().max().item(), 0.)
        self.assertEqual(joint.abs().max().item(), 0.)
        target = torch.randn_like(joint)
        optimizer = torch.optim.SGD(adapter.parameters(), lr=.1)
        (joint - target).square().mean().backward()
        self.assertGreater(adapter.mlp[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(adapter.mlp[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual(features.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        features.grad = None
        _, joint = adapter(features)
        (joint - target).square().mean().backward()
        for p in adapter.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(joint[..., 0].abs().max().item(), 0)

    def test_feature_difference_and_buffers_move(self):
        adapter = SkeletonRelativeDepthAdapter(128).double()
        torch.nn.init.normal_(adapter.mlp[-1].weight)
        features = torch.randn(2, 3, 17, 128, dtype=torch.float64)
        offset = torch.randn(2, 3, 1, 128, dtype=torch.float64)
        a, b = adapter(features), adapter(features + offset)
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second)
        self.assertEqual(adapter.path_matrix.dtype, torch.float64)
        self.assertEqual(adapter.child_indices.dtype, torch.long)

    def test_invalid_tree(self):
        with self.assertRaises(ValueError):
            skeleton_path_matrix([(1, 2), (2, 1)], num_joints=3)
        with self.assertRaises(ValueError):
            skeleton_path_matrix([(0, 1), (2, 1)], num_joints=3)


if __name__ == '__main__':
    unittest.main()
