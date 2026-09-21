"""CPU checks: python -m unittest discover -s tests -p 'test_relative_depth_loss.py'.

The training-loop test executes train.py's actual train_one_epoch AST with a
small pose predictor, avoiding CUDA/Mamba and wandb imports. It does not replace
an end-to-end run on the real H36M dataset.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from data.const import H36M_BONES
from loss import pose3d
from utils.depth_diagnostics import BONES


class Meter:
    def __init__(self):
        self.sum = self.count = self.avg = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def training_loop():
    source = Path(__file__).resolve().parents[1] / 'train.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'train_one_epoch')
    namespace = dict(vars(pose3d), torch=torch, tqdm=lambda loader: loader)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def original_loss(pred, target, args):
    # Independent reference: baseline e1d208b's original loss expression.
    return (pose3d.loss_mpjpe(pred, target)
            + args.lambda_scale * pose3d.n_mpjpe(pred, target)
            + args.lambda_3d_velocity * pose3d.loss_velocity(pred, target)
            + args.lambda_lv * pose3d.loss_limb_var(pred)
            + args.lambda_lg * pose3d.loss_limb_gt(pred, target)
            + args.lambda_a * pose3d.loss_angle(pred, target)
            + args.lambda_av * pose3d.loss_angle_velocity(pred, target))


class RelativeDepthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def test_shared_skeleton_and_existing_limb_topology(self):
        self.assertIs(BONES, H36M_BONES)
        self.assertEqual(len(BONES), 16)
        node = next(n for n in ast.walk(ast.parse(Path(pose3d.__file__).read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == 'get_limb_lens')
        assignment = next(n for n in node.body if isinstance(n, ast.Assign)
                          and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'limbs_id')
        self.assertEqual([tuple(e) for e in ast.literal_eval(assignment.value)], BONES)

    def test_zero_and_known_value(self):
        target = torch.zeros(4, 243, 17, 3)
        self.assertEqual(pose3d.relative_depth_loss(target, target).item(), 0.)
        pred = target.clone()
        pred[..., 3, 2] = 2.  # Leaf joint: exactly one of 16 edges has error 2.
        self.assertEqual(pose3d.relative_depth_loss(pred, target).item(), 2. / 16)
        pred[..., :2] = 100.  # X/Y must not enter the auxiliary loss.
        self.assertEqual(pose3d.relative_depth_loss(pred, target).item(), 2. / 16)

    def test_shape_gradient_and_translation_invariance(self):
        target = torch.randn(4, 243, 17, 3)
        pred = torch.randn_like(target, requires_grad=True)
        indices = torch.tensor(BONES)
        for value in (pred, target):
            delta = value[..., indices[:, 0], 2] - value[..., indices[:, 1], 2]
            self.assertEqual(tuple(delta.shape), (4, 243, 16))
        loss = pose3d.relative_depth_loss(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad[..., 2].abs().sum().item(), 0.)
        self.assertEqual(pred.grad[..., :2].abs().sum().item(), 0.)
        offset = torch.randn(4, 243, 1, 3)
        torch.testing.assert_close(pose3d.relative_depth_loss(pred + offset, target), loss)
        torch.testing.assert_close(pred.grad[..., 2].sum(-1), torch.zeros(4, 243), atol=1e-8, rtol=0)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            pose3d.relative_depth_loss(torch.zeros(1, 3, 17), torch.zeros(1, 3, 17))

    def test_training_enabled_disabled_and_missing_config(self):
        args = SimpleNamespace(root_rel=True, lambda_scale=.5, lambda_3d_velocity=20.,
                               lambda_lv=0., lambda_lg=0., lambda_a=0., lambda_av=0.,
                               grad_clip_norm=1., relative_depth_weight=.1)
        x, y = torch.randn(4, 243, 17, 3), torch.randn(4, 243, 17, 3)
        initial = nn.Linear(3, 3)
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                if enabled is not None:
                    args.use_relative_depth_loss = enabled
                model, reference = copy.deepcopy(initial), copy.deepcopy(initial)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
                ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.0005, weight_decay=.01)
                ns = training_loop()
                spy = Mock(wraps=pose3d.relative_depth_loss) if enabled else Mock(
                    side_effect=AssertionError('Disabled auxiliary loss was called'))
                ns['relative_depth_loss'] = spy
                names = ['3d_pose', '3d_scale', '3d_velocity', 'lv', 'lg', 'angle',
                         'angle_velocity', 'total', 'original', 'relative_depth',
                         'grad_norm', 'grad_clip_fraction']
                meters = {name: Meter() for name in names}
                ns['train_one_epoch'](args, model, [(x, y)], optimizer, 'cpu', meters)

                target = y - y[..., :1, :]
                pred = reference(x)
                base = original_loss(pred, target, args)
                total = base
                if enabled:
                    total = base + .1 * pose3d.relative_depth_loss(pred, target)
                ref_optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
                ref_optimizer.step()
                self.assertEqual(meters['total'].avg, total.item())
                self.assertEqual(meters['original'].avg, base.item())
                self.assertEqual(spy.call_count, int(bool(enabled)))
                if enabled:
                    self.assertEqual(tuple(spy.call_args.args[0].shape), (4, 243, 17, 3))
                    torch.testing.assert_close(spy.call_args.args[1], target, rtol=0, atol=0)
                for p, q in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                    for key in optimizer.state[p]:
                        torch.testing.assert_close(optimizer.state[p][key], ref_optimizer.state[q][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
