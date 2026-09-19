import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.depth_diagnostics import DepthDiagnostics, temporal_mask


class DepthMetricTests(unittest.TestCase):
    def run_case(self, frames, predictions, truth=None, actions=None, sources=None, threshold=10.):
        frames = np.asarray(frames)
        n = int(frames.max()) + 1
        actions = ['Walking'] * n if actions is None else actions
        sources = ['s_09_act_01_subact_01_cam01'] * n if sources is None else sources
        labels = {i: str(i) for i in range(17)}
        diag = DepthDiagnostics(frames, actions, sources, labels, threshold)
        for i, pred in enumerate(predictions):
            gt = np.zeros_like(pred) if truth is None else truth[i]
            diag.update(i, pred, gt)
        with tempfile.TemporaryDirectory() as tmp:
            summary = diag.save(tmp)
            self.assertEqual(len(list(Path(tmp).glob('*.csv'))), 4)
            self.assertEqual(json.loads((Path(tmp) / 'depth_summary.json').read_text()), summary)
        return summary

    def test_translation_removed_and_perfect_pose_counted(self):
        gt = np.random.default_rng(0).normal(size=(4, 17, 3)) * 20
        pred = gt + np.array([0., 0., 128.])
        summary = self.run_case([[0, 1, 2, 3]], [pred], [gt])
        self.assertEqual(summary['frames'], 4)
        self.assertAlmostEqual(summary['frame_weighted']['mpjpe_mm'], 0., places=10)

    def test_known_z_and_squared_share(self):
        pred = np.zeros((3, 17, 3))
        pred[:, 1, :] = [3., 0., 4.]
        s = self.run_case([[0, 1, 2]], [pred])['frame_weighted']
        self.assertAlmostEqual(s['mpjpe_mm'], 5 / 17)
        self.assertAlmostEqual(s['z_mae_mm'], 4 / 17)
        self.assertAlmostEqual(s['z_squared_error_fraction'], 16 / 25)

    def test_repeat_and_overlap_average_errors_not_predictions(self):
        a = np.zeros((3, 17, 3)); b = a.copy()
        a[:, 1, 2] = 2.; b[:, 1, 2] = -6.
        s = self.run_case([[0, 0, 1], [1, 2, 2]], [a, b])
        self.assertEqual(s['frames'], 3)
        self.assertAlmostEqual(s['frame_weighted']['z_mae_mm'], 4 / 17)

    def test_action_macro_differs_from_frame_average(self):
        pred = np.zeros((3, 17, 3)); pred[0, 1, 2] = 12.
        s = self.run_case([[0, 1, 2]], [pred], actions=['A', 'B', 'B'])
        self.assertAlmostEqual(s['frame_weighted']['z_mae_mm'], 4 / 17)
        self.assertAlmostEqual(s['action_macro']['z_mae_mm'], 6 / 17)

    def test_ordering_zero_prediction_is_wrong_and_small_gap_excluded(self):
        gt = np.zeros((3, 17, 3)); gt[:, 1, 2] = 20.
        pred = np.zeros_like(gt)
        s = self.run_case([[0, 1, 2]], [pred], [gt])['frame_weighted']
        self.assertEqual(s['ordering_accuracy'], 0.)
        self.assertAlmostEqual(s['ordering_coverage'], 16 / 136)
        s = self.run_case([[0, 1, 2]], [pred], [gt], threshold=20.)['frame_weighted']
        self.assertIsNone(s['ordering_accuracy'])

    def test_velocity_acceleration_and_boundaries(self):
        pred = np.zeros((4, 17, 3)); pred[:, 1, 2] = np.arange(4.) ** 2
        s = self.run_case([[0, 1, 2, 3]], [pred])['frame_weighted']
        self.assertAlmostEqual(s['z_velocity_error_mm_per_frame'], 3 / 17)
        self.assertAlmostEqual(s['z_acceleration_error_mm_per_frame2'], 2 / 17)
        ids = np.array([0, 1, 1, 3, 4])
        np.testing.assert_array_equal(temporal_mask(ids, np.array(['a','a','a','b','b']),
                                                   np.array(['A'] * 5), 1), [True, False, False, True])
        np.testing.assert_array_equal(temporal_mask(ids, np.array(['a'] * 5),
                                                   np.array(['A'] * 5), 2), [False, False, False])
        s = self.run_case([[0, 1, 2, 3]], [pred], sources=['a','a','b','b'])['frame_weighted']
        self.assertIsNone(s['z_acceleration_error_mm_per_frame2'])

    def test_incomplete_or_duplicate_clip_rejected(self):
        diag = DepthDiagnostics([[0, 1, 2]], ['A'] * 3, ['video'] * 3, {i: str(i) for i in range(17)})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                diag.save(tmp)
        diag.update(0, np.zeros((3, 17, 3)), np.zeros((3, 17, 3)))
        with self.assertRaises(ValueError):
            diag.update(0, np.zeros((3, 17, 3)), np.zeros((3, 17, 3)))


if __name__ == '__main__':
    unittest.main()
