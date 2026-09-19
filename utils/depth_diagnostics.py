"""NumPy-only, root-relative depth diagnostics in millimetres.

Errors (not predictions) are averaged for repeated frame occurrences. Temporal
windows must have consecutive original indices, identical source and action.
"""
import csv
import json
from pathlib import Path

import numpy as np


BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
         (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
         (12, 13), (8, 14), (14, 15), (15, 16)]
PAIRS = [(i, j) for i in range(17) for j in range(i + 1, 17)]
BLOCK_LIST = {'s_09_act_05_subact_02', 's_09_act_10_subact_02',
              's_09_act_13_subact_01'}
METRICS = ['mpjpe_mm', 'x_mae_mm', 'y_mae_mm', 'z_mae_mm',
           'xy_error_mm', 'x_mse_mm2', 'y_mse_mm2', 'z_mse_mm2']


def temporal_mask(ids, sources, actions, order):
    n = len(ids) - order
    if n <= 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(max(n, 0), dtype=bool)
    for k in range(1, order + 1):
        mask &= ids[k:k+n] == ids[:n] + k
        mask &= sources[k:k+n] == sources[:n]
        mask &= actions[k:k+n] == actions[:n]
    return mask


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class DepthDiagnostics:
    def __init__(self, frame_clips, actions, sources, joint_labels, threshold_mm=10.):
        if not np.isfinite(threshold_mm) or threshold_mm < 0:
            raise ValueError('Ordering threshold must be finite and nonnegative')
        self.frames = np.asarray(frame_clips, dtype=np.int64)
        self.actions = np.asarray(actions).astype(str)
        self.sources = np.asarray(sources).astype(str)
        if self.frames.ndim != 2 or self.frames.size == 0:
            raise ValueError('Expected nonempty [clips, frames] indices')
        if self.frames.min() < 0 or self.frames.max() >= len(self.actions):
            raise ValueError('Frame indices out of range')
        if len(self.sources) != len(self.actions):
            raise ValueError('Source/action lengths differ')
        self.labels = joint_labels
        self.threshold = float(threshold_mm)
        self.counts = [np.zeros(len(self.actions), dtype=np.int64) for _ in range(3)]
        self.keep = []
        for ids in self.frames:
            # Exactly the sequence exclusion used by train.evaluate.
            keep = self.sources[ids[0]][:-6] not in BLOCK_LIST
            self.keep.append(keep)
            if not keep:
                continue
            np.add.at(self.counts[0], ids, 1)
            for order in (1, 2):
                mask = temporal_mask(ids, self.sources[ids], self.actions[ids], order)
                np.add.at(self.counts[order], ids[:-order][mask], 1)
        self.stats = {}
        self.seen = set()

    def _stats(self, action):
        if action not in self.stats:
            self.stats[action] = dict(n=0., pose=np.zeros(len(METRICS)),
                joint=np.zeros(17), bone=np.zeros(16), p_sum=0., p_n=0.,
                correct=np.zeros(len(PAIRS)), valid=np.zeros(len(PAIRS)),
                temporal_sum=np.zeros(3), temporal_n=np.zeros(3))
        return self.stats[action]

    def update(self, clip_index, pred, gt, p_errors=None):
        if clip_index in self.seen:
            raise ValueError('Clip processed twice')
        self.seen.add(clip_index)
        if not self.keep[clip_index]:
            return
        ids = self.frames[clip_index]
        pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
        if pred.shape != (len(ids), 17, 3) or gt.shape != pred.shape:
            raise ValueError('Expected matching [T,17,3] arrays')
        if not np.isfinite(pred).all() or not np.isfinite(gt).all():
            raise ValueError('Nonfinite pose encountered')
        pred = pred - pred[:, :1]
        gt = gt - gt[:, :1]
        diff = pred - gt
        mae = np.abs(diff).mean(axis=1)
        mse = np.square(diff).mean(axis=1)
        pose = np.column_stack((np.linalg.norm(diff, axis=-1).mean(axis=1),
                                mae, np.linalg.norm(diff[..., :2], axis=-1).mean(axis=1), mse))
        bone = np.stack([np.abs(diff[:, j, 2] - diff[:, i, 2]) for i, j in BONES], axis=1)
        pair_gt = np.stack([gt[:, j, 2] - gt[:, i, 2] for i, j in PAIRS], axis=1)
        pair_pred = np.stack([pred[:, j, 2] - pred[:, i, 2] for i, j in PAIRS], axis=1)
        valid = np.abs(pair_gt) > self.threshold
        correct = valid & (np.sign(pair_gt) == np.sign(pair_pred))
        if p_errors is not None:
            p_errors = np.asarray(p_errors)
            if p_errors.shape != (len(ids),) or not np.isfinite(p_errors).all():
                raise ValueError('Invalid P-MPJPE results')
        actions = self.actions[ids]
        weights = 1. / self.counts[0][ids]
        for action in np.unique(actions):
            select = actions == action
            w = weights[select]
            s = self._stats(action)
            s['n'] += w.sum()
            s['pose'] += (pose[select] * w[:, None]).sum(axis=0)
            s['joint'] += (np.abs(diff[select, :, 2]) * w[:, None]).sum(axis=0)
            s['bone'] += (bone[select] * w[:, None]).sum(axis=0)
            s['valid'] += (valid[select] * w[:, None]).sum(axis=0)
            s['correct'] += (correct[select] * w[:, None]).sum(axis=0)
            if p_errors is not None:
                s['p_sum'] += (p_errors[select] * w).sum()
                s['p_n'] += w.sum()
        for order in (1, 2):
            mask = temporal_mask(ids, self.sources[ids], actions, order)
            starts = ids[:-order][mask]
            if not len(starts):
                continue
            delta = np.diff(diff, n=order, axis=0)[mask]
            errors = np.abs(delta[..., 2]).mean(axis=1)
            tw = 1. / self.counts[order][starts]
            for action in np.unique(self.actions[starts]):
                select = self.actions[starts] == action
                s = self._stats(action)
                slot = order - 1
                s['temporal_sum'][slot] += (errors[select] * tw[select]).sum()
                s['temporal_n'][slot] += tw[select].sum()
                if order == 2:
                    acc = np.linalg.norm(delta[select], axis=-1).mean(axis=1)
                    s['temporal_sum'][2] += (acc * tw[select]).sum()
                    s['temporal_n'][2] += tw[select].sum()

    @staticmethod
    def _row(s):
        row = {name: float(value / s['n']) for name, value in zip(METRICS, s['pose'])}
        row['p_mpjpe_mm'] = safe_ratio(s['p_sum'], s['p_n'])
        row['z_squared_error_fraction'] = safe_ratio(s['pose'][-1], s['pose'][-3:].sum())
        row['bone_delta_z_mae_mm'] = float(s['bone'].mean() / s['n'])
        row['ordering_accuracy'] = safe_ratio(s['correct'].sum(), s['valid'].sum())
        row['ordering_coverage'] = safe_ratio(s['valid'].sum(), s['n'] * len(PAIRS))
        for i, name in enumerate(['z_velocity_error_mm_per_frame',
                                  'z_acceleration_error_mm_per_frame2',
                                  'acceleration_error_mm_per_frame2']):
            row[name] = safe_ratio(s['temporal_sum'][i], s['temporal_n'][i])
        return row

    def save(self, output_dir, metadata=None):
        if len(self.seen) != len(self.frames):
            raise ValueError('Cannot summarize an incomplete evaluation')
        if not self.stats:
            raise ValueError('No eligible frames')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        action_rows = []
        for action, s in sorted(self.stats.items()):
            action_rows.append(dict(action=action, frames=int(round(s['n'])),
                                    velocity_windows=int(round(s['temporal_n'][0])),
                                    acceleration_windows=int(round(s['temporal_n'][1])), **self._row(s)))
        total = {k: sum(s[k] for s in self.stats.values()) for k in next(iter(self.stats.values()))}
        micro = self._row(total)
        macro = {}
        for key in micro:
            values = [row[key] for row in action_rows if row[key] is not None]
            macro[key] = float(np.mean(values)) if values else None
        summary = dict(metadata=metadata or {}, coordinate_frame='root-relative; evaluation units mm',
                       joints_include_root=True, ordering_threshold_mm=self.threshold,
                       ordering_pairs='all 136 unordered joint pairs; strict GT gap > threshold',
                       frames=int(round(total['n'])), skipped_clips=self.keep.count(False),
                       frame_weighted=micro, action_macro=macro,
                       missing_actions=sorted(set(self.actions) - set(self.stats)),
                       temporal_note='Within-clip consecutive source frames only; no clip-boundary derivatives. '
                                     'Own valid-window counts; differs from legacy train.py acceleration aggregation.')
        (output_dir / 'depth_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        self._csv(output_dir / 'action_z_error.csv', action_rows)
        self._csv(output_dir / 'joint_z_error.csv', [dict(joint_id=j, joint=self.labels[j],
            frame_weighted_z_mae_mm=float(total['joint'][j] / total['n']),
            action_macro_z_mae_mm=float(np.mean([s['joint'][j] / s['n'] for s in self.stats.values()])))
            for j in range(17)])
        self._csv(output_dir / 'bone_delta_z_error.csv', [dict(parent=i, child=j,
            bone=f'{self.labels[i]} -> {self.labels[j]}',
            frame_weighted_delta_z_mae_mm=float(total['bone'][k] / total['n']),
            action_macro_delta_z_mae_mm=float(np.mean([s['bone'][k] / s['n'] for s in self.stats.values()])))
            for k, (i, j) in enumerate(BONES)])
        self._csv(output_dir / 'depth_ordering.csv', [dict(joint_i=i, joint_j=j,
            label_i=self.labels[i], label_j=self.labels[j], is_bone=(i, j) in BONES,
            effective_valid_frames=float(total['valid'][k]),
            coverage=float(total['valid'][k] / total['n']),
            accuracy=safe_ratio(total['correct'][k], total['valid'][k])) for k, (i, j) in enumerate(PAIRS)])
        return summary

    @staticmethod
    def _csv(path, rows):
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
