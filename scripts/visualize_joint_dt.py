"""Dataset-level per-joint temporal delta and 2D motion; no flip augmentation.
Run from repo root. Produces per-layer CSV/PNG plus an equal-layer mean plot.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.tools import get_config, set_random_seed
from utils.learning import load_model

NAMES = ['Pelvis','R hip','R knee','R ankle','L hip','L knee','L ankle',
         'Spine','Thorax','Neck','Head','L shoulder','L elbow','L wrist',
         'R shoulder','R elbow','R wrist']

class JointDeltaCollector:
    def __init__(self):
        self.sums, self.counts = {}, {}

    def observer(self, name):
        @torch.no_grad()
        def collect(logits, bias, batch, joints):
            dt = F.softplus(logits.detach().float() + bias.detach().float()[None, :, None])
            # [B*J,C,T] -> [B,J,C,T]; exclude first frame (zero-padded dt context).
            dt = dt.reshape(batch, joints, dt.shape[1], dt.shape[2])[..., 1:]
            values = dt.sum((0, 2, 3)).double().cpu()
            self.sums[name] = self.sums.get(name, torch.zeros(joints, dtype=torch.float64)) + values
            self.counts[name] = self.counts.get(name, 0) + batch * dt.shape[2] * dt.shape[3]
        return collect

    def means(self):
        return {k: (v / self.counts[k]).numpy() for k, v in self.sums.items()}


def save_results(directory, delta_means, motion, clips):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    curves = dict(delta_means)
    curves['all_layers_mean'] = np.mean(list(delta_means.values()), axis=0)
    order = np.argsort(-motion, kind='stable')  # same motion ordering for EVERY layer
    with (directory / 'joint_statistics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer','joint','label','mean_delta','motion_2d_normalized_units_per_frame','clips'])
        for layer, dt in curves.items():
            for j in range(17):
                writer.writerow([layer,j,NAMES[j],float(dt[j]),float(motion[j]),clips])
    for layer, dt in curves.items():
        fig, ax = plt.subplots(figsize=(11, 4.5))
        right = ax.twinx()
        a = ax.plot(range(17), dt[order], 's-', color='coral', label='Mean delta')[0]
        b = right.plot(range(17), motion[order], 'o-', color='skyblue', label='2D motion intensity')[0]
        ax.set_xticks(range(17), [str(j) for j in order])
        ax.set_xlabel('Joint index (sorted by descending 2D motion)')
        ax.set_ylabel('Mean delta (learned SSM scale)', color='coral')
        right.set_ylabel('2D displacement / frame (normalized input units)', color='steelblue')
        ax.set_ylim(bottom=0)
        right.set_ylim(bottom=0)
        ax.set_title(layer)
        ax.legend(handles=[a,b], loc='upper right')
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(directory / (layer.replace('.', '_') + '.png'), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True, help='Full path to checkpoint file')
    p.add_argument('--output', default='outputs/joint_dt')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-batches', type=int, default=0, help='0=entire test set; >0=rough subset')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    opt = p.parse_args()
    if not Path(opt.checkpoint).is_file():
        raise FileNotFoundError(opt.checkpoint)
    set_random_seed(0)
    args = get_config(opt.config)
    if args.n_frames < 2:
        raise ValueError('Need at least two frames')
    model = load_model(args).to(opt.device)
    ckpt = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('module.'): v for k,v in ckpt['model'].items()}
    model.load_state_dict(weights, strict=True)
    model.eval()
    collector = JointDeltaCollector()
    for name, module in model.named_modules():
        if getattr(module, 'mode', None) == 'temporal' and hasattr(module, 'dt_rank'):
            module.joint_dt_observer = collector.observer(name)
    from data.reader.motion_dataset import MotionDataset3D
    loader = DataLoader(MotionDataset3D(args, args.subset_list, 'test'),
                        batch_size=opt.batch_size, shuffle=False, num_workers=0)
    motion_sum = torch.zeros(17, dtype=torch.float64)
    count = clips = 0
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            if opt.max_batches and idx >= opt.max_batches:
                break
            motion = torch.linalg.vector_norm(x[:, 1:, :, :2] - x[:, :-1, :, :2], dim=-1)
            motion_sum += motion.double().sum((0,1))
            count += motion.shape[0] * motion.shape[1]
            clips += x.shape[0]
            model(x.to(opt.device))  # unflipped only; joint identity stays consistent
            if (idx + 1) % 10 == 0:
                print(f'Processed {idx+1} batches / {clips} clips', flush=True)
    if not count or not collector.sums:
        raise RuntimeError('No data or temporal delta observations')
    save_results(opt.output, collector.means(), (motion_sum/count).numpy(), clips)
    metadata = dict(vars(opt), clips=clips, config_values=dict(args),
                    checkpoint_epoch=ckpt.get('epoch'), frames_exclude_first=True, flip=False)
    (Path(opt.output) / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    print(f'Saved PNG/CSV to {opt.output}; {clips} clips, no flip, t=1..T-1')

if __name__ == '__main__':
    main()
