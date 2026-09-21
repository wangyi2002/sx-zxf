"""Evaluate an existing H36M checkpoint without changing train.py or the model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.const import H36M_JOINT_TO_LABEL
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from loss.pose3d import p_mpjpe
from utils.data import flip_data
from utils.depth_diagnostics import DepthDiagnostics
from utils.tools import get_config, set_random_seed




def camera_resolution(camera):
    if camera in ('54138969', '60457274'):
        return 1000, 1002
    if camera in ('55011271', '58860488'):
        return 1000, 1000
    raise ValueError(f'Invalid H36M camera: {camera}')


def normalize_pose_frames(coords, cameras):
    """Normalize each frame using its own camera, as in read_2d/read_3d."""
    result = np.asarray(coords, dtype=np.float32).copy()
    if len(result) != len(cameras):
        raise ValueError('Pose/camera frame counts differ')
    for t, camera in enumerate(cameras):
        width, height = camera_resolution(camera)
        result[t, :, :2] = result[t, :, :2] / width * 2 - [1, height / width]
        if result.shape[-1] == 3:
            result[t, :, 2:] = result[t, :, 2:] / width * 2
    return result


def denormalize_pose_frames(coords, cameras):
    """Invert normalization per frame; never reuse the first frame's height."""
    result = np.asarray(coords, dtype=np.float32).copy()
    if len(result) != len(cameras):
        raise ValueError('Pose/camera frame counts differ')
    for t, camera in enumerate(cameras):
        width, height = camera_resolution(camera)
        result[t, :, :2] = (result[t, :, :2] + [1, height / width]) * width / 2
        result[t, :, 2:] = result[t, :, 2:] * width / 2
    return result


class MetadataTestClips(Dataset):
    """Use one index map for detector inputs, labels and evaluation metadata.

    Legacy preprocessed pickles contain no frame IDs; random resampling of short
    sequences cannot be reliably reconstructed from a seed chosen afterwards.
    """
    def __init__(self, test, frame_clips):
        self.test = test
        self.frame_clips = frame_clips

    def __len__(self):
        return len(self.frame_clips)

    def __getitem__(self, index):
        ids = self.frame_clips[index]
        inputs = np.asarray(self.test['joint_2d'])[ids, :, :2].astype(np.float32)
        labels = np.asarray(self.test['joint3d_image'])[ids, :, :3].astype(np.float32)
        cameras = np.asarray(self.test['camera_name'])[ids]
        inputs = normalize_pose_frames(inputs, cameras)
        labels = normalize_pose_frames(labels, cameras)
        if 'confidence' in self.test:
            confidence = np.asarray(self.test['confidence'])[ids].astype(np.float32)
            if confidence.ndim == 2:
                confidence = confidence[..., None]
        else:
            confidence = np.ones(inputs.shape[:-1] + (1,), dtype=np.float32)
        inputs = np.concatenate((inputs, confidence), axis=-1)
        return torch.from_numpy(inputs), torch.from_numpy(labels)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/h36m/MotionAGFormer-large.yaml')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file (not directory)')
    parser.add_argument('--data-root', help='Override sliced data and metadata root')
    parser.add_argument('--output-dir', default='depth_outputs/baseline')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch only')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--ordering-threshold-mm', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input-source', choices=['metadata', 'slices'], default='metadata',
                        help='metadata: aligned detector input/GT from raw pkl; slices: strict legacy slice validation')
    opts = parser.parse_args()
    if opts.batch_size < 1 or opts.num_workers < 0:
        parser.error('batch-size must be positive; num-workers must be nonnegative')
    return opts


def main():
    opts = parse_args()
    config = get_config(opts.config)
    if config.num_joints != 17 or not config.root_rel or config.add_velocity:
        raise ValueError('First version supports H36M 17 joints, root_rel=True, add_velocity=False')
    if config.use_proj_as_2d:
        raise ValueError('Use detector 2D input for baseline diagnosis, not projected GT input')
    checkpoint_path = Path(opts.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError('Checkpoint diagnosis uses the original CUDA Mamba kernel. '
                           'For CPU shape/parameter validation run tools/smoke_depth.py.')
    from utils.learning import load_model
    if opts.data_root:
        config.data_root = opts.data_root
    set_random_seed(opts.seed)
    # Match the legacy dataset's NumPy seed=0 before creating frame indices.
    # Metadata mode never mixes saved random slices with regenerated indices.
    np.random.seed(0)
    reader = DataReaderH36M(config.n_frames, sample_stride=1,
                           data_stride_train=config.n_frames // 3,
                           data_stride_test=config.n_frames,
                           dt_root=config.data_root, dt_file=config.dt_file)
    _, frame_clips = reader.get_split_id()
    frame_clips = np.asarray(frame_clips)
    test = reader.dt_dataset['test']
    if opts.input_source == 'metadata':
        dataset = MetadataTestClips(test, frame_clips)
        print('[INFO] Input source: metadata; detector inputs and GT share exact frame indices.')
    else:
        dataset = MotionDataset3D(config, config.subset_list, 'test')
        if len(dataset) != len(frame_clips):
            raise ValueError(f'Sliced files ({len(dataset)}) and metadata clips ({len(frame_clips)}) differ')
    diagnostics = DepthDiagnostics(frame_clips, test['action'], test['source'],
                                   H36M_JOINT_TO_LABEL, opts.ordering_threshold_mm)
    model = load_model(config)
    # Keep CUDA training's selective_scan_fn and all model state keys unchanged.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: total={total_params:,}, trainable={trainable_params:,}; added=0')
    del state, checkpoint
    model = model.cuda().eval()
    loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False,
                        num_workers=opts.num_workers, pin_memory=True)
    cursor = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(loader, desc='Depth diagnosis'):
            inputs = inputs.cuda(non_blocking=True)
            predicted = model(inputs)
            if config.flip:
                predicted = (predicted + flip_data(model(flip_data(inputs)))) / 2
            predicted[:, :, 0, :] = 0
            predicted = predicted.cpu().numpy()
            for b, pred in enumerate(predicted):
                idx = cursor + b
                ids = frame_clips[idx]
                cameras = np.asarray(test['camera_name'])[ids]
                # Per-frame camera geometry must match the input/label reader.
                expected_label = normalize_pose_frames(
                    np.asarray(test['joint3d_image'])[ids, :, :3], cameras)
                if not np.allclose(labels[b].numpy(), expected_label, atol=1e-5, rtol=1e-5):
                    raise ValueError(f'Clip {idx} labels do not match metadata frame indices. '
                                     f'source={test["source"][ids[0]]}, '
                                     f'max_abs_label_diff={np.max(np.abs(labels[b].numpy() - expected_label)):.6g}. '
                                     f'input_source={opts.input_source}, cameras={np.unique(cameras).tolist()}. '
                                     'Check slice alignment and camera normalization; do not disable validation.')
                pred = denormalize_pose_frames(pred, cameras)
                pred *= np.asarray(test['2.5d_factor'])[ids, None, None]
                gt = np.asarray(test['joints_2.5d_image'])[ids].copy()
                pred = pred - pred[:, :1]
                gt = gt - gt[:, :1]
                aligned_errors = p_mpjpe(pred, gt) if diagnostics.keep[idx] else None
                diagnostics.update(idx, pred, gt, aligned_errors)
            cursor += len(predicted)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    summary = diagnostics.save(opts.output_dir, dict(config=dict(config),
        checkpoint=str(checkpoint_path), checkpoint_sha256=digest.hexdigest(),
        git_revision=revision, total_parameters=total_params, trainable_parameters=trainable_params,
        added_parameters=0, seed=opts.seed, split_numpy_seed=0, input_source=opts.input_source,
        inference_batch_size=opts.batch_size, camera_normalization='per-frame',
        joint_labels_source='user supplied data/const.py; interpret IDs as authoritative'))
    print(json.dumps(summary['action_macro'], indent=2))
    print(f'Reports: {opts.output_dir}')


if __name__ == '__main__':
    main()
