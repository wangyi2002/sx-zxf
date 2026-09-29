# GT image-XY input control (inference only)
Parent: depth-diagnosis @ e1d208b9aeb02ff1be6edd15445e8356de7a420a.
Use configs/h36m/MotionAGFormer-base.yaml and the ORIGINAL baseline checkpoint.
Do not use relative-depth-loss, SRDA, joint-adapter or frozen-SRDA weights.
Adapter checkpoint keys are explicitly rejected; all model keys/shapes load strictly.
Loss-only experiment checkpoints have baseline-compatible keys, so architecture
checks cannot identify them. Use the known baseline backup, optionally verify
against its detector summary via --reference-summary.

New flag: --input-xy gt-image (default detector).
Only input XY changes: raw metadata joint3d_image[..., :2], before root
centering, normalized using each frame's camera width/height.
Detector confidence, frame indices, flip averaging, targets, denormalization,
block list and DepthDiagnostics metrics are unchanged. No GT Z is fed to model.
The project already uses this image-label XY convention in read_3d and
MotionDataset3D._construct_motion2d_by_projection. This is NOT taking camera
coordinate XY or implementing a new camera projection.
Code inspection establishes the project's coordinate convention, not the
independent physical provenance of a user's preprocessed pickle.
Runtime guards check alignment shape, camera count, finite values, confidence.
They cannot prove annotation accuracy or detect a jointly mislabelled dataset.

Use a separate output directory:
python diagnose_depth.py --config configs/h36m/MotionAGFormer-base.yaml --checkpoint checkpoint/baseline_best.pth.tr --input-source metadata --input-xy gt-image --batch-size 1 --num-workers 0 --seed 0 --output-dir depth_outputs/baseline_gt2d

If the original detector report is available, append:
--reference-summary depth_outputs/baseline/depth_summary.json
This verifies checkpoint SHA256, config, seed, batch size and normalization.
Files with differing settings are rejected rather than silently compared.
No weights are written. No training/architecture change.
Reuse the detector result only with identical weights/data/evaluation settings.

Interpret action_macro Z MAE, Bone delta-Z and MPJPE alongside X/Y.
This is oracle-input sensitivity, NOT an attainable upper bound or an additive
decomposition of detector/lifter error. Detector-trained model sees shifted
inputs; retained confidence is a controlled but potentially inconsistent cue.
GT XY can directly improve XY-related metrics, so overall MPJPE alone does not
establish improvement in depth inference.
