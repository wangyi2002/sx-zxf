# Frozen-baseline SRDA
Source: exp/skeleton-relative-depth-adapter @ 0ff5fb47b2e392259d35d2e4223f8d705d8d872e.
Branch: exp/frozen-baseline-srda.

Load a clean baseline checkpoint, never a trained adapter checkpoint.
Baseline parameters and buffers stay fixed and baseline modules stay in eval mode.
Only a fresh zero-initialized SRDA (4,161 parameters) is optimized.
Original losses/settings retained: batch 4, seed 0, 60 ADDITIONAL epochs.
This is a staged mechanism test, not an equal-budget from-scratch comparison.

Use --baseline-checkpoint checkpoint/baseline_best.pth.tr.
Source must differ from output best_epoch.pth.tr/latest_epoch.pth.tr,
including aliases. Baseline key and shape checking is strict.
A real batch checks exact initial full/raw-head equality before evaluation.
checkpoint/frozen_baseline_reference.json records the input SHA256 and
initial evaluation. Each epoch checks frozen parameters AND buffers by hash.
Best checkpoint means best trained epoch, even when worse than baseline.
The original evaluator and diagnostic metric definitions are unchanged.
Use metadata diagnosis on both the backup baseline (base config) and the
trained model (frozen config) for matched depth comparisons.

Resume is rejected for this first fixed-budget experiment. Diagnostic
inference uses the frozen config and trained checkpoint normally.
Freezing adds no model state keys. No complete/GPU training was run remotely.

Training:
python train.py --config configs/h36m/MotionAGFormer-frozen-srda.yaml --baseline-checkpoint checkpoint/baseline_best.pth.tr --new-checkpoint checkpoint --seed 0 --use-wandb --wandb-name frozen-srda-seed0

Diagnosis:
python diagnose_depth.py --config configs/h36m/MotionAGFormer-frozen-srda.yaml --checkpoint checkpoint/best_epoch.pth.tr --input-source metadata --batch-size 1 --num-workers 0 --seed 0 --output-dir depth_outputs/frozen_srda_seed0
