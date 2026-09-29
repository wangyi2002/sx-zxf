# Experiment: late XY/Z feature specialization

Parent: `depth-diagnosis` at `e1d208b9aeb02ff1be6edd15445e8356de7a420a`.
Branch: `exp/xy-z-late-split`.

## Hypothesis and implementation

Test whether independent late spatial/temporal processing for Z helps depth
estimation. This is a mechanism validation inspired by PoseMoE, not a reproduction
or a claim of architectural novelty. It is not SRDA or a 2D coordinate corrector.

The embedding and first 28 of 30 blocks remain shared. Original blocks 29-30
(both spatial + temporal attention blocks) form the XY suffix. Independent copies
form the Z suffix. Both receive `[B,T,17,128]`. Each has its own LayerNorm and
128->512 + Tanh representation projection. Heads output `[B,T,17,2]` and
`[B,T,17,1]`; concatenation returns `[B,T,17,3]`. Z is a direct prediction, not a
bone residual. No skeleton changes, router, cross-attention, detach or freezing.
The shared prefix receives gradients from both tasks. The original joint losses
still couple coordinate optimization; independent paths do not imply independent
losses or complete statistical disentanglement.

Initialize the full baseline first, then deepcopy suffix/norm/representation
without consuming random numbers. Partition the initialized xyz output rows into
XY and Z heads. Equal initial values do not share parameter storage. This is
random initialization from scratch, NOT loading trained baseline weights.

Config: `configs/h36m/MotionAGFormer-xy-z-split.yaml`.
`use_xy_z_split: true`, `xy_z_split_blocks: 2`.
Setting the toggle false restores baseline model state keys and computation.
In split mode `return_rep=True` returns `(xy_rep, z_rep)`, each `[B,T,17,512]`.
In baseline mode it retains the original single-tensor return.

## Budget and metrics

Unchanged: 60 epochs, batch 4, seed 0, T=243, width 128, AdamW lr=0.0005,
weight decay=0.01, lr decay=0.99, gradient clipping=1, detector XY/confidence,
augmentation, root-relative normalized training coordinates, original loss
(pose + 0.5 scale + 20 velocity; other existing weights zero), evaluation and
best-checkpoint selection. No relative-depth auxiliary loss.
Existing WandB metrics remain; the full config records the two new flags.
Training prints baseline and added parameters. Diagnosis metrics are unchanged;
its parameter metadata now reports the actual added count instead of zero.

Baseline parameters: 11,342,421. Split total: 12,201,301.
Added: 858,880 (7.5723%). Output-head row partition adds no parameters;
the increment is two blocks plus independent norm/representation projection.
This is not a parameter-matched comparison. If promising, compare against a
similar-cost shared model before attributing gains to specialization.

Use action-macro MPJPE and Z MAE as primary outcomes, X/Y as tradeoff checks;
also retain P-MPJPE, Bone delta-Z, ordering, acceleration, joint/action reports.
Reference saved baseline diagnostic MPJPE=38.5080, Z MAE=30.1927,
Bone delta-Z=24.3518 mm. Do not mix with frame-weighted MPJPE=39.0648.
Single-seed small gains require replication; no automatic layer/width sweep.

## Checks and limitations

CPU PyTorch 2.5.1, explicit repository scan reference (not the CUDA kernel):
- Full `[1,243,17,3]` forward finite, expected output shape.
- Split initialization max difference from baseline: 2.24e-7.
- Initialization RNG unchanged and copied suffix storage independent.
- Toggle off vs exact parent source: state keys/values, RNG, full forward exact.
- Short T=9 backward: XY/Z suffix gradient isolation and shared-prefix gradient.
- Active original losses, clipping and AdamW step checked; Z head updates.
- Strict split state roundtrip, baseline checkpoint rejected in split mode.
- Original GCN's four unused BatchNorm parameters remain untouched.

No real dataset training or GPU VRAM/speed measurement was available here.
The included synthetic CUDA smoke command uses the production kernel and reports
baseline/split peak allocated VRAM and step time (warm-up + 3 measured steps).
It does not load training data or overwrite checkpoints. This is a short capacity
check, not a full-epoch throughput benchmark. Do not start full training if it
OOMs or shows an unexplained large slowdown.

## Local commands

```bash
git fetch origin
git switch --track origin/exp/xy-z-late-split

# Short production-kernel check before the full run:
python tools/smoke_xy_z_split.py --device cuda --batch-size 4 --frames 243

# Fresh run: do NOT pass --checkpoint or --resume or load baseline_best.
python train.py --config configs/h36m/MotionAGFormer-xy-z-split.yaml \
  --new-checkpoint checkpoint --seed 0 \
  --use-wandb --wandb-name xy-z-late-split-seed0

python diagnose_depth.py --config configs/h36m/MotionAGFormer-xy-z-split.yaml \
  --checkpoint checkpoint/best_epoch.pth.tr --input-source metadata \
  --batch-size 1 --num-workers 0 --seed 0 \
  --output-dir depth_outputs/xy_z_late_split
```

The existing best/latest output files in `checkpoint/` will be overwritten during
training, as agreed. Keep `checkpoint/baseline_best.pth.tr` as the separate backup.
Split checkpoints require the split config; baseline checkpoints require baseline
architecture. No automatic partial loading is introduced. Training is not launched
by this change. Resume only with a checkpoint from this same architecture;
the baseline resume/RNG behavior is unchanged.
