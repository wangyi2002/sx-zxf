# Experiment 2: Skeleton-Relative Depth Adapter

Source: `depth-diagnosis@e1d208b9aeb02ff1be6edd15445e8356de7a420a`.
Branch: `exp/skeleton-relative-depth-adapter`, created after user confirmation.
No Exp1 auxiliary loss is included. Full training must be started only after review.

## Hypothesis and implementation

Test whether a shared, lightweight MLP on child-minus-parent backbone features
can improve metric relative depth via bone-wise Z corrections. This experiment
does not establish novelty or isolate the benefit from extra capacity; any gain
would motivate a later matched-capacity control.

The production backbone ends with `self.norm`, giving **F=[B,T,17,128]**.
SRDA branches here, before the unchanged `rep_logit:128->512,Tanh` and original
`head:512->3`. It does not use the 512-dimensional head representation.

`edge_feature = F_child - F_parent` has shape `[B,T,16,128]`.
All edges share one `Linear(128,32) -> GELU -> Linear(32,1)` in
`model/modules/srda.py::SkeletonRelativeDepthAdapter`.
Its last weight and bias are zero-initialized. The result, delta, is `[B,T,16]`.

Root is **joint 0**. Directed edges, in the same order as the original depth diagnostic:

```text
0->1, 1->2, 2->3, 0->4, 4->5, 5->6,
0->7, 7->8, 8->9, 9->10, 8->11, 11->12,
12->13, 8->14, 14->15, 15->16
```

Parent list by joint index (root parent is -1):
`[-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]`.
The original diagnostic constant was moved, unchanged, to
`data/const.py::H36M_BONES`; each edge has a joint-label comment.
The diagnostic imports this constant. Labels follow this repository; IDs are authoritative.

Initialization constructs a fixed path matrix A=[17,16], where A[j,e]=1 if edge e
is on the root-to-joint path. Parent/child indices and A are registered buffers,
not trainable parameters, and follow model device transfers.
Only tree construction uses joint/edge Python loops, once at initialization.
Forward uses indexed feature differences and `r = delta @ A.T` -> `[B,T,17]`.
For every edge, `r_child-r_parent=delta_edge`. Root residual is always zero.

Final output is `[x_base, y_base, z_base+r]`, shape `[B,T,17,3]`.
XY are copied from the original pose head. Because F is not detached, training
can still change shared backbone weights and therefore indirectly change XY.
Corrections use the same normalized coordinates as the original pose loss;
there is no mm conversion in the adapter.

## Configuration and training conditions

`configs/h36m/MotionAGFormer-srda.yaml` copies the confirmed base config with only:

```yaml
use_skeleton_relative_depth_adapter: true
depth_adapter_ratio: 0.25
use_relative_depth_loss: false
```

Old configs default to SRDA off. Off constructs no adapter parameters or buffers
and executes the original pose path. `return_rep=True` still returns the original
512-dimensional representation. Added initialization preserves the CPU RNG stream
and occurs after all baseline parameters have been initialized.

Loss module and original total loss expression are unchanged. Original velocity
loss (20), scale loss (0.5) and global gradient clipping (1.0) remain. No new loss.
Enabling `use_relative_depth_loss` raises an error in this experiment branch.
Other settings: batch=4, epochs=60, T=243, AdamW, lr=0.0005, weight_decay=0.01,
lr_decay=0.99, flip augmentation; training command uses seed=0.
Checkpoint selection and evaluation remain baseline MPJPE-based.
The pre-existing resume behavior is not modified in this experiment.

Normal inference returns only the pose tensor. Training requests a small detached
sum/count tensor to aggregate logging correctly across DataParallel replicas:

- `train/depth_adapter_bone_residual_abs_mean`
- `train/depth_adapter_joint_residual_abs_mean`

The two means are also printed each epoch, in normalized training units.
No feature tensors are cached on the module; no detached F enters the adapter.

## Parameters and validation

| Item | Parameters |
|---|---:|
| Baseline | 11,342,421 |
| SRDA / added | 4,161 |
| Total | 11,346,582 |

Formula: `128*32+32 + 32*1+1 = 4161`, approximately 0.037% overhead.

- 12 CPU tests: original 7 diagnostic regression tests plus 5 SRDA tests.
- Module check uses `[4,243,17,128]`; edge and residual shapes verified.
- Full production model CPU reference forward uses `[1,243,17,3]`; all shapes
  and finite outputs verified. Zero-init output equals baseline; residual means 0.
- At initialization the last Linear receives gradients; earlier adapter layers
  and its input have zero branch gradients as expected. After the last layer is
  updated, both Linear layers and F have finite nonzero gradients.
- Small production backbone backward (1 layer, T=9, C=16) isolates the SRDA
  gradient by subtracting the LIVE raw-head Z from final Z; nonzero gradient
  reaches `joints_embed` after an adapter update. This is a test-only loss.
- Off/zero initialization and `return_rep` compatibility checked against original
  backbone source. Same-forward zero correction is exact; independent CPU
  executions in the small integration test allow floating-point roundoff.
- Root residual and unchanged same-forward XY are verified even with learned
  nonzero adapter weights. Path differences recover the supplied bone residuals.
- Diagnostic formulas, train evaluation and checkpoint routines are unchanged;
  `diagnose_depth.py` only corrects added-parameter metadata from hardcoded 0.

CPU reference forward timing: batch=1,T=243, 2 timed iterations after warm-up:

| Variant | Seconds/iteration |
|---|---:|
| Baseline | 1.13124 |
| SRDA | 1.16138 |

Approximately +2.7% in this short CPU measurement; not a GPU speed prediction.
No CUDA GPU, H36M dataset or user checkpoint was available in the implementation
environment. Baseline/SRDA VRAM and GPU training iteration/epoch times are **not
measured**. CPU short tests do not substitute for a local GPU preflight.

## Pull and local preflight

```bash
git fetch origin
git switch --track origin/exp/skeleton-relative-depth-adapter
python -m unittest discover -s tests -v
python tools/smoke_srda.py --device cuda --batch-size 4 --steps 5
```

If the branch already exists locally: `git switch exp/skeleton-relative-depth-adapter`
then `git pull --ff-only` instead of `git switch --track`.

The GPU smoke tool performs synthetic short training with the actual production
epoch body and original losses, comparing off/on. It reports peak allocated and
reserved VRAM, seconds/iteration, parameters and shapes into
`depth_outputs/srda_smoke.json`. It does not read data, save checkpoints or start
wandb/full training. An OOM or abnormal slowdown should be investigated first.
The output directory is already ignored by the baseline rules.

## Proposed full training command (after user review)

Reuse `checkpoint/` as requested; best/latest files may overwrite existing files.
Train from scratch, without `--checkpoint` or `--resume` for this comparison.

```bash
python train.py \
  --config configs/h36m/MotionAGFormer-srda.yaml \
  --new-checkpoint checkpoint \
  --seed 0 \
  --use-wandb \
  --wandb-name srda-r025-seed0
```

After training, diagnose the best checkpoint:

```bash
python diagnose_depth.py \
  --config configs/h36m/MotionAGFormer-srda.yaml \
  --checkpoint checkpoint/best_epoch.pth.tr \
  --input-source metadata \
  --batch-size 1 \
  --num-workers 0 \
  --seed 0 \
  --output-dir depth_outputs/srda_seed0
```

Keep the original five report formats. Compare like-for-like aggregates:

| Metric | Baseline frame_weighted | Baseline action_macro |
|---|---:|---:|
| MPJPE mm | 39.06484 | 38.50804 |
| P-MPJPE mm | 33.10445 | 32.63729 |
| Z MAE mm | 30.61430 | 30.19269 |
| Bone delta-Z MAE mm | 24.55437 | 24.35180 |
| Ordering % | 95.25747 | 95.34569 |

Also compare X/Y, squared-error share, joint/action errors and acceleration.
Do not mix frame-weighted diagnostics with action-macro training MPJPE.
First judge Bone delta-Z, Z MAE and MPJPE jointly. If they worsen, stop and inspect
residuals/chain accumulation before proposing further changes. No automatic
hyperparameter search or follow-up architecture change is authorized.
