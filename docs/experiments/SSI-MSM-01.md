# SSI + MSM combined experiment

## Scope and provenance

Branch: `feat/ssi-msm-state-v1`, based on SSI commit
`8d153628adf1cc9ecc0f492dc7a29e852babbab2` (itself based on sx/main).
Temporal direct MSM is ported from `feat/msm-dt-v1` commit
`a984edf052dfd9080c48640b99f1377b4684a91a`.

Earlier experiments: baseline roughly 38.4 mm, direct MSM roughly 38.9 mm,
feature SSI roughly 39 mm (user-reported; single runs, clipping not fully matched).
Post-training alpha intervention worsened SSI (0.5 ~44 mm, 0 ~55 mm): this
is dependence on a trained path, NOT evidence of positive architectural benefit.

This branch provides TWO experiments. Run feature SSI+MSM first for a clean
combination of the existing modules, then optionally add state SSI. Neither
configuration changes losses, dataset, LR schedule or checkpoint saving rules.

## Switches

| YAML option | Meaning |
|---|---|
| spatial_ssi: True | Fixed graph feature fusion inside every spatial MAM |
| temporal_msm: True | Previous/current temporal Conv1d directly generates dt logits |
| spatial_state_ssi: False/True | Disable/enable fixed graph fusion of spatial states before readout |
| state_ssi_chunk_size: 32 | Number of batch*frames sequences processed per state-scan chunk |
| grad_clip_norm: 1.0 | Global norm clipping, after backward and before optimizer step |

`MotionAGFormer-ssi-msm.yaml`: feature SSI + MSM, state SSI OFF.
`MotionAGFormer-ssi-msm-state.yaml`: feature SSI + MSM, state SSI ON.
Change `spatial_state_ssi` manually BEFORE a fresh run to select the experiment.
State fusion introduces no parameters; checkpoints load across the toggle, but
this does NOT make toggling at evaluation a valid retrained ablation. Use the
same config during training and analysis; the checkpoint does not encode this flag.

"Input SSI" here means the x branch AFTER local conv+SiLU and BEFORE B/C/dt
projection inside spatial MAM, as in the previous SSI experiment. It is not an
additional GCN on raw input coordinates. z is unchanged. All 13 spatial MAMs
use the same fixed H36M topology, with no learned graph or softmax.
P = D^(-1/2)(A+I)D^(-1/2); x' = x + P x (joint-first notation).

MSM applies only to the 13 temporal MAMs, using kernel size 2, left zero padding,
input C=128, output 64 SSM channels. dt = softplus(Conv(u)+dt_bias), bias once.
Temporal B/C still come from x. Spatial legacy dt bias behavior is unchanged.

## What hidden-state SSI does; official SSM implications

h_j = exp(delta_j A) h_(j-1) + delta_j B_j x'_j
H_j = h_j + sum_k P[j,k] h_k
y_j = C_j H_j + D x'_j

This mixes the complete spatial state sequence before C readout. It does NOT
feed H_j into the next recurrence, does NOT mix different frames or samples,
and does NOT mix state channels. It can use later-index joints; spatial joint
indices are not a causal time axis. D skip is added once and is not graph-mixed
again in the state readout.

- State SSI OFF: CUDA still uses official `selective_scan_fn`; CPU uses the
  preexisting reference recurrence for verification.
- State SSI ON: spatial scan calls repository-owned `state_ssi.py`, a real
  differentiable PyTorch recurrence on CPU/CUDA. The installed official package
  is NOT edited, no custom CUDA compilation is needed. Temporal scan remains official on CUDA.
- Why a custom scan: the current official interface does not expose all internal
  states needed for this readout. Fusing final output y is not equivalent to fusing h.
- This is a fixed-graph adaptation inspired by SSI, not an exact SAMA reproduction.
- Training chunks batch*frames and activation-checkpoints the custom state scan.
  This retains the mathematical output, but adds recomputation. It can be slower
  than the fused official kernel. GPU speed/peak memory are not verified here.
  Reducing chunk size can reduce temporary state memory but adds overhead.

## Commands

Fresh feature+MSM run:
```bash
python train.py --config configs/h36m/MotionAGFormer-ssi-msm.yaml --seed 0 --use-wandb --wandb-name sx-ssi-msm-s0
```
Fresh state+feature+MSM run:
```bash
python train.py --config configs/h36m/MotionAGFormer-ssi-msm-state.yaml --seed 0 --use-wandb --wandb-name sx-ssi-msm-state-s0
```
Both write `checkpoint/best_epoch.pth.tr` and `checkpoint/latest_epoch.pth.tr`.
Preserve earlier weights before starting a new run if they are needed for analysis.
Do not resume a different architecture. Resume same experiment by adding
`--checkpoint checkpoint --resume` with the original config.

## Delta visualization

After training, choose the SAME config as the saved checkpoint:
```bash
python scripts/visualize_joint_dt.py \
  --config configs/h36m/MotionAGFormer-ssi-msm-state.yaml \
  --checkpoint checkpoint/best_epoch.pth.tr \
  --max-batches 100 \
  --output outputs/joint_dt
```
Set `--max-batches 0` for the entire test set. Default batch size is 1; inference
uses a single device. Script rejects a nonexistent checkpoint and loads strictly.
Requires the same local `data/reader` package and H36M data as train.py; these
are not present in the cloud checkout. matplotlib is needed for plotting.

Outputs: `joint_statistics.csv`, one PNG per temporal MAM, and
`all_layers_mean.png`. Offline plotting does not need W&B and does not change
training metrics. Existing four gradient metrics remain logged to W&B.

Definitions:
- Actual positive scan delta AFTER bias and softplus, not logits or dt_bias alone.
- For each layer and joint, mean over samples, channels and frames t=1..T-1.
  First frame excluded because temporal convolution uses left padding.
- Motion intensity: mean Euclidean displacement of INPUT 2D xy between adjacent
  frames, in normalized input units/frame. Confidence channel excluded. It is
  not GT 3D velocity, not root-relative velocity and not feature-space velocity.
- Both measurements use identical unflipped clips and exclude clip boundaries.
  No flip testing: avoids mixing left/right joint identities in the statistics.
- Sort joints by decreasing measured motion, using the SAME order for all layers.
  Dual axes preserve distinct units. No fabricated limb/trunk partition or values.
- Global curve is the equally weighted mean of the 13 layer means. Per-layer
  curves/CSV are preferable for diagnosing heterogeneity. Delta alone does not
  determine memory length: A also matters. Correlation does not establish causality.
- With a batch limit, this is a deterministic first subset, not guaranteed to
  represent all actions. Final analysis should use the full set. Dataset window
  repetitions/padding are weighted as presented by the existing loader, not
  deduplicated into unique source frames.

## Validation

Cloud CPU checks use synthetic data, not H36M accuracy. No test files are added.
Parameter count for BOTH combination variants: **11,542,101**.
Feature/state SSI add no trainable parameters; compared with senior base
11,342,421, the increase is 199,680 from direct temporal MSM.
Validation details and limitations are recorded after the checks below.

Completed checks:
- Both full 30-layer models: [1,243,17,3] forward, MPJPE+velocity-loss backward,
  finite gradients, global clip=1 and AdamW step passed. Preclip norms ~25.888
  on synthetic random targets (not comparable to real training metrics).
- Zero graph state scan agrees with original CPU scan in outputs and gradients.
- Nonzero graph agrees with explicit per-joint state fusion formula; numerical
  gradient check passed; chunk sizes 1/32 agree.
- Feature-only legacy MAM parameters/outputs exactly agree when new flags are off.
- State SSI has a measured nonzero effect at isolated MAM level and state
  parameters receive gradients (tiny outer LayerScale can hide initial full-model differences).
- Delta observer joint/channel/sample reductions checked against known tensors,
  including unequal batch weighting; all 13 temporal layers collected.
- PNG/CSV output generated and visually inspected with synthetic observations.
  This verifies the plotting path, not the real-data trend. metadata.json records
  config, checkpoint path/epoch and sample count for actual runs.
- No GPU kernel execution, multi-GPU behavior, real-data loader execution or
  wall-time/VRAM measurements available in cloud. CPU torch 2.5.1, timm 0.9.16
  (cloud Python 3.12 compatibility); repository requirements unchanged.
