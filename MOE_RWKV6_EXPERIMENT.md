# Mamba + RWKV-6 Temporal MoE Experiment

## 1. Goal

This branch explores a first-stage temporal Mixture-of-Experts design on top of the existing `sx-zxf` MotionAGFormer baseline.

The current hypothesis is:

> Different joints and different motion states may benefit from different temporal modeling biases. Therefore, instead of using one fixed temporal operator in every block, use Mamba and RWKV-6 as parallel temporal experts and learn data-dependent fusion weights with a router.

The first implementation intentionally keeps the design simple so that the effect of expert fusion can be evaluated independently.

---

## 2. Branch

Experiment branch:

```text
moe-rwkv6
```

The original `main` branch is not modified by this experiment.

---

## 3. Main architectural change

The original block is conceptually:

```text
Spatial Module
    ↓
Temporal Module
    ↓
Next Block
```

The MoE version is:

```text
Spatial Module
    ↓
Shared Feature X [B,T,J,C]
    ├────────────→ Mamba Temporal Expert ──┐
    ├────────────→ RWKV-6 Temporal Expert ─┤
    └────────────→ Router ──────────────────┘
                         ↓
                 Softmax weights
                  [B,T,J,2]
                         ↓
                 Weighted Fusion
                         ↓
                    Next Block
```

The original spatial schedule is retained:

- early blocks: Spatial Mamba
- middle blocks: Spatial Attention
- later blocks: Spatial Mamba / Attention according to the original `create_layers` schedule

Only the temporal stage is replaced by a Mamba + RWKV-6 dense expert module.

---

## 4. Router design

The initial router is a small MLP:

```text
C
↓
Linear(C, C * router_hidden_ratio)
↓
GELU
↓
Linear(..., 2)
↓
Softmax
```

Input:

```text
[B,T,J,C]
```

Output:

```text
[B,T,J,2]
```

Therefore every frame and every joint receives two fusion weights:

```text
w_mamba
w_rwkv
```

with:

```text
w_mamba + w_rwkv = 1
```

The router is initialized so the first forward pass starts close to:

```text
Mamba = 0.5
RWKV  = 0.5
```

This is a **dense soft routing** design, not sparse Top-K routing.

Both experts always process the complete sequence, so frame-level routing weights do not break temporal continuity inside Mamba or RWKV.

---

## 5. Expert fusion

For shared feature `X`:

```text
Y_mamba = Mamba(X)
Y_rwkv  = RWKV6(X)
```

Router:

```text
W = softmax(Router(X))
```

Fusion:

```text
Y = W[..., 0] * Y_mamba + W[..., 1] * Y_rwkv
```

Both expert outputs preserve the original shape:

```text
[B,T,J,C]
```

---

## 6. RWKV-6 pose adaptation

The new RWKV module is implemented in:

```text
model/modules/rwkv6.py
```

The pose wrapper converts:

```text
[B,T,J,C]
```

into independent joint trajectories:

```text
[B*J,T,C]
```

RWKV-6 then performs temporal modeling along `T`, after which the output is restored to:

```text
[B,T,J,C]
```

This matches the existing temporal Mamba behavior in the repository.

The current RWKV-6 implementation is a **pure PyTorch reference implementation**. It contains:

- token shift / time mixing
- data-dependent RWKV-6 mixing
- dynamic decay
- R/K/V/G projections
- matrix-valued recurrent state
- ChannelMix
- residual block structure

It intentionally does not depend on an external RWKV package or custom CUDA kernel.

Important limitation:

> The WKV recurrence currently uses a Python loop over the temporal dimension. It is suitable for architecture verification and ablation, but it is not appropriate for final speed benchmarking. A CUDA WKV6 kernel can replace the internal recurrent operator later without changing the MoE interface.

---

## 7. New files

### `model/modules/rwkv6.py`

Implements:

- `RWKV6TimeMix`
- `RWKV6ChannelMix`
- `RWKV6Block`
- `RWKV6TemporalExpert`

### `model/modules/temporal_moe.py`

Implements:

- `TemporalMoE`
- `TemporalMoEBlock`

`TemporalMoE` contains:

- existing repository Mamba temporal expert
- RWKV-6 temporal expert
- frame-joint-level router
- soft weighted fusion

### `model/MotionAGFormer_MoE.py`

Defines the new main architecture:

```text
MotionAGFormerMoE
```

The following parts are retained from the baseline design:

- joint embedding
- spatial GCN stem
- temporal GCN stem
- joint positional embedding
- spatial Mamba / Attention schedule
- representation head
- final 3D pose head

The temporal module of each block is replaced by `TemporalMoEBlock`.

### `configs/h36m/MotionAGFormer-moe-rwkv6.yaml`

First H36M experiment config.

### `utils/learning.py`

Updated so `load_model` supports:

```yaml
model_name: MotionAGFormerMoE
```

---

## 8. Initial configuration

Current H36M settings inherit the baseline large configuration:

```yaml
n_layers: 30
dim_feat: 128
n_frames: 243
num_joints: 17
```

MoE-specific settings:

```yaml
router_hidden_ratio: 0.5
rwkv_head_size: 32
rwkv_ffn_mult: 3.5
```

For `dim_feat=128` and `rwkv_head_size=32`, RWKV uses 4 heads.

---

## 9. Dependencies

No additional third-party RWKV package is required by the new RWKV-6 module.

The new module itself only depends on PyTorch.

However, the repository's existing Mamba implementation already imports:

```python
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
```

so the runtime environment still needs a compatible `mamba-ssm` installation even though it is not currently listed in `requirements.txt`.

---

## 10. Router inspection

`MotionAGFormerMoE.forward` supports:

```python
output, router_weights = model(x, return_router=True)
```

Each element in `router_weights` has shape:

```text
[B,T,J,2]
```

This is intended for later analysis such as:

- joint-specific expert preference
- action-specific expert preference
- frame-wise expert switching
- layer-wise routing heatmaps

Potential visualizations:

```text
Layer × Joint × Expert
```

and comparisons such as:

```text
walking wrist
running wrist
sitting wrist
```

---

## 11. Important experimental caveat

This first implementation puts dense Mamba + RWKV-6 temporal experts in every block.

Therefore compared with the baseline it will increase:

- parameter count
- FLOPs
- activation memory
- optimizer state
- training time

The first experiment should be treated as an architecture feasibility test, not yet as an efficiency-optimized final design.

If the idea is effective, follow-up ablations should evaluate:

1. MoE only in selected blocks.
2. Joint-level `[B,J,2]` routing versus frame-joint `[B,T,J,2]` routing.
3. Fixed `0.5 / 0.5` fusion versus learned router.
4. Mamba-only versus RWKV-only versus MoE.
5. Motion-aware routing using temporal differences.
6. Expert specialization and possible sparse routing.

---

## 12. Recommended first ablation sequence

```text
Experiment 0
Original sx-zxf baseline

Experiment 1
Mamba temporal only

Experiment 2
RWKV-6 temporal only

Experiment 3
Mamba + RWKV-6 fixed 0.5 / 0.5 fusion

Experiment 4
Mamba + RWKV-6 learned router
```

Only after Experiment 4 shows useful expert complementarity should the design be expanded toward more complex routing.

---

## 13. Current research question

The current branch is designed to answer the following first-order question:

> Can Mamba and RWKV-6 provide complementary temporal representations for 3D human pose estimation, and can a lightweight router learn when to rely more on each expert?

If the answer is positive, the next stage can explore true motion-aware expert specialization rather than simply increasing model capacity.
