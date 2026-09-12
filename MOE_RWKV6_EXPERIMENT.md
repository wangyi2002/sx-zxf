# Mamba + RWKV-6 Temporal MoE Experiment

## 1. Goal

This branch explores a parameter-efficient temporal Mixture-of-Experts design on top of the existing `sx-zxf` MotionAGFormer baseline.

Core hypothesis:

> Different temporal modeling biases can be complementary for 3D human pose estimation. Mamba remains the main temporal expert, while a lightweight RWKV-6 TimeMix branch acts as an auxiliary temporal expert whose contribution is selected by a learned router.

The current version intentionally keeps the parameter count close to the baseline instead of placing two full experts in all 30 blocks.

---

## 2. Branch

```text
moe-rwkv6
```

The original `main` branch remains unchanged.

---

## 3. Current architecture

The original 30-layer Mamba / Attention schedule is retained.

### Original Attention blocks

These are kept unchanged:

```text
Spatial Attention
      ↓
Temporal Attention
      ↓
Shared MLP
```

### Original Mamba blocks

Only these blocks are upgraded to temporal MoE:

```text
Spatial Mamba
      ↓
Shared feature X [B,T,J,128]
      │
      ├──────────────→ Mamba Temporal Expert, D=128 ───────────┐
      │                                                       │
      ├→ Linear 128→64 → RWKV6 TimeMix, D=64 → Linear 64→128 ┤
      │                                                       │
      └──────────────→ Router [B,T,J,2] ──────────────────────┘
                              ↓
                         Softmax fusion
                              ↓
                          Shared MLP
```

For the default `n_layers=30` schedule, the original model contains 13 Mamba blocks, so only those 13 blocks use temporal MoE. The remaining Attention blocks are unchanged.

---

## 4. Why the RWKV expert is lightweight

The first version used a full RWKV-6 block in every layer, including both TimeMix and ChannelMix. That increased the model from roughly baseline scale to about 19M parameters and also increased activation memory substantially.

The current version makes three reductions:

1. **MoE only in the original Mamba blocks**, rather than all 30 blocks.
2. **RWKV TimeMix only**; RWKV ChannelMix is removed from the active expert path because the outer block already contains a shared AGFormer MLP.
3. **RWKV bottleneck dimension = 64**, while the shared backbone stays at 128 dimensions.

The active RWKV path is therefore:

```text
128 → 64 → RWKV6 TimeMix(64) → 128
```

The full `RWKV6Block` implementation is still kept in `model/modules/rwkv6.py` for future ablations, but the current MoE does not use it.

---

## 5. Router design

The router remains dense and frame-joint adaptive:

```text
[B,T,J,128]
      ↓
Linear(128, 32)
      ↓
GELU
      ↓
Linear(32, 2)
      ↓
Softmax
```

With the default configuration:

```yaml
router_hidden_ratio: 0.25
```

Router output:

```text
[B,T,J,2]
```

and:

```text
w_mamba + w_rwkv = 1
```

The final temporal representation is:

```text
Y = w_mamba * Y_mamba + w_rwkv * Y_rwkv
```

Both experts still process complete temporal trajectories, so frame-wise routing weights do not break temporal continuity inside either expert.

---

## 6. RWKV-6 lightweight configuration

Default H36M settings:

```yaml
rwkv_dim: 64
rwkv_head_size: 32
rwkv_mix_rank: 16
rwkv_decay_rank: 32
```

Thus the RWKV expert has:

```text
shared feature dim = 128
expert feature dim = 64
head size          = 32
number of heads    = 2
mix low-rank dim   = 16
decay low-rank dim = 32
```

The low-rank dimensions are scaled down together with the expert width to keep the auxiliary expert compact.

---

## 7. Files changed

### `model/modules/rwkv6.py`

Adds the active lightweight expert:

```text
RWKV6TimeMixTemporalExpert
```

It performs:

```text
[B,T,J,128]
→ [B*J,T,128]
→ 128→64
→ RWKV6 TimeMix
→ 64→128
→ [B,T,J,128]
```

The full RWKV-6 TimeMix + ChannelMix classes are retained for future comparison.

### `model/modules/temporal_moe.py`

`TemporalMoE` now contains:

- full-width Mamba temporal expert
- lightweight RWKV6 TimeMix temporal expert
- frame-joint router
- dense soft fusion

### `model/MotionAGFormer_MoE.py`

The original block schedule is preserved:

- original Mamba blocks → Spatial Mamba + Temporal MoE
- original Attention blocks → Spatial Attention + Temporal Attention

`model.moe_layer_indices` records which layers contain MoE.

### `model/modules/mamba.py`

Adds a small CPU reference selective-scan fallback so architecture smoke tests can run on CPU even without `mamba-ssm`.

Important:

> GPU training behavior is unchanged. CUDA inputs still use the original optimized `mamba_ssm` selective scan and will raise an error if `mamba-ssm` is unavailable.

### `tools/cpu_smoke_test_moe.py`

Reusable verification script. It:

- instantiates the production H36M model for exact parameter counting
- instantiates the original baseline for comparison
- runs the full 30-layer MoE architecture on CPU with a short sequence
- validates output and router shapes
- checks that outputs are finite

Run:

```bash
python tools/cpu_smoke_test_moe.py
```

### `configs/h36m/MotionAGFormer-moe-rwkv6.yaml`

Contains the lightweight MoE hyperparameters.

### `utils/learning.py`

Passes the new lightweight MoE configuration into `MotionAGFormerMoE`.

---

## 8. CPU verification rule

For this experimental branch, every future model-code modification should be followed by:

1. CPU model instantiation.
2. A small-batch forward pass.
3. Output-shape validation.
4. Router-shape validation when applicable.
5. Total parameter count.
6. Trainable parameter count.

The CPU selective-scan reference exists specifically to support this smoke-test workflow. It is not intended for speed benchmarking or final training.

### Latest verified result

Production H36M configuration (`n_layers=30`, `dim_feat=128`, `n_frames=243`):

```text
Lightweight MoE parameters: 12,072,527
Baseline parameters:        11,342,421
Increase:                      730,106  (+6.44%)
```

MoE layer indices:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23]
```

CPU smoke test used the full 30-layer schedule with `B=1`, `T=9`, `J=17`:

```text
output shape:        [1, 9, 17, 3]
router tensor count: 13
router shape:        [1, 9, 17, 2]
finite outputs:      PASS
```

---

## 9. Training configuration

Current H36M config:

```text
configs/h36m/MotionAGFormer-moe-rwkv6.yaml
```

Training command:

```bash
python train.py --config configs/h36m/MotionAGFormer-moe-rwkv6.yaml
```

With WandB:

```bash
python train.py \
  --config configs/h36m/MotionAGFormer-moe-rwkv6.yaml \
  --use-wandb \
  --wandb-name moe-rwkv6-light
```

---

## 10. Recommended ablations

```text
A0  Original sx-zxf baseline
A1  Current lightweight MoE
A2  Fixed 0.5 / 0.5 fusion instead of learned router
A3  RWKV dim 32 / 64 / 96
A4  MoE in fewer selected Mamba layers
A5  Joint-level routing [B,J,2] vs frame-joint routing [B,T,J,2]
A6  Motion-aware router using Δ features
```

The current primary research question is:

> Can a small RWKV-6 temporal adapter provide complementary temporal information to the existing Mamba path while keeping parameters and memory close to the original patent baseline?
