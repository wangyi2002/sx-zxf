# E6: XY-guided depth specialization

Source: `exp/xy-z-late-split@f778e88e1f21fc0c3edcb4bf3120660c6e7c66ef`.
Clean baseline: `depth-diagnosis@e1d208b9aeb02ff1be6edd15445e8356de7a420a`.
Branch: `exp/xy-guided-depth`.

## Hypothesis and architecture

Test whether reading XY-specialized features helps the dedicated depth stream.
E5's small single-seed gain motivates this experiment but does not establish
specialization or exclude increased capacity as its explanation. GT-XY input
intervention motivates the direction; it does not prove internal XY features
are accurate geometry or that this interaction will work.

Keep 28 shared blocks and two independent spatial-temporal blocks per stream.
After the first independent block, normalized Z provides queries and normalized
XY provides keys/values to 8-head, 128-channel spatial cross-attention. Each frame
attends across its 17 joints independently. Add the projected result to Z before
the second Z block, which retains its spatial and temporal processing. XY never
reads Z in the forward pass. No detach: Z gradients can reach the first XY suffix
block once the projection learns. The second XY suffix block is downstream of
the read and is not on the Z-only gradient path. Shared features and the original
joint loss still couple optimization.

Output projection weight/bias start at zero, initially matching E5. At the first
step only this projection receives nonzero interaction gradients; Q/K/V and
normalizations become active after the projection updates. No extra zero gate,
auxiliary loss, coordinate correction head, or temporal cross-attention.

`use_xy_depth_interaction` defaults false. Enabled mode requires split=true and
exactly two suffix blocks. Old configs retain their architecture/state keys.
E6 checkpoints require the E6 config and strict loading. Do not load E5 trained
weights: this experiment trains from scratch. Baseline and E5 files are retained.

## Protocol

60 epochs, seed 0, batch 4, T=243, original optimizer and losses:
MPJPE + 0.5 normalized MPJPE + 20 velocity loss. No hyperparameter sweep.
Report action-macro MPJPE, P-MPJPE, X/Y/Z MAE, Bone delta-Z and ordering through
the existing diagnostic. Keep training best-epoch and diagnostic values distinct.
A positive result still requires later capacity controls/repeats before causal claims.

## Local commands (repository root)

Ensure current work is committed/saved before switching. For a new local branch:

```bash
git fetch origin
git switch --track origin/exp/xy-guided-depth
python tools/smoke_xy_guided_depth.py --device cuda --batch-size 4 --frames 243
python train.py --config configs/h36m/MotionAGFormer-xy-guided-depth.yaml --new-checkpoint checkpoint/xy_guided_depth --seed 0 --use-wandb --wandb-name xy-guided-depth-seed0
python diagnose_depth.py --config configs/h36m/MotionAGFormer-xy-guided-depth.yaml --checkpoint checkpoint/xy_guided_depth/best_epoch.pth.tr --input-source metadata --batch-size 1 --num-workers 0 --seed 0 --output-dir depth_outputs/xy_guided_depth
```

For an existing branch, switch to it and use `git pull --ff-only` only with a
clean, non-diverged worktree. Preserve E5 and baseline weights. Do not add weights
or datasets to Git. GPU smoke compares E5/E6 in isolation on synthetic data; its
step timing is not a real epoch benchmark. Do not start full training if it fails.

## Validation

CPU smoke uses the repository's explicit selective-scan reference, not CUDA Mamba.
Checks cover full-length initial E5 equivalence, strict E6 roundtrip, two original
loss/clip/AdamW updates, learned XY-to-Z gradients, XY-only isolation from Z,
and per-frame spatial interaction. GPU checks and full training are performed
locally; neither is implied by a CPU pass.

### Executed checks (2026-10-07)

Python 3.12, PyTorch 2.14.1+cpu, timm 1.0.22. The pinned timm 0.6.11 fails
on Python 3.12 dataclass import; only this check environment used newer timm.
Repository requirements were not changed. DropPath is disabled by this config.
Full `[1,243,17,3]` CPU initial output difference from E5: **0.0**.
E5: 12,201,301 parameters; E6: **12,267,477** (+66,176, about 0.54%).
The documented E6 checks passed, using T=9 for CPU backward. Existing E5 smoke
also passed at T=9. No GPU measurements or real-data training were run here.
