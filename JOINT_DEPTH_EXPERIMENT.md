# Joint Depth Adapter control

Source: exp/skeleton-relative-depth-adapter @ 0ff5fb47b2e392259d35d2e4223f8d705d8d872e.
Branch: exp/joint-depth-adapter.

Final normalized backbone features F [B,T,17,128] feed one shared
Linear(128,32) -> GELU -> Linear(32,1).
u = MLP(F); r = u - u[..., :1] (root index 0).
Output = [x_head, y_head, z_head + r]. No detach of F or r.
No parent-child feature difference and no path accumulation.
The last layer is zero initialized; initial output equals the original head.
As with SRDA, first-layer/feature branch gradients start at zero and become
available after the last-layer weight updates. The last scalar bias cancels
in root subtraction, so its gradient is always zero; it is retained for the
same nominal 4,161 parameter count (4,160 effective parameters).
Baseline: 11,342,421; adapter: 4,161; total: 11,346,582.
This control compares both SRDA representation and accumulation together;
it does not isolate each component individually.

Config: configs/h36m/MotionAGFormer-joint-depth.yaml.
use_joint_depth_adapter=true, use_skeleton_relative_depth_adapter=false.
Both flags false restores baseline. Both true is rejected.
Training stays at batch 4, 60 epochs, clip 243, original losses and clipping.
Train from scratch with seed 0; do not resume or load SRDA weights.
The old SRDA config and implementation remain available unchanged.

Monitoring retains train/depth_adapter_joint_residual_abs_mean and
train/depth_adapter_bone_residual_abs_mean. For the joint adapter, bone
statistics are detached differences of joint residuals for comparison only;
skeleton connections do not contribute to the prediction or gradient path.
Diagnosis and its metric definitions are unchanged. Use the joint config
with its own best checkpoint. Primary reporting uses action_macro.

Training:
```bash
python train.py --config configs/h36m/MotionAGFormer-joint-depth.yaml --new-checkpoint checkpoint --seed 0 --use-wandb --wandb-name joint-depth-r025-seed0
```

Diagnosis:
```bash
python diagnose_depth.py --config configs/h36m/MotionAGFormer-joint-depth.yaml --checkpoint checkpoint/best_epoch.pth.tr --input-source metadata --batch-size 1 --num-workers 0 --seed 0 --output-dir depth_outputs/joint_depth_seed0
```
