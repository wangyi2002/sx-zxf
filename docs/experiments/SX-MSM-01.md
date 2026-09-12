# SX-MSM-01：师兄 Mamba1 架构上的时间动态 Δt

- 日期：2026-09-12
- 分支：`feat/msm-dt-v1`
- 基于：`wangyi2002/sx-zxf/main`，提交 `a2346dd89a1d7edf696475f61631d5145a8d45e9`
- 状态：代码与 CPU 验证完成；CUDA 验证和 Human3.6M 训练结果待填。
- 目标：检验在师兄 MAM 上给 Δt 增加两帧时间卷积，能否改善姿态估计。

## 1. 修改边界

仅在时间 MAM 的 Δt 生成路径增加卷积残差。默认 30 层配置中共有
13 个时间 MAM，因此有 13 组新增权重。空间 MAM 不增加 MSM 参数。

本轮不加 SSI，不更改前置 GCN、Attention/Mamba 的层级排列、MLP、
SSM 状态维度、损失、优化器、学习率策略、数据和帧数。
原始 `MotionAGFormer-base.yaml` 保持原样；缺省 `temporal_msm=False`。

## 2. 实现公式与 SAMA 的关系

令 `u_t` 为进入时间 MAM 的特征（上游已完成 LayerNorm），
`r_t` 为师兄原有 x 分支卷积、x_proj 得到的低秩 Δt 特征。
新增无偏置的两帧卷积：

```text
m_t = K_previous u_(t-1) + K_current u_t
legacy_double: Δ_t = softplus(W_dt r_t + 2 b_dt + m_t)
single:        Δ_t = softplus(W_dt r_t +   b_dt + m_t)
```

实现用 `F.pad(..., (1, 0))` 和 `F.conv1d`，卷积核大小为 2，
首帧的前一帧补零，不用 roll，不跨 batch/clip/关节串接。
输入重排为 `[B*J, C, T]` 后才计算卷积。
默认 `C=128`，输出为实际 SSM 通道数 `64`，不是 Mamba2 的 4 个 head。

两个卷积核均零初始化，且不消耗额外随机数。因此，相同 seed 时，
所有原有参数的初始化与基线一致，初始输出也一致。
卷积权重直接接收梯度，不增加额外零门控。

这是借鉴 SAMA 时间卷积生成 Δt 思想的**残差适配**，不是完整复现：

- 保留 Mamba1 原有输入相关 Δt，卷积负责学习增量；不替换原通路。
- 使用前一帧/当前帧；提供的 SAMA `kernel_size=2, padding="same"`
  与此边界约定不同，不能称为原样移植。
- 没有显式计算 `u_t-u_(t-1)`；两组权重可以学习差分，也可以学习其他组合。
- 此适配也不与 motionformer/Mamba2 的 packed-projection 改法逐项等价。
  两种骨干上的结果应分别与各自基线比较。
- 仅新增卷积残差是因果的。原有 x 分支使用居中卷积，且整体有时间
  Attention/GCN，因此不能把整个网络或完整 Δt 路径称为因果模型。

## 3. 双偏置隔离与配置

原始 MAM 在 `dt_proj` 中加一次偏置，又通过扫描函数的 `delta_bias`
加一次。主实验保留这一行为，避免把偏置修正与 MSM 收益混在一起。
`mamba_dt_bias_mode=single` 为独立对照，同时作用于空间和时间 MAM。

| 配置文件（位于 configs/h36m/） | MSM | 偏置 | 用途 |
|---|---|---|---|
| MotionAGFormer-base.yaml | 关闭 | 原始双偏置 | 主实验基线 A |
| MotionAGFormer-msm-dt.yaml | 开启 | 原始双偏置 | 主实验 B |
| MotionAGFormer-baseline-single-bias.yaml | 关闭 | 单偏置 | 可选基线 C |
| MotionAGFormer-msm-dt-single-bias.yaml | 开启 | 单偏置 | 可选实验 D |

**第一轮只比较 A 与 B。** 若后续研究偏置问题，再比较 A/C 和 C/D；
不要用 A/D 的差值解释 MSM 的独立收益。

参数经过 `utils.learning.load_model → MotionAGFormer → create_layers →
MotionAGFormerBlock → AGFormerBlock → MAM` 传递，不是仅写进 YAML。

## 4. 检验与参数量

CPU 环境：Python 3.12、PyTorch 2.5.1+cpu、einops 0.8.2、timm 0.9.16。
仓库原有 timm 0.6.11 在此 Python 版本导入时报 dataclass 错误，
因此仅本次验证环境使用 0.9.16，未更改仓库 requirements 或要求训练环境升级。

```bash
python -m unittest discover -s tests -p test_msm_dt.py -v
python tools/verify_msm_dt.py --device cpu
```

已通过：

- 6 项单元测试：零初始化参数/输出/输入梯度一致，单帧与多帧边界，
  batch 隔离，两帧卷积公式，空间模块无新增参数，新增参数可学习，
  单/双偏置的差值，CPU 递推闭式解与数值梯度检查。
- 完整默认模型：30 层、128 维，输入和输出均为 `[1,243,17,3]`。
- 完整模型零初始化输出与同配置基线最大误差 **0**。
- 真实 CPU SSM 递推、前向、反向和一次 AdamW 更新通过；
  13 组 MSM 权重均有非零有限梯度，更新后参数和输出发生变化。
- 合成 MSE 为 `1.0523490905761719`，仅用于流程检验，**不是 MPJPE**。
- 额外用固定 main 提交的原模型源码对照：关闭 MSM 时，参数键、
  参数值及小尺寸模型输出完全一致。该对照仅将原 CUDA 扫描替换为
  相同 CPU 递推，以便在 CPU 上执行。

| 配置 | 实际实例化参数量 |
|---|---:|
| 基线 A | 11,342,421 |
| MSM B | 11,555,413 |
| 增量 | 212,992（约 1.88%） |

新增量为 `13 × 64 × 128 × 2`。参数量包含模型原有的未参与 forward
的 BatchNorm 参数。未测训练峰值显存、吞吐或数据集精度。

CPU 后端在 `model/modules/scan_backend.py`，计算实际 SSM 递推，
只支持本仓库使用的实数、非分组 B/C 形式。CUDA 路径仍调用原来
`mamba_ssm.ops.selective_scan_interface.selective_scan_fn`，不静默回退 CPU。

本次环境没有 CUDA。正式训练前在本地执行：

```bash
python tools/verify_msm_dt.py --device cuda
```

该命令先检查 CUDA 扫描与参考递推的输出/梯度，再运行完整模型与参数统计。
它使用 batch=1 的合成数据，不能代替 batch=4 的训练显存测量。

## 5. 第一轮训练

在仓库根目录、已配置原训练环境和数据的情况下执行：

```bash
# A：原始基线（若已有同提交、同配置、同 seed 的有效记录，可复用）
python train.py --config configs/h36m/MotionAGFormer-base.yaml --seed 0 --new-checkpoint checkpoint/sx-baseline-s0 --use-wandb --wandb-name sx-baseline-s0

# B：本轮 MSM 实验
python train.py --config configs/h36m/MotionAGFormer-msm-dt.yaml --seed 0 --new-checkpoint checkpoint/sx-msm-dt-s0 --use-wandb --wandb-name sx-msm-dt-s0
```

主实验从头训练，不加 `--checkpoint` 或 `--resume`。
原基线 checkpoint 不能直接作为 MSM 的续训 checkpoint：新增参数导致
严格加载/优化器状态不匹配。验证脚本的 `strict=False` 仅用于检验
参数兼容关系，且断言缺失键恰好为新增 MSM 权重。
同一 MSM 配置自身的 checkpoint 可按原仓库方式恢复训练。

记录实际 commit、配置、seed、数据划分、P1/P2、加速度误差、
最佳 epoch、训练时间和峰值显存。先观察 A/B，再决定是否增加 seed
或展开 C/D，不同时叠加 SSI、MLP 删除或其他模块。

## 6. 修改文件与后续记录

- `model/modules/mamba.py`：时间卷积残差、偏置模式。
- `model/modules/scan_backend.py`：CPU 真实递推与 CUDA 分发。
- `model/MotionAGFormer.py`、`utils/learning.py`：开关传递。
- 三份新增 YAML：主实验与两份可选偏置对照。
- `tests/test_msm_dt.py`、`tools/verify_msm_dt.py`：数值和完整流程检验。

| 实验 | commit / seed | P1 | P2 | 加速度误差 | 显存/耗时 | 结论 |
|---|---|---|---|---|---|---|
| A 原始基线 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| B MSM | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
