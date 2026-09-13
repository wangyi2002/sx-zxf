# SX-MSM-02：梯度裁剪与训练异常监控

- 日期：2026-09-13；分支：`feat/msm-dt-v1`。
- 基于：`ee5c34c874599e7706e11f44d8c2a9028166d408`。
- 背景：SX-MSM 曲线约第 14 轮出现损失突增，并伴随骨长/角度误差升高。
  原因尚未确认。本次加入全局梯度裁剪和诊断，不修改 Δt 公式、双偏置、
  尺度损失分母、学习率、模型结构或训练损失权重。

## 配置与使用

```yaml
grad_clip_norm: 1.0
monitor_every: 100
```

`grad_clip_norm` 是全模型梯度的 L2 范数阈值，不是逐参数/逐层裁剪。
每个 batch 在 backward 后、optimizer.step 前执行。设为 `0` 关闭裁剪，
仍统计梯度范数。阈值 1.0 是本轮起始设置，不保证消除所有异常。
当前代码没有 AMP GradScaler；以后加入 AMP 时，必须先 unscale 再裁剪。

`monitor_every` 每轮从第 0 个 batch 起，每隔指定 batch 数采集 Δt/姿态诊断；
设为 `0` 关闭这部分采样。梯度统计始终覆盖每个 batch。
统计按 epoch 汇总并与原有 loss/eval 指标在同一次 `wandb.log(step=epoch+1)`
中上传，不改变 W&B 的 step 语义，也不创建新的 W&B run。
因此这些图是 epoch 级统计，不是逐 batch 的实时曲线。

```bash
# 从头训练，继续使用默认 checkpoint 目录覆盖保存
python train.py --config configs/h36m/MotionAGFormer-msm-dt.yaml --seed 0 --use-wandb --wandb-name sx-msm-dt-clip1-s0
```

已启动的训练进程不会自动应用新代码；需下次启动时生效。
参数名、形状与 checkpoint 格式未改变，原 SX-MSM checkpoint 可续训：
在上述命令中加入 `--checkpoint checkpoint --resume`。这属于中途改变
优化策略，应记录切换 epoch，不能当作“从头使用裁剪”的独立实验。

主 MSM 配置和两份单偏置对照配置均显式启用上述设置。
原 `MotionAGFormer-base.yaml` 未修改，缺省裁剪/Δt采样关闭。
新增 `MotionAGFormer-baseline-monitored.yaml` 提供无 MSM、但相同裁剪/监控
设置的匹配基线。比较机制收益时，应给基线与 MSM 使用相同裁剪设置。

## W&B 中优先查看的指标

| 指标 | 含义 |
|---|---|
| `grad/global_norm_mean` | 所有 batch 裁剪前全局梯度范数的均值 |
| `grad/global_norm_max` | 本轮裁剪前梯度范数最大值，用于定位尖峰 |
| `grad/clipped_steps_fraction` | 本轮梯度范数超过阈值的 batch 比例 |
| `grad/clip_threshold` | 裁剪阈值，0 表示关闭 |
| `grad/msm_norm_mean`、`grad/msm_norm_max` | 新增 MSM 权重梯度的裁剪前 L2 范数 |
| `msm/<层名>/dt_mean`、`dt_min`、`dt_max` | 加入扫描偏置并 softplus 后的有效 Δt |
| `msm/<层名>/dt_p99_sampled` | 各次采样/各副本 p99 的最大值，非全 epoch 精确 p99 |
| `msm/<层名>/dt_below_1e-5_fraction` | 有效 Δt 过小的比例，阈值仅用于观测 |
| `msm/<层名>/residual_rms` | 新增卷积残差的 RMS |
| `msm/<层名>/residual_to_base_logit_rms` | 残差 RMS / 原 Δt logits RMS 的最大观测比值，分母包含原偏置 |
| `msm/<层名>/dt_to_baseline_mean_ratio` | 有效 Δt 均值 / 同次输入去掉卷积残差后的 Δt 均值，取最大观测比值 |
| `pose/pred_limb_mean`、`target_limb_mean` | 预测/目标的平均骨长 |
| `pose/limb_ratio_mean`、`limb_ratio_min` | 预测/目标平均骨长之比的均值/最小值 |
| `pose/pred_rms` | 预测坐标幅度 |
| `pose/scale_denominator_min`、`scale_denominator_mean` | 原 n_mpjpe 分母的最小值/均值，未修改训练分母 |
| `pose/pred_velocity_rms`、`target_velocity_rms` | 预测/目标一阶时间差分 RMS |

层名例如 `layers_0_temporal_mixer`。每个时间 MAM 独立统计，不混合成一条曲线。
无 MSM 的匹配基线仍记录时间 MAM 的有效 Δt，但没有残差指标。
Δt 均值按元素数加权；min/max 为采样 batch/副本的极值。
p99 每次最多抽取 4096 个确定性间隔样本，避免大张量排序。
骨架指标使用训练数据本身的坐标单位，不额外转换成毫米。

诊断使用 detach/no_grad，只保留 Python 标量。DataParallel 副本通过共享回调
与锁汇总，不依赖副本上的属性更新，不保留计算图。训练结束/异常退出时
清除回调，验证阶段不采集训练 Δt 统计。采样仍有额外计算和设备同步成本；
过短的单 batch 异常可能被 Δt/姿态采样漏掉，必要时将间隔降至 10 或 1。

检测到非有限损失或梯度时会报错，并在该次 optimizer.step 前停止；
不会尝试把 NaN/Inf 裁剪成有效值。异常导致本轮未完成时，该轮 W&B 汇总
可能尚未上传，应同时保留终端错误日志。

## 验证

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tools/verify_training_monitor.py --device cpu
# 本地 GPU 检查
python tools/verify_training_monitor.py --device cuda
```

CPU 环境：Python 3.12、PyTorch 2.5.1+cpu、timm 0.9.16。
11 项测试通过，包括原 MSM 数值测试，以及裁剪阈值、关闭裁剪、非有限梯度
拒绝、监控不改变输出/梯度/state_dict、采样开关、有效偏置、姿态退化指标。

完整 30 层、128 维模型使用 `[1,243,17,3]` 合成输入和实际姿态损失，通过
forward、backward、裁剪、AdamW 更新。脚本直接提取 train.py 的实际
`train_one_epoch` 函数执行，避免加载无关的数据集和 W&B 服务依赖；
未用模型/损失占位实现，也未运行 Human3.6M 数据集训练。

- 参数量：**11,555,413**，新增参数 **0**；state_dict 键不变。
- 裁剪前梯度范数：**17.00019455**；裁剪后：**0.99999988**。
- 采集 **119** 个标量统计项，13 个时间 MAM 均有 Δt 记录。
- 合成训练总损失：6.85969591，仅用于验证，不是数据集 MPJPE。
- 新 MSM 权重梯度有限且非零；优化器更新后参数有限。
- CUDA、多 GPU 和实际 W&B 服务上传未在当前 CPU 环境验证。

本轮主要关注尖峰是否消失、裁剪比例是否持续过高，以及验证集 MPJPE
能否正常下降。若 Δt/骨架指标仍异常，再依据记录选择下一项单独修改。
