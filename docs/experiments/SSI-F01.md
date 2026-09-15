# SSI-F01：空间 MAM 内固定骨架图的特征融合

## 范围与来源

- 分支：`feat/ssi-feature-v1`。
- 基于 `sx-zxf/main`：`a2346dd89a1d7edf696475f61631d5145a8d45e9`。
- 动机：此前直接时间卷积 Δt 的完整训练未显示明确 MPJPE 收益，
  本轮独立检验空间 MAM 内部的骨架特征交互。
- 借鉴 SAMA 论文式 (5) 的特征融合思想，不是完整 SSI 复现。
  本轮不实现隐藏状态融合、可学习图、MSM 或额外门控/MLP。

## 确切计算路径

空间 MAM 的输入投影、x/z 分支局部卷积与 SiLU 完成后，仅融合 x：

```text
x: [B*T, 64, 17]
A: H36M 16 条骨骼边构成的双向邻接矩阵
P = D^(-1/2) (A + I) D^(-1/2)
x'_j = x_j + sum_k P[j,k] x_k
```

D 是 `A+I` 的度矩阵。P 的行表示接收关节，列表示来源关节；
没有 softmax，非骨骼邻接位置保持零。原始 x 的残差和图中的归一化自连接
均保留，这是本轮明确采用的融合公式。

融合后的 x 用于原来的 x_proj（生成 B/C/Δt 特征）、selective scan 输入
以及 D 跳连；z 分支不变。融合发生在生成 B/C/Δt 之前。
空间 Δt 的生成公式没有改变，但其输入随 x 融合改变；时间 MAM 完全保留
师兄原来的实现，包括原有双偏置行为。未修正尺度损失分母等其他问题。

固定图作用于默认 30 层配置的 **13 个空间 MAM**。各层的图数值一致，
在样本、帧、特征通道之间共享。图不会跨 batch/帧做混合，不含可学习参数。
图作为非持久 buffer 自动随模型迁移设备，由固定拓扑重建。

骨骼边使用 `loss.pose3d.get_limb_lens` 的完整 17 关节定义，包括 0↔4。
没有修改入口处原有的 gcn_s/gcn_t，也没有修改其邻接矩阵。

与 SAMA 的区别：论文式 (4) 包含 softmax 和后续可学习邻接，完整 SSI
还包括式 (6) 的隐藏状态融合。本轮按已确认方案使用固定、保留零连接的
度归一化图，先检验内部特征融合是否有价值。

## 匹配实验配置

| 配置（configs/h36m/） | spatial_ssi | grad_clip_norm |
|---|---|---:|
| MotionAGFormer-baseline-clip.yaml | False | 1.0 |
| MotionAGFormer-ssi-feature.yaml | True | 1.0 |

两份配置只有 `spatial_ssi` 不同。层数、宽度、帧数、学习率、损失、
batch_size、数据与 seed 约定均相同。原 `MotionAGFormer-base.yaml` 未改动；
旧配置缺省 `spatial_ssi=False`、`grad_clip_norm=0`。

裁剪在每个 batch 的 backward 之后、optimizer.step 之前执行；
NaN/Inf 损失或梯度在参数更新前报错。若将来增加 AMP，需要先 unscale。
梯度指标按 epoch 与原 W&B 指标一同记录：

- `grad/global_norm_mean`：裁剪前范数均值。
- `grad/global_norm_max`：裁剪前范数最大值。
- `grad/clipped_steps_fraction`：触发裁剪的 batch 比例。
- `grad/clip_threshold`：当前阈值。

本轮没有 MSM 特有监控。主要比较最佳 MPJPE，以及该 checkpoint 对应的
P-MPJPE、加速度误差；同时记录实际峰值显存和训练耗时。

## 训练命令

在 sx-zxf 仓库根目录，从头训练；使用默认 `checkpoint` 目录覆盖保存。

```bash
# A：匹配裁剪的基线。已有相同设置的有效记录时可以复用。
python train.py --config configs/h36m/MotionAGFormer-baseline-clip.yaml --seed 0 --use-wandb --wandb-name sx-base-clip1-s0

# B：固定图 SSI 特征融合
python train.py --config configs/h36m/MotionAGFormer-ssi-feature.yaml --seed 0 --use-wandb --wandb-name sx-ssi-feature-s0
```

两条命令依次运行，不并发写同一个 checkpoint 目录。分别记录 W&B run ID。
新实验不加 `--checkpoint` 或 `--resume`，避免加载其他实验的训练状态。
只有中断后继续本次 SSI 实验时，使用原配置并加：

```bash
--checkpoint checkpoint --resume
```

基线与 SSI 的可学习参数键/形状相同，但这不意味着两种配置的 checkpoint
可以混作同一次续训；必须匹配生成 checkpoint 时的配置。
此前 MSM checkpoint 的参数结构不同，不用于本轮实验。

## 云端验证记录

按照用户要求，使用临时内存脚本验证，未新增仓库 test 文件或验证脚本。
环境为 Python 3.12、PyTorch 2.5.1+cpu、timm 0.9.16；仓库原 requirements
未修改。CPU 使用实际 SSM 递推，CUDA 路径仍调用原官方 selective_scan_fn。

已通过：

1. 16 条骨骼边、双向连接、自连接、对称度归一化及非边为零。
2. 图聚合与逐关节公式一致，梯度有限，batch/通道互不串接。
3. SSI 关闭时，空间/时间 MAM 与固定 main 源码的参数和输出完全一致。
4. 时间 MAM 忽略空间开关；完整模型恰好 13 个空间 MAM 启用 SSI。
5. 相同 seed 下 A/B 的全部可学习参数初值相同，state_dict 键相同，
   开启 SSI 的空间 MAM 输出确实改变。
6. 梯度裁剪、关闭裁剪、非有限梯度拒绝均通过。
7. A/B 均使用完整模型、真实姿态损失、train.py 实际 train_one_epoch 函数，
   在 `[1,243,17,3]` 合成数据上完成前向、反向、AdamW 更新和更新后前向。
   训练函数在内存中提取执行，避免加载无关数据集/W&B 服务依赖。

| 项目 | A 基线 | B SSI |
|---|---:|---:|
| 参数量 | 11,342,421 | 11,342,421 |
| 裁剪前梯度范数 | 16.77573776 | 16.77788162 |
| 裁剪后梯度范数 | 0.99999994 | 0.99999988 |
| 更新后的输出形状 | [1,243,17,3] | [1,243,17,3] |

新增可训练参数 **0**。图仍增加计算；按每个 clip 的 13 次稠密图乘法核算，
额外约 `13*243*64*17*17 = 58,428,864 MACs`，不含残差加法。
这不是全模型 MACs，也不是 GPU 性能实测。

当前云端没有 CUDA，未验证 GPU 内核、多 GPU、真实训练峰值显存或 W&B 上传。
合成数据检查不代表 Human3.6M 精度；完整训练仍需在本地数据环境完成。

## 结果待填

| run / commit / seed | 最佳 MPJPE | 同 checkpoint P-MPJPE | 加速度误差 | 峰值显存/耗时 |
|---|---|---|---|---|
| A | 待填 | 待填 | 待填 | 待填 |
| B | 待填 | 待填 | 待填 | 待填 |

先评价 A/B；本轮不自动展开可学习边权或隐藏状态融合。
