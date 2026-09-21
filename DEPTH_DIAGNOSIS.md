# 第一阶段：Baseline 深度诊断

起点：`main` 的 `a2346dd89a1d7edf696475f61631d5145a8d45e9`。
新分支：`depth-diagnosis`，不继承旧诊断分支。

## 目标与边界

先确定 baseline 的 Z 误差属于幅度、前后排序、骨骼结构还是时间变化问题。
此阶段只评估已有 checkpoint，不训练新模型。

- 诊断初版不修改模型、loss 和训练；后续按用户要求在 `train.py` 加入梯度裁剪，并在 large 配置中启用。模型结构和 loss 公式不变，参数增量仍为 0。
- 新入口 `diagnose_depth.py`，默认推理 batch=1，保留配置中的 flip 测试增强。
- 恢复用户提供的 `data/reader/*.py` 和 `data/const.py`；`h36m(1).py` 放回 `data/reader/h36m.py`。
- `.gitignore` 只允许 data 下这些源码进入版本控制，数据集、切片、权重和诊断输出不提交。
- 当前只支持 Human3.6M、17 关节、root_rel=True、add_velocity=False。

输入第三维是检测置信度，不是深度。当前任务中 Z 指关节相对根关节的深度，不能由这些指标推断人体到相机的绝对距离。
骨盆误差为零来自根对齐，并不代表模型预测准确。
关节名称沿用用户提供的 const.py；图和表同时给出关节 ID。左右命名应以实际预处理定义核对，勿直接换用其他仓库名称。

## 本地运行

在现有科研环境、仓库根目录运行，无需重新训练。先切换到此分支；本地原有 data 下的同名未追踪源码可能阻止切换，先将这些源码备份到仓库外，再切换，数据集无需移动。

```bash
git fetch origin
git switch --track origin/depth-diagnosis
python tools/smoke_depth.py --config configs/h36m/MotionAGFormer-large.yaml
python diagnose_depth.py \
  --config configs/h36m/MotionAGFormer-large.yaml \
  --checkpoint checkpoint/best_epoch.pth.tr \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir depth_outputs/baseline
```

将 checkpoint 路径替换为自己的文件，配置必须与该权重完全一致。
`--checkpoint` 在新入口中接收**完整文件路径**，不同于 train.py 的目录参数。
checkpoint 使用 strict=True 加载，兼容 DataParallel 的 module. 前缀，不会忽略权重不匹配。

数据目录不同可传 `--data-root /path/to/motion3d`。默认 `--input-source metadata` 只需要该目录中的元数据 pkl；选择 `--input-source slices` 时还需要 `H36M-243/test/` 切片。
程序逐片校验标签与元数据索引，不匹配就停止，避免静默错位。检查切片配置、顺序和短序列重采样种子。
默认 seed=0；沿用 MotionDataset3D 中已有的 NumPy seed 行为。

原 baseline 的评估仍使用原命令：

```bash
python train.py --eval-only \
  --config configs/h36m/MotionAGFormer-large.yaml \
  --checkpoint checkpoint --checkpoint-file best_epoch.pth.tr
```

## 输出与统计口径

| 文件 | 内容 |
|---|---|
| depth_summary.json | 逐轴 MAE、XY、MPJPE、P-MPJPE、Z 平方误差占比、ΔZ、排序、时间指标和运行信息 |
| joint_z_error.csv | 每个关节的 Z MAE，包含 ID |
| bone_delta_z_error.csv | 16 条骨骼的相对深度差误差 |
| depth_ordering.csv | 136 个关节对的排序正确率、覆盖率、有效帧数及是否为骨骼边 |
| action_z_error.csv | 每个动作的空间和时间指标及有效帧/时间窗口数 |

JSON 提供 `frame_weighted`（按帧平均）和 `action_macro`（先每动作平均，再对有效动作等权平均）。
MPJPE/P-MPJPE 对照原评估时看 action_macro。每个动作内部先平均同一帧的重复预测误差，再平均帧；不是先平均坐标再算误差。
完美预测帧仍计入；这与原 train.py 的 `e1_all > 0` 筛选存在极端零误差情况的差别。
沿用原评估的三个排除序列，不做 Procrustes 对齐后再计算 Z；对齐只用于 P-MPJPE。

- 逐轴 MAE 为 `mean(abs(pred_axis - gt_axis))`，包括对齐后为零的根关节，便于与 17 关节 MPJPE 对应。
- XY error 是平面欧氏距离。XY error + Z error 不等于 MPJPE；判断某个轴是否更难先比较逐轴 MAE。
- Z 平方误差占比为 `sum(ez²) / sum(ex²+ey²+ez²)`，不是 MPJPE 的线性贡献占比。全零误差时输出 null。
- 骨骼 ΔZ 取 child - parent；关节对取 j - i。排序仅保留 GT 深度差绝对值严格大于阈值的样本（默认 10 mm）。预测相等记错误；没有有效样本时输出 null。
- 默认统计 all-pairs 排序，骨骼边可从 CSV 的 is_bone 列单独分析。阈值可以通过 `--ordering-threshold-mm` 调整。
- 排序整体正确率按有效 pair-frame 加权；action_macro 对各动作正确率再等权平均。CSV 中每对关节给出按帧统计。
- 时间差分只用同一 source、同一 action、原始索引真正连续的两帧/三帧。不跨片段，不对重复补帧计算时间导数。
- 速度单位 mm/frame，加速度单位 mm/frame²；没有乘采样帧率。时间指标使用各自有效窗口计数。

**新加速度指标不声称与原 train.py 加速度数值一致。** 原代码按位置帧计数归一化加速度，并用 mm/s² 文案打印未除时间间隔的二阶差分。此处报告按有效连续窗口归一化的加速度，原评估代码保持不变。比较历史结果时仍使用原入口。

## 从诊断到最小实验

| 证据 | 候选假设与下一步 |
|---|---|
| 排序好、ΔZ 幅度差 | 用骨骼相对深度幅度辅助监督做最小对照 |
| 有效大深度差样本的排序仍差 | 考虑轻量深度关系表示，先排除输入遮挡/检测误差因素 |
| 末端 Z 大、局部骨骼误差逐级累积 | 检查骨骼链误差相关性，再考虑结构约束 |
| 时间误差集中于运动片段 | 先按真实运动量分组验证，再考虑深度时间建模 |

这些是候选解释，单个指标不足以证明成因；动态动作名称也不等价于真实运动量。
本版先提供基础报告，深度差分桶、逐帧运动量/置信度分组和案例可视化留给看到结果之后。
单独增加一个 ΔZ loss 或小 Z head 只是验证实验，不能直接当作充分创新。

## 验证

```bash
python -m unittest discover -s tests -v
python tools/smoke_depth.py
```

`tools/cpu_scan_reference.py` 只由显式 CPU smoke 入口加载，在该进程中提供真实递推参考。不会被训练或诊断入口导入，不修改任何模型参数/结构或 GPU scan。
参考实现只支持当前 MAM 实际使用的实数、非分组 B/C 形式，不用于速度测量，也不代表已验证 GPU 数值一致性。
完整 large 配置使用 batch=1、T=243；形状和参数报告写到 `depth_outputs/cpu_smoke.json`。
真实 checkpoint 的指标必须在有数据和 CUDA 的本地环境执行后才可得出。

### 本次实际验证结果

- CPU 实例化及完整 T=243 前向：通过；输入输出均为 `[1,243,17,3]`，输入/输出均无 NaN/Inf。
- 总参数 11,342,421；可训练参数 11,342,421；相对 main 增量 0。
- 7 项诊断单元测试通过：平移消除、逐轴数值、重复索引/重叠、动作权重、排序阈值、时间连续性及不完整输入保护。
- 合成数据上，与原 loss.pose3d 的 MPJPE/P-MPJPE 一致；反归一化与原 DataReaderH36M.denormalize 逐元素一致。
- CPU scan 在已知闭式递推样例上通过校验。
- 测试环境为 Python 3.12、PyTorch 2.5.1+cpu、timm 0.9.16、einops 0.8.2。仓库锁定的 timm 0.6.11 在 Python 3.12 下存在 dataclass 导入错误，因此只在临时测试环境使用 0.9.16；requirements.txt 未修改。
- 尚未运行真实 H36M checkpoint、GPU kernel 或测量 GPU 显存/速度，不能据此报告精度提升或 GPU 数值一致性。

## 后续训练稳定性更新：梯度裁剪

按用户要求，在 H36M 的 train.py 中加入全局 L2 梯度范数裁剪：
反向传播 → clip_grad_norm_ → optimizer.step。
large 配置新增 `grad_clip_norm: 1.0`；其他 H36M 配置未写该项时也默认 1.0，设置为 0 可关闭。
阈值是所有参数梯度合并后的范数上限，不是逐元素截断，也不是 loss 上限。

开启裁剪时，NaN/Inf 梯度范数会在参数更新前报错停止，避免把非有限梯度写入参数。
裁剪无法修复已经损坏的权重或优化器状态。恢复训练应使用异常发生之前的有效 checkpoint。
当前运行进程需重启才能加载新代码；对照实验应记录裁剪阈值并保持一致。

控制台每轮打印裁剪前范数的逐步均值和裁剪比例。
开启 wandb 时新增：
- `train/grad_norm_before_clip`：每轮裁剪前梯度范数的逐步均值；
- `train/grad_clip_fraction`：该轮触发裁剪的训练步比例（0～1）；
- `train/grad_clip_max_norm`：裁剪阈值。

本次仅修改 train.py、large YAML 和说明文档。按用户明确要求未执行云端测试；
上方的 CPU/单元测试结果属于此前诊断初版，不代表此次训练修改已测试。

## 切片标签不匹配修复

诊断默认使用 `--input-source metadata`：直接从同一元数据中的 joint_2d/confidence 读取检测输入，
并使用完全相同的 frame_clips 索引提取真值、动作、相机尺度和来源。仍使用原读取器的归一化与划片逻辑。
划片的 NumPy seed 固定为 0，与旧 MotionDataset3D 的初始化行为一致，记录在 summary 中。

原因：原短序列 resample 使用 NumPy 随机数，预处理没有固定 NumPy seed，
且保存的切片仅有 data_input/data_label，没有 frame IDs。重建索引不保证重现历史切片。
这是一种可能导致标签校验失败的机制，不表示已通过用户实际数据确认该具体切片的原因。
也不应直接忽略标签不一致后继续计算指标。

元数据模式不改写已有切片、不使用 GT 构造 2D 输入，也不需要重新训练。
它保证本次诊断输入与标签一致，但短序列上下文/采样可能与历史切片不同，
因此不能承诺与旧 train.py 评估数值逐位一致。比较实验应统一使用同一诊断模式与划片设置。
需要核查旧切片时使用 `--input-source slices`，仍严格校验，并打印 source 和最大标签差异。
此修复尚未用用户真实数据和 checkpoint 验证。

## 逐帧相机尺寸修复

metadata 输入按每帧 camera_name 归一化，但先前校验及反归一化沿用 get_hw() 返回的片段首帧尺寸，
两者处理不一致。1000 与 1002 的高度差对应归一化 Y 偏移 0.002。
现将输入、标签校验、预测反归一化统一为逐帧相机尺寸，保留严格标签校验，
不扩大容差。结果 metadata 记录 camera_normalization=per-frame。
高度导致的平移本身会被根对齐消除，不能据此断言历史 MPJPE 已受影响。

使用四种相机的混合高度合成数据复现旧校验 0.002 差值；
修复后的归一化数值和逆变换检查通过，脚本语法检查通过。
未运行模型/GPU 测试，尚无用户真实元数据验证。实际片段是否混合尺寸仍需用户数据确认。
