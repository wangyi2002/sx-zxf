# Experiment 1: skeleton-relative delta-Z supervision

来源：`wangyi2002/sx-zxf` 的 `depth-diagnosis`，commit
`e1d208b9aeb02ff1be6edd15445e8356de7a420a`。
实验分支：`exp/relative-depth-loss`。创建前已获用户确认。

## 假设与唯一训练变量

检验在原训练目标上额外监督骨骼的连续深度差，能否改善 Bone ΔZ MAE、Z MAE、MPJPE。
这是一项监督方式验证，不是结构创新，也不是绝对相机深度估计。

`L_total = L_original + 0.1 * mean(abs((pred_z_i - pred_z_j) - (gt_z_i - gt_z_j)))`

均值覆盖 batch、时间和全部 16 条骨骼；使用 L1，无排序阈值或 all-pair 损失。
模型、输入、增强、输出和原损失均保持不变；原有速度损失（权重 20）、scale loss
（权重 0.5）和梯度裁剪（全局 L2，1.0）保留，没有增加其他约束。

## 实现与坐标

- 函数：`loss/pose3d.py::relative_depth_loss`，由 `train.py::train_one_epoch` 调用。
- `data/const.py::H36M_BONES` 从原 `utils/depth_diagnostics.py::BONES` 原样迁移，
  诊断仍以 `BONES` 名称导入。16 条边的次序、端点不变，每条边有 joint 标签注释。
  已核对原 limb loss 骨骼列表及 GCN 邻接的无向边集合；不修改 GCN 的有向邻接。
- 训练 `pred, y` 为 `[B,T,17,3]`；读取预处理的归一化标签，
  `train_one_epoch` 对 GT 做逐帧根相对化。prediction 是原 pose loss 接收的原始模型输出。
- H36M reader 对 XY 使用 `2 * coord / width - [1, height/width]`，
  对 Z 使用 `2 * z / width`。新损失直接使用原 pose loss 的 `pred, y`，
  不乘 `2.5d_factor`、不反归一化为 mm、不 detach。
- ΔZ 张量为 `[B,T,16]`，使用 tensor 索引，无 batch/frame Python 循环。
  训练损失的归一化量纲与诊断的毫米数值不同，不能直接比较两者数值大小。
- 无可学习参数、无新增模型 buffer，不改变 checkpoint 的 model state_dict。

## 配置与日志

新增独立配置 `configs/h36m/MotionAGFormer-relative-depth.yaml`，完整复制确认版本的
`MotionAGFormer-base.yaml`，仅增加：

```yaml
use_relative_depth_loss: true
relative_depth_weight: 0.1
```

原 base 配置不改。batch_size=4，epochs=60，T=243，seed 使用命令行的 0。
`false` 或旧配置缺少开关时不调用辅助损失，也不对原 total loss 添加零值张量。
不支持根据这一轮结果直接认定最优权重；暂不实现权重搜索。

保留全部原 wandb 指标（包括 `train/total`、`train/loss_3d_pose`），新增：

- `train/loss_original`：所有原损失加权之和。
- `train/loss_relative_depth`：未乘 λ 的 ΔZ L1；关闭时为 0（未计算）。
- `train/use_relative_depth_loss`：实际启用状态，0/1。
- `train/relative_depth_weight`：配置权重。

控制台每轮也输出上述损失及开关、权重。原 evaluation 与保存 checkpoint 的实现不变。

## 本地执行

在仓库根目录、已配置好的训练环境中运行。第一次切换实验分支：

```bash
git fetch origin
git switch --track origin/exp/relative-depth-loss
```

如本地已有同名分支，使用 `git switch exp/relative-depth-loss` 后 `git pull --ff-only`。

从头训练，使用独立目录，保留 baseline 权重与诊断结果。不要传 baseline checkpoint
或 `--resume`；微调会改变实验条件。训练命令：

```bash
python train.py \
  --config configs/h36m/MotionAGFormer-relative-depth.yaml \
  --new-checkpoint checkpoint/relative_depth_l01_seed0 \
  --seed 0 \
  --use-wandb \
  --wandb-name relative-depth-l01-seed0
```

训练完成，用相同的诊断脚本和 metadata 模式：

```bash
python diagnose_depth.py \
  --config configs/h36m/MotionAGFormer-relative-depth.yaml \
  --checkpoint checkpoint/relative_depth_l01_seed0/best_epoch.pth.tr \
  --input-source metadata \
  --batch-size 1 \
  --num-workers 0 \
  --seed 0 \
  --output-dir depth_outputs/relative_depth_l01_seed0
```

请比较新旧 `depth_summary.json` 的 **frame_weighted** 字段；本次基准为：

| 指标 | Baseline |
|---|---:|
| MPJPE | 39.06484 mm |
| P-MPJPE | 33.10445 mm |
| Z MAE | 30.61430 mm |
| Bone ΔZ MAE | 24.55437 mm |
| Ordering accuracy | 95.25747% |

同时保留 action_macro、X/Y MAE、关节/骨骼/动作分项和原 acceleration 指标。
不要将 train.py 的动作宏平均直接与上表的帧加权结果比较。
诊断时间差分单位为 mm/frame 和 mm/frame²，统计口径与旧训练脚本有区别。

## 验证记录与限制

- 12 项 CPU 测试通过：原 7 项诊断回归测试 + 5 项新增测试。
- `[4,243,17,3]` 合成训练输入，ΔZ 为 `[4,243,16]`；相等输入损失为 0；
  单个叶关节已知误差数值正确；梯度有限、非零且只作用于 Z；平移不改变损失。
- 实际 `train_one_epoch` 函数体在小型预测器上执行，与独立原损失表达式对照；
  开启、关闭和缺少开关三种路径均检查。关闭时验证辅助函数调用次数为零，
  total、梯度、参数更新及 AdamW 状态与原流程逐位一致。开启时核对权重和坐标。
- 完整模型使用已有的显式 CPU scan reference 做前向，输入/输出均为
  `[1,243,17,3]` 且有限；生产模型文件和 CUDA scan 路径不修改。
- 参数量 11,342,421，新增参数 0。模型源码与来源 commit 保持一致。
- 没有本地 H36M 数据与用户 checkpoint，未运行真实数据训练或 CUDA 内核测试。
  这些检查证明代码接入与数值性质，不代表实验有效；最终指标须由用户本地训练获得。

复查命令：

```bash
python -m unittest discover -s tests -v
```

解释结果时，Bone ΔZ、Z MAE、MPJPE 同时下降将支持本假设；只改善部分指标需检查
X/Y 权衡和动作分布。单个 seed、单个 λ 的成败不能证明某种 representation 必然必要，
也不能彻底否定监督方案。若收益很小，后续需要重复实验评估波动。
