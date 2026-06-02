# DLST 阶段性汇报

更新日期：2026-05-08  
当前状态：DLST 训练仍在进行中，已有 7 个 epoch 的阶段性结果。本文档用于梳理测试动机、设计思路、论文叙事、当前实验现象和后续验证计划。

## 一句话结论

DLST（Depth-Layer Sorting Transformer）的目标不是替代 AMS 采样，也不是做 UDE 式不确定性去噪，而是把 AMS 输出的高维 depth-aware joint tokens 组织成显式的全身前后层次。

当前训练日志显示：DLST 的内部排序分支已经稳定学到 GT 前后关系，验证集 `dlst_rel_sign_acc` 已达到约 `0.95`，assignment 没有塌缩；同时 MPJPE 在第 7 个 epoch 已降到 `39.0mm`，训练未完成但趋势积极。

## 背景问题

ASFnet 的 AMS 模块可以围绕 2D joint 从 RGB/depth feature map 中采集局部证据，输出每个关节的 depth-aware token：

```text
F_d ∈ R^{B×17×128}
```

这个 token 是高维特征，不是标量深度。它可能包含深度、遮挡、边缘、局部图像上下文等信息，但原始 feature 空间并没有显式保证：

```text
哪个关节在前，哪个关节在后
```

单目 3D pose 的关键困难之一不是单个 joint 的绝对深度，而是肢体交叠、自遮挡、左右肢体接近时的相对前后顺序。UDE 主要解决深度特征噪声问题，DLST 关注的是另一类问题：深度证据已经被 AMS 采到，但还缺少全局一致的骨架前后组织。

## 动机测试

现有动机诊断脚本：

```bash
tools/analyze_dlst_motivation.py
```

4096 个随机验证样本上的关键结果：

| 诊断项 | 结果 | 含义 |
|---|---:|---|
| K=4 oracle layer `R=AΩA^T` 全 pair ordinal acc | 0.872 | 4 个有序深度层可以覆盖大部分显著前后关系。 |
| K=4 oracle layer ambiguous-pair acc | 0.859 | 对手腕、肘、膝、踝等易混淆 pair 也有效。 |
| AMS `F_d` 线性预测 4 层 exact acc | 0.496 | AMS token 里有深度线索，但层次没有显式整理。 |
| AMS `F_d` 线性预测 adjacent acc | 0.804 | 多数错误在相邻层，说明层次信号存在。 |
| 线性 probe soft `R` 全 pair acc | 0.749 | 简单 probe 已能抽出部分前后关系。 |
| 当前 coarse depth ambiguous-pair best acc | 0.551 | 标量 coarse depth 对易混淆前后关系接近随机。 |
| 原始 `F_d` 距离 vs `|ΔZ|` Spearman | 0.103 | 原始 feature 空间没有自然按真实深度差组织。 |

这组测试支持的结论是：

```text
AMS 已采到局部深度证据，但需要一个结构化模块把它整理成全局前后层次。
```

## 设计思路

DLST 放在 AMS 之后、multimodal fusion 之前：

```text
AMS depth token -> DLST -> updated depth token -> RGB/depth/pose fusion -> 3D pose head
```

核心流程：

```text
F_d -> ordered depth layers -> assignment A -> R=AΩA^T -> depth-biased attention
```

### 1. 输入

```text
F_d ∈ R^{B×J×C}, J=17, C=128
```

这里的 `F_d` 是 AMS 输出的高维 depth-aware joint token，不是每个关节一个标量深度。

### 2. 有序深度层

DLST 引入 `K=4` 个 learnable depth layer tokens：

```text
Layer 0: foreground
Layer 1: near-middle
Layer 2: far-middle
Layer 3: background
```

这些 layer 不是固定人体部位，而是每一帧自适应的深度容器。

### 3. 关节到层的软分配

每个 joint token 对 4 个 depth layer 做 soft assignment：

```text
A = Softmax(Q(F_d) K(L)^T / sqrt(C))
A ∈ R^{B×J×K}
```

`A_{i,k}` 表示第 `i` 个关节属于第 `k` 个深度层的概率。

### 4. 固定层间顺序

定义固定反对称矩阵：

```text
Ω_{a,b} = tanh((b-a)/τ)
```

因此：

```text
Ω_{a,b} = -Ω_{b,a}
Ω_{a,a} = 0
```

### 5. 推导关节相对深度矩阵

```text
R = AΩA^T
R ∈ R^{B×J×J}
```

含义：

```text
R_ij > 0: joint i 更可能在 joint j 前面
R_ij < 0: joint i 更可能在 joint j 后面
R_ij ≈ 0: 两者深度接近
```

因为 `Ω` 是反对称矩阵，所以 `R=AΩA^T` 也天然反对称。这比直接预测任意 pairwise relation 更有结构约束。

### 6. 深度分层注意力

在 depth token 内部 attention 里加入 `R`：

```text
Attn_h = Softmax(Q_h K_h^T / sqrt(d) + α_h R) V_h
```

`α_h` 是每个 attention head 的可学习 gate。这样模型可以自己决定是否以及如何使用前后关系。

## 监督方式

DLST 不直接回归每个 joint 的绝对 z，而是监督 `R` 的符号是否符合 GT 前后顺序。

从 GT 3D pose 的 z 坐标构造：

```text
y_ij = sign(Z_j - Z_i)
```

只选择真实深度差足够明显的 joint pair：

```text
|Z_i - Z_j| > δ
```

当前默认：

```text
δ = 0.05m = 50mm
```

这个阈值只用于 GT z 坐标筛 pair，不用于 AMS feature。抽样统计显示，在 Human3.6M 验证集上 `50mm` 会保留约 `84%` 的非对角 joint pairs，属于合理默认值：过滤掉近似同层、不稳定的 pair，同时保留大部分有意义的前后关系。

排序损失：

```text
logits_ij = γ R_ij
L_order = mean softplus(-y_ij logits_ij)
```

当前配置：

```text
γ = 4.0
λ_order = 0.001
```

总损失：

```text
L = L_3D + λ_order L_order
```

## 和 UDE 的区别

UDE 的逻辑：

```text
depth feature 有噪声 -> 估计 mean/variance -> 用 uncertainty 抑制不可靠深度特征
```

DLST 的逻辑：

```text
depth feature 缺少全局前后组织 -> 学有序深度层 -> 推导关节前后关系 -> 引导 depth token 交互
```

所以 DLST 不是 UDE 的简单替代，而是从另一个角度利用 AMS depth token：UDE 强调可靠性建模，DLST 强调全身前后结构建模。

## 论文叙事建议

可以把 DLST 叙述为一个可微的 Skeleton Z-buffer：

```text
Instead of treating each depth-aware joint token independently, DLST organizes
the human body into a small set of ordered depth layers. By softly assigning
joints to these layers and deriving a globally consistent relative-depth matrix,
DLST converts local AMS depth evidence into an explicit front-back skeleton
layout, which is then injected into depth-token attention.
```

中文表述：

```text
DLST 将人体建模为一个有序深度层栈。它不是直接预测每个关节的绝对深度，而是通过软分配把关节放入少量有序层中，再由层间固定前后关系推导出全局一致的关节相对深度矩阵，从而把 AMS 的局部深度证据转换为显式骨架前后布局。
```

## 当前实现

代码落点：

```text
mvn/models/DGLifting_dlst.py
mvn/models/DGPose_dlst.py
experiments/human36m/human36m_single_dlst.yaml
train.py
mvn/models/loss.py
```

核心替换位置：

```python
depth_joint_tokens = x[:, -1]
depth_joint_tokens, rel_depth, layer_assign = self.dlst(depth_joint_tokens)
x = torch.cat((x[:, :-1], depth_joint_tokens.unsqueeze(1)), dim=1)
```

训练返回：

```text
keypoints_3d_pred, rel_depth, layer_assign
```

`DepthGuidedPoseDLST` 使用 `DepthOrderingLoss`，并在训练/验证时额外记录 DLST 诊断指标。

## 新增指标

### order_loss

DLST 相对深度排序辅助损失。下降表示 `R` 的符号和置信度更贴近 GT depth order。

### dlst_rel_sign_acc

对所有满足 `|ΔZ| > 50mm` 的 joint pair，比较：

```text
sign(R_ij) vs sign(Z_j - Z_i)
```

它衡量 DLST 的中间关系矩阵是否学到了真实前后顺序。它是机制诊断指标，不是最终性能指标。

### dlst_assign_entropy

衡量 joint-to-layer assignment 的软硬程度。过高说明分配太平均，难以形成明确层次；过低可能说明 assignment 过硬或塌缩。

### dlst_layer_usage_min / max

衡量 K 个 depth layer 是否都被使用。若 `min≈0` 或 `max≈1`，说明可能存在 layer collapse。

## 当前实验效果

当前 DLST 日志：

```text
logs/ConPose@07.05.2026-10:46:00/out.txt
```

训练仍未完成，已有 7 个 epoch：

| Epoch | Val P1 MPJPE | Val P2 | Val order_loss | Val rel sign acc | Val entropy | Usage min | Usage max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46.3 | 38.1 | 0.1957 | 0.9356 | 0.3785 | 0.2055 | 0.2979 |
| 2 | 41.5 | 34.7 | 0.1800 | 0.9426 | 0.3667 | 0.2005 | 0.3016 |
| 3 | 40.4 | 33.8 | 0.1699 | 0.9487 | 0.3533 | 0.2063 | 0.2974 |
| 4 | 40.4 | 33.7 | 0.1676 | 0.9498 | 0.3572 | 0.1918 | 0.3010 |
| 5 | 40.5 | 33.7 | 0.1644 | 0.9509 | 0.3608 | 0.2082 | 0.2954 |
| 6 | 40.1 | 33.4 | 0.1623 | 0.9530 | 0.3398 | 0.2100 | 0.2926 |
| 7 | 39.0 | 32.4 | 0.1609 | 0.9528 | 0.3589 | 0.2156 | 0.2868 |

阶段性观察：

- `dlst_rel_sign_acc` 从第 1 个 epoch 就达到 `0.9356`，第 7 个 epoch 约 `0.9528`，说明 `R=AΩA^T` 在验证集上稳定表达 GT 前后顺序。
- `order_loss` 从 `0.1957` 降到 `0.1609`，排序分支仍在变好。
- `assign_entropy` 稳定在 `0.34-0.38` 附近，分层明确但没有硬塌缩。
- `layer_usage_min/max` 维持在健康范围，4 个 depth layer 都被使用。
- MPJPE 从 `46.3mm` 快速下降到 `39.0mm`，第 7 个 epoch 已达到当前仓库中 ASFnet 复现日志的较强区间，训练仍未完成。

## 和已有日志的粗略对比

注意：以下日志的训练设置并非完全统一，正式主表需要统一配置重跑。因此这里只作为阶段性参考。

| 模型/日志 | 参数量 | 当前/最佳 P1 | 当前/最佳 P2 | 备注 |
|---|---:|---:|---:|---|
| ASFnet/UDE local log `20.04` | 20.85M | best 39.0 | best 32.0 | 30 epoch 已完成；用户反馈本地目标线约 38.8。 |
| PoseAlign log `28.04` | 20.28M | best 38.9 | best 32.4 | 21 epoch 日志。 |
| DLST current `07.05` | 20.21M | epoch 7: 39.0 | epoch 7: 32.4 | 未训练完；排序诊断很强。 |

更关键的早期趋势是：DLST 在第 7 个 epoch 已达到 `39.0mm`，而历史 ASFnet/UDE 日志第 7 个 epoch 为 `40.2mm`，PoseAlign 第 7 个 epoch 为 `39.6mm`。这不是最终结论，但说明 DLST 当前收敛趋势积极。

## 当前能支持的结论

可以比较有把握地说：

```text
DLST 的中间结构学到了有效的前后排序信号。
```

证据：

- 验证集 `dlst_rel_sign_acc≈0.95`；
- `order_loss` 稳定下降；
- assignment 没有塌缩；
- depth layer usage 健康；
- MPJPE 早期下降速度正常且第 7 个 epoch 已到 `39.0mm`。

还不能完全说：

```text
DLST 最终一定优于 UDE / PoseAlign。
```

原因：

- 当前训练未完成；
- `dlst_rel_sign_acc` 是机制指标，不等价于 MPJPE；
- 排序分支有直接监督，高 sign acc 证明排序可学，但还需证明排序信息确实改善最终 pose。

## 后续必须补的验证

### 1. 完整训练

当前最重要的是跑完 30 epoch，观察是否达到或超过本地 ASFnet 复现线：

```text
目标线：P1 <= 38.8mm
```

### 2. w/o order loss

设置：

```yaml
lambda_order: 0.0
```

目的：证明提升不是来自多加参数，而是来自 depth ordering 监督。

### 3. rel_depth 反事实 eval

在同一个 DLST checkpoint 下做 eval-time 干预：

```text
normal
rel_depth = 0
rel_depth shuffle
depth_gate = 0
```

如果 normal 明显优于这些反事实版本，说明 `R=AΩA^T` 确实被 pose 分支使用。

### 4. 同参数普通 attention 对照

保留近似参数量，但移除 depth-biased attention，只用普通 self-attention。  
目的：回答提升来自“更多参数”还是“深度层排序结构”。

### 5. per-joint MPJPE

重点统计：

```text
wrist / elbow / knee / ankle
```

DLST 主打前后层次和肢体交叠，因此末端关节应该比 torso 更能体现收益。

### 6. per-action MPJPE

重点看自遮挡较多动作：

```text
SittingDown, Sitting, Greeting, TakingPhoto, WalkingDog
```

如果收益集中在这些动作，更符合 DLST 的论文叙事。

### 7. 多阈值排序指标

当前 `dlst_rel_sign_acc` 默认使用 `50mm`。建议补充：

```text
sign_acc@30mm
sign_acc@50mm
sign_acc@100mm
```

如果不同阈值下都高，说明排序证据更稳。

## 可运行脚本

训练前动机诊断：

```bash
python3 tools/analyze_dlst_motivation.py \
  --config experiments/human36m/human36m_single.yaml \
  --checkpoint <asfnet_checkpoint> \
  --num_samples 4096 \
  --random_subset \
  --output_dir dlst_motivation_4096_random_action
```

训练中信号诊断：

```bash
python3 tools/analyze_dlst_training_signal.py \
  --log logs/ConPose@07.05.2026-10:46:00/out.txt \
  --output_dir dlst_training_signal_current \
  --target_mpjpe 38.8
```

正式训练命令：

```bash
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=2345 \
  train.py \
  --config experiments/human36m/human36m_single_dlst.yaml \
  --logdir ./logs
```

## 风险和边界

1. `R=AΩA^T` 是低秩/分层表达，适合排序，不适合精确 metric depth gap。
2. `dlst_rel_sign_acc` 高不等于 MPJPE 一定低，需要最终主表验证。
3. 排序监督过强可能让模型过度优化中间任务，因此需要观察 `λ_order`，当前 `0.001` 看起来稳定。
4. 当前对比日志训练设置不完全统一，正式汇报结果需要统一配置和 seed 后重跑。

## 给老师汇报时的推荐表述

可以这样说：

```text
我们发现 AMS 后的 depth-aware token 中确实包含深度层次信息，但这种信息没有被显式组织。DLST 的思路是把人体关节软分配到少量有序深度层中，再由层间固定顺序推导全局一致的相对深度矩阵，并把这个矩阵作为 attention bias 引导 depth token 交互。

目前训练尚未完成，但中间诊断非常稳定：验证集 pairwise depth-order sign accuracy 达到约 95%，assignment 没有塌缩，4 个 depth layer 都被使用。MPJPE 在第 7 个 epoch 已降到 39.0mm，接近本地 ASFnet 复现线。下一步会补 w/o order loss、rel_depth zero/shuffle、per-joint 和 per-action 结果，验证该排序结构是否确实带来最终 3D pose 收益。
```
