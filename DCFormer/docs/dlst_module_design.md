# DLST：深度分层排序 Transformer

## 核心定位

DLST（Depth-Layer Sorting Transformer）放在 AMS 之后、multimodal fusion 之前，用来把 AMS 采集到的 depth-aware joint tokens 组织成显式的骨架前后层次。

一句话：

```text
AMS 负责采集每个关节附近的 RGB/depth 证据，DLST 负责把 depth joint tokens 排成全局一致的前后深度层次。
```

DLST 不是 UDE 式逐关节不确定性去噪，也不是普通 pairwise graph。它的核心是一个可微的 Skeleton Z-buffer：

```text
depth joint tokens -> ordered depth layers -> assignment A -> R=AΩA^T -> depth-biased attention
```

## 设计动机

单目 3D pose 的困难不只是“每个关节的绝对深度不准”，还包括“多个关节在图像平面接近或重叠时，模型难以判断它们的前后层次”。

ASFnet 的 AMS 已经能从 RGB/depth feature map 中采到 pose-relevant depth tokens，但这些 depth tokens 仍然没有被显式整理成全身一致的前后结构。当前诊断结果支持这个判断：

| 诊断项 | 4096 随机验证样本结果 | 含义 |
|---|---:|---|
| K=4 oracle layer `R=AΩA^T` 全 pair ordinal acc | 0.872 | 4 个有序深度层能表达大部分显著前后关系。 |
| K=4 oracle layer ambiguous-pair ordinal acc | 0.859 | 对手腕、肘、膝、踝等易混淆 pair 仍然有效。 |
| 从 AMS `F_d` 线性预测 4 层 exact acc | 0.496 | AMS token 里有深度线索，但精确层次没有显式整理好。 |
| 从 AMS `F_d` 线性预测 adjacent acc | 0.804 | 多数错误落在相邻层，说明层次信号是存在的。 |
| 线性预测 assignment 得到 soft `R` 全 pair acc | 0.749 | 简单探针已经能抽出一部分前后关系。 |
| 当前 coarse depth ambiguous-pair best acc | 0.551 | 逐关节标量深度对易混淆前后关系接近随机。 |
| 原始 `F_d` 距离 vs `|ΔZ|` Spearman | 0.103 | 原始特征空间没有自然按真实深度差组织。 |

对应脚本和结果：

```text
tools/analyze_dlst_motivation.py
dlst_motivation_4096_random_action/summary.json
```

结论不是“DLST 一定涨点”，而是：

```text
AMS 已采到局部深度证据，但缺少把这些证据整理成全局前后层次的显式结构。
```

## 模块接口

输入：

```text
F_d ∈ R^{B×J×C}   AMS 后的 depth joint tokens
```

输出：

```text
F_d' ∈ R^{B×J×C}   depth-ordered joint tokens
R    ∈ R^{B×J×J}   joint relative depth matrix
A    ∈ R^{B×J×K}   joint-to-depth-layer assignment
```

默认设置：

```text
J=17
C=128
K=4
DLST block=1
```

第一版只使用 `x[:, -1]` 这个 AMS depth token，不显式加入 `x_pose`、RGB token、原始 depth map、part token 或 occlusion head。

## 有序深度层

DLST 引入 K 个 learnable depth layer tokens：

```text
L = {l_1, l_2, ..., l_K},  L ∈ R^{K×C}
```

这些层从相机近到远有固定顺序：

```text
Layer 1: foreground
Layer 2: near-middle
Layer 3: far-middle
Layer 4: background
```

它们不是固定人体部位，而是每张图自适应的深度层容器。一帧里 Layer 1 可能主要包含右手和右膝，另一帧里可能包含左手和头部。

代码中用 learnable layer content 加 fixed order embedding 来区分层身份：

```text
layer_tokens = layer_content + layer_order_embed
```

## Step 1：层 token 聚合全局骨架证据

先让 depth layer tokens 通过 cross-attention 从 joint tokens 中聚合当前样本的全局深度证据：

```text
L' = CrossAttn(query=L, key=F_d, value=F_d)
```

这一步的作用是让 4 个层 token 适配当前人体姿态，而不是固定成静态模板。

## Step 2：关节到深度层的软分配

每个关节对 K 个有序深度层做 soft assignment：

```text
A = Softmax(Q(F_d) K(L')^T / sqrt(C))
```

其中：

```text
A ∈ R^{B×J×K}
```

`A_{i,k}` 表示 joint i 属于 depth layer k 的概率。软分配比硬分层更合理，因为有些关节本来就在层边界附近。

## Step 3：固定层间前后关系

定义固定的有序层间矩阵：

```text
Ω_{a,b} = tanh((b - a) / τ)
```

约定 layer 0 最靠近相机，layer K-1 最靠后。因此：

```text
Ω_{a,b} > 0  表示 layer a 在 layer b 前面
Ω_{a,b} = -Ω_{b,a}
Ω_{a,a} = 0
```

默认：

```text
τ = 1.0
```

## Step 4：得到关节相对深度矩阵

由 assignment 和固定层顺序得到：

```text
R = A Ω A^T
```

其中：

```text
R ∈ R^{B×J×J}
R_ij > 0: joint i 更可能在 joint j 前面
R_ij < 0: joint i 更可能在 joint j 后面
R_ij ≈ 0: 两者处于相近深度层
```

这个设计的优点：

- 不需要给每对 joint 单独建 MLP；
- `R` 天然反对称；
- 关系来自全局层排序，减少 pairwise graph 的循环矛盾；
- `A` 和 `R` 都可视化，论文解释清楚。

## Step 5：深度分层注意力

DLST 使用标准 residual Transformer block 的训练骨架，但 attention 不是普通 self-attention，而是 depth-biased attention：

```text
Attn_h = Softmax(Q_h K_h^T / sqrt(d) + α_h R) V_h
```

其中 `α_h` 是每个 head 一个 learnable scalar，默认初始化为 0.1。它不是手工指定 same-layer/front-to-back/back-to-front head，而是让不同 head 自己学习是否使用以及如何使用 `R`。

当前实现不加入：

```text
same_layer bias
skeleton hop bias
manual head type
```

这样主线保持干净：

```text
A -> Ω -> R -> depth-biased attention
```

## Loss：Depth Ordering Loss

DLST 不使用 UDE 的 heteroscedastic uncertainty loss。它只监督 `R` 是否表达真实前后顺序。

从 GT 3D pose 的 Z 坐标构造：

```text
y_ij = sign(Z_j - Z_i)
```

如果 joint i 比 joint j 更靠近相机，则 `y_ij=1`。

只选择深度差明显的 pair：

```text
|Z_i - Z_j| > δ
```

默认：

```text
δ = 0.05 m / 50 mm
```

因为 `R` 被 `Ω` 限制在有界范围内，loss 中使用 scaled logits：

```text
logits_ij = γ R_ij
L_order = mean softplus(-y_ij logits_ij)
```

默认：

```text
γ = 4.0
λ_order = 0.001
```

总损失：

```text
L = L_3D + λ_order L_order
```

## 代码落点

新增文件：

```text
mvn/models/DGLifting_dlst.py
mvn/models/DGPose_dlst.py
experiments/human36m/human36m_single_dlst.yaml
```

在 DLST 版 `DGLifting` 中，原 UDE 位置替换为：

```python
depth_joint_tokens = x[:, -1]
depth_joint_tokens, rel_depth, layer_assign = self.dlst(depth_joint_tokens)
x = torch.cat((x[:, :-1], depth_joint_tokens.unsqueeze(1)), dim=1)
```

返回：

```python
return x, rel_depth, layer_assign
```

训练时 `DepthGuidedPoseDLST` 使用 `DepthOrderingLoss`，而不是 `BNNLoss`。

## 和 UDE 的区别

UDE 的逻辑是：

```text
depth feature 有噪声 -> 逐关节估计 mean/variance -> 用 uncertainty 抑制不可靠深度
```

DLST 的逻辑是：

```text
depth feature 缺少全局前后组织 -> 学有序深度层 -> 推导关节前后关系 -> 引导 depth token 交互
```

论文表述可以写成：

```text
Unlike uncertainty-based enhancement that treats each joint depth feature independently,
DLST models the human body as an ordered stack of depth layers. By softly assigning
joints to these layers and deriving a globally consistent relative depth matrix,
DLST transforms local depth evidence into a structured front-back skeleton layout.
```

## 推荐实验

不要做太多碎消融。建议主表：

```text
AMS only
AMS + RePOSE-style spatial ordering loss
AMS + DLST w/o order loss
AMS + DLST full
PoseAlign
```

补充：

```text
K = 3 / 4 / 5
λ_order = 0.001 / 0.003
```

关键辅助指标：

- `R` 的 pairwise ordinal accuracy；
- ambiguous pair ordinal accuracy；
- layer assignment heatmap；
- relative depth matrix 可视化；
- wrist / elbow / knee / ankle per-joint MPJPE；
- attention head 对 `R` 的使用强度 `α_h`。

## 主要风险

1. **低秩表达过粗**

`R=AΩA^T` 受 `K` 限制，只表达层级关系，不表达任意连续 depth gap。这是设计取舍。DLST 主打排序，不主打 metric depth。

2. **Assignment 塌缩**

如果没有 order loss，所有 joint 可能分到相近层，导致 `R≈0`。所以 `L_order` 是 DLST 的关键监督。

3. **收益不是必然**

诊断支持“分层中间结构合理”和“当前 AMS 表达没有显式整理好”，但最终 MPJPE 是否超过 PoseAlign 仍取决于训练和数据分布。目标应设为小幅稳定收益，而不是大幅提升。
