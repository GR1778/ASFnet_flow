# DKDL：动态运动学深度布局模块高级实现方案

## 设计判断

OARDG 第三轮出现负收益并不意外。它的问题不是“相对深度布局方向错了”，而是实现上把太多未校准信号同时注入：

- raw depth delta；
- 2D near gate；
- depth contrast gate；
- 无监督 `occ_pred`；
- relation bias；
- occlusion bias；
- `1 + occ_pred.mean` 放大 residual update。

这些部件单独看都有动机，但它们没有被统一监督到一个稳定对象上。训练早期 `z_anchor`、`occ_gate`、`rel_pred` 都不可靠时，attention bias 和 residual update 会直接扰乱 AMS 已经学到的有效 `F_d`。

DKDL 应该反过来：只构造一个核心对象，并让所有计算服务于它。

```text
核心对象：Sparse Directed Kinematic Depth Graph
节点：joint depth token
边：有方向的相对深度布局 R_ij
边置信度：c_ij，只控制消息强弱，不解释为遮挡
```

## 模块接口

```text
DKDL(F_d, P) -> F_d_layout, R, C
```

输入：

```text
F_d ∈ R^{B×J×C}   AMS 后深度 token
P   ∈ R^{B×J×2}   归一化 2D joint 坐标
```

不输入 RGB，不输入 raw depth map。AMS 已经做过多模态采样对齐；DKDL 只负责深度布局转换。

输出：

```text
F_d_layout ∈ R^{B×J×C}   layout-aware depth token
R          ∈ R^{B×J×J}   directed relative depth layout
C          ∈ R^{B×J×J}   relation confidence / message mask
```

## 结构

### 1. Depth Residual Canonicalization

去掉 frame-level 共享成分，保留关节间 residual：

```text
F_bar = mean_j(F_d)
X_i = LN(F_d_i - F_bar + XYEmbed(P_i - P_root) + JointEmbed_i)
```

原因：我们的诊断显示 `F_d` 的 body-mean energy fraction 约 `0.544`，说明全局成分偏强。

### 2. Sparse Relation Proposal

不要全 136 对 joint pair 全部同等处理。构造稀疏边集合：

```text
E = E_skeleton ∪ E_symmetric ∪ E_ambiguous ∪ E_close
```

其中：

- `E_skeleton`：人体骨架边；
- `E_symmetric`：左右对称肢体边，如 left wrist/right wrist；
- `E_ambiguous`：手腕-躯干、肘-躯干、膝/踝等容易出现前后歧义的 pair；
- `E_close`：每帧 2D 距离最近的 top-k 非骨架 pair。

这个设计比 OARDG 的 `near_gate * contrast_gate` 稳定，因为 edge 是否参与由结构和 2D 几何决定，不依赖未校准 raw depth contrast。

### 3. Anti-Symmetric Relative Depth Field

直接利用我们验证过最强的信号 `F_i - F_j`：

```text
q_ij = [X_i - X_j, X_i * X_j, XY_i - XY_j, ||XY_i - XY_j||]
a_ij = MLP_edge(q_ij)
R_ij = tanh(a_ij - a_ji)
R_ji = -R_ij
R_ii = 0
```

重点：不要先做 anchor bin 再推 relation。`F_i - F_j -> ΔZ` 的线性 probe 已经很强，第一版应当直接利用这个差分信号。

### 4. Relation Confidence

置信度只控制消息强度，不解释成遮挡：

```text
C_ij = sigmoid(MLP_conf([|X_i-X_j|, ||XY_i-XY_j||, edge_type_ij]))
```

如果希望模块更简洁，第一版也可以不用 learnable confidence，只用固定 sparse edge mask。`C_ij` 不是 DKDL 的核心，核心是 directed depth layout `R_ij`。

### 5. Graphormer-Style Sparse Depth Attention

只在稀疏边上做 relation-guided attention：

```text
A_ij = Q_i K_j / sqrt(d)
     + b_depth(R_ij)
     + b_hop(hop_ij)
     + b_type(edge_type_ij)
```

mask 非边：

```text
A_ij = -inf, if (i,j) not in E and i != j
```

消息：

```text
M_i = Σ_{j∈N(i)} softmax_j(A_ij) · C_ij · V_j
```

这里借鉴的是 Graphormer 的 structural bias 思想，但不是照搬 Graphormer 代码。

### 6. Layout Residual Adapter

最后用 layout message 更新 AMS depth token。这里不需要复杂保护项，只保留标准残差形式：

```text
u_i = MLP_update(LN(M_i))
g_i = sigmoid(MLP_gate([F_d_i, X_i, M_i]))
F_d_layout_i = F_d_i + g_i · u_i
```

这个 residual adapter 是模块机制的一部分，不是工程补丁：它表示 DKDL 只把显式 depth layout 转换成对原 `F_d` 的结构化增量，而不是替换 AMS 已经提取到的深度 token。

## 训练目标

不要拆太多 loss。第一版只用一个统一布局损失：

```text
L = L_3D + λ_layout L_DKDL
```

```text
target_ij = tanh( normalize_per_frame(Z_i - Z_j) )
L_DKDL = SmoothL1(R_ij, target_ij), (i,j)∈E
```

可选加入 sign loss，但不作为第一版主线：

```text
L_ord = BCEWithLogits(raw_R_ij, 1[Z_i > Z_j])
```

`λ_layout` 是唯一需要调的权重。第一版建议从 `1e-3` 或 `1e-2` 比较，不引入额外 warmup 机制。

## 为什么比 OARDG 稳定

| OARDG | DKDL |
|---|---|
| raw depth contrast 参与 gate | 不用 raw depth，避免深度噪声直接控制更新 |
| occ_pred 无直接监督 | 不建模遮挡伪标签，只建模 relation confidence |
| 全 pair relation + occ bias | 稀疏结构边 + 动态 close pair |
| residual 被 `1+occ` 放大 | 标准 residual layout adapter |
| 多个启发式信号并联 | 单一对象 `Sparse Directed Depth Graph` |
| 机制难以归因 | 通过 `R_ij` 的 layout 指标直接验证 |

## 最小实现版本

第一版只实现：

```text
canonicalization
fixed sparse edges
anti-symmetric R_ij
relation bias attention
residual layout update
layout loss
```

暂时不做：

- raw depth gate；
- occlusion head；
- part token；
- cycle loss；
- 复杂训练保护项。

如果第一版稳定超过 AMS / posealign，再逐步加：

1. dynamic close pair；
2. edge type bias；
3. part-level layout token。

## 实验设计

不做大量拆解。只做高信号对比：

```text
AMS only
AMS + PoseAlign
AMS + DKDL
AMS + DKDL w/o layout loss
```

机制指标：

- `R_ij` pairwise ordinal accuracy；
- ambiguous-pair ordinal accuracy；
- relation geometry Spearman with `|Z_i-Z_j|`；
- wrist / elbow / knee / ankle per-joint MPJPE；
- MPJPE 是否在早期训练退化。
