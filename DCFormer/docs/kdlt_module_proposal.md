# KDLT：运动学深度布局 Transformer

## 一句话概括

AMS 解决 **在哪里采样**；KDLT 解决 **采到的深度线索如何组成骨架三维布局**。

KDLT 是一个放在 AMS 之后的模块，用来把点式深度 token 转换成结构化的 **运动学深度布局场**，包含 root-relative joint depth、directed pairwise depth order 和 limb-level layout message。它不是 UDE 式去噪，而是表示转换：把隐式深度证据转换成显式骨架深度几何。

## 为什么需要这个模块

我们已有诊断给出几个同时成立的事实：

| 观察 | 指标 | 含义 |
|---|---:|---|
| AMS 输出已经具有姿态信息 | `F_d` 距离 vs 3D pose 距离 Spearman `0.854` | 问题不是缺少 pose-aware 表征。 |
| 原始 `F_d` 几何不服从真实深度布局 | `F_d` pairwise 距离 vs `|Z_i-Z_j|` Spearman `-0.063` | 深度关系没有被显式组织到特征空间里。 |
| 相对深度信息是潜在可恢复的 | 从 `F_i-F_j` 线性预测 `ΔZ`，R2 `0.965` | 信号存在，但藏在特征差分里。 |
| 标量深度头的排序能力较弱 | `mu` 全 joint pair ordinal acc `0.623` | 单点深度标量不足以稳定表达关节前后关系。 |
| `F_d` 有明显全局共享成分 | body-mean energy fraction `0.544` | 关系相关的 residual 没有被单独分离出来。 |

因此，新的模块不应该只是继续“增强特征”，而应该显式构建骨架相对深度表示。

## 外部论文动机

相关工作从不同角度支持这个方向：

- Ordinal depth supervision 表明，单目 2D-to-3D 的歧义很大程度来自关节前后顺序不明确。
- DRPose3D 认为 depth ranking 含有丰富 3D 信息，可以缓解 2D-to-3D lifting 的病态问题。
- HMOR 说明层级 ordinal relation 可以同时表达 body-part 级和 joint 级语义，并保持全局一致性。
- 关节深度预测类工作通常同时利用全局骨架结构和局部图像证据，这与 AMS 后需要把深度 token 转换成骨架结构是一致的。

KDLT 的区别在于：它不直接从 RGB patch 或 heatmap 推 depth ranking，而是在 lifting 网络内部，把 AMS 采样后的深度 token 转换成可微的 depth layout field。

## 模块定义

主模块输入：

```text
F_d    ∈ R^{B×J×C}    AMS 深度 token
P      ∈ R^{B×J×2}    归一化 2D 关节坐标
G_skel                 固定骨架拓扑
```

输出：

```text
F_d^layout ∈ R^{B×J×C}
z           ∈ R^{B×J×1}     root-relative latent depth
R           ∈ R^{B×J×J}     directed pairwise depth layout
```

`R_ij` 表示 joint `i` 相对 joint `j` 的前后/深度关系，并且强制反对称：

```text
R_ij = -R_ji, R_ii = 0
```

这里有一个明确设计边界：KDLT 主模块不直接输入 `F_rgb` 或原始深度图 `D_raw`。AMS 已经完成 RGB / depth / pose 到 joint-level token 的对齐，KDLT 的职责应当是把 AMS 后的 `F_d` 转换成显式深度布局。如果再次引入 RGB 或 raw depth，模块会退化成另一个多模态融合器，收益归因也会变得不清楚。

`F_p` 可以作为消融变体加入，但不建议放进主版本。`P` 已经提供相对 2D 几何和骨架位置先验；主版本保持 `KDLT(F_d, P, G_skel)` 更干净，也更符合我们从 `F_d` 诊断得到的动机。

## 模块结构

### 1. 深度布局规范化

先去掉 AMS 深度 token 中的 frame-shared/body-shared 成分：

```text
G = LN(F_d - mean_j(F_d) + XYProj(P - P_root))
```

这一步直接对应我们的诊断结果：`F_d` 里全局成分较强，容易淹没关节间相对深度 residual。

### 2. Root-Relative 深度势能

预测一个归一化的 latent depth potential：

```text
z_i = Head_z(G_i)
z_i ← z_i - mean_j(z_j)
```

这里的 `z_i` 不是最终 metric Z 坐标，而是一个 scale-free 的深度势能，用来组织后续关系推理。

### 3. Directed Pairwise Layout Field

对选定 joint pair `(i,j)` 建模相对深度。这里使用我们实验证明最强的信号 `F_i - F_j`，而不是只依赖 anchor bin：

```text
φ_ij = [G_i - G_j, G_i * G_j, z_i - z_j, P_i - P_j, ||P_i-P_j||]
r_ij = MLP_pair(φ_ij)
R_ij = tanh((z_i - z_j) + r_ij - r_ji)
```

pair 集合不建议一开始用全部 136 对，而是优先使用：

- skeleton bones；
- symmetric limb pairs；
- wrists / elbows / knees / ankles 等易混淆 pair；
- 每帧 2D 距离最近的 top-k pair。

这样可以避免把无关 pair 的噪声也纳入监督。

### 4. 层级肢体布局 token

把关系消息按三层聚合：

```text
joint -> bone -> limb/part
```

建议使用五个 part：

```text
right leg, left leg, torso, right arm, left arm
```

每个 part 从内部 pairwise relation 中生成一个 layout token，再把消息传回成员 joint。这样比单纯 pairwise attention 更容易保持局部肢体一致性和全局骨架一致性。

注意：part token 来自 `F_d`、`P` 和固定 skeleton，不来自 RGB。这样它仍然是一个 depth-layout 模块，而不是提前做多模态融合。

### 5. Layout-Guided Recomposition

不要只把 `R` 当 attention bias。应该把它转换成 layout message，然后残差更新深度 token：

```text
m_i = Σ_j softmax_j(Q_iK_j + Bias(R_ij) + TopologyBias_ij) V_j
p_i = PartMessage(i)
g_i = sigmoid(MLP([F_d_i, G_i, m_i, p_i]))
F_d^layout_i = F_d_i + γ · g_i · MLP(LN(m_i + p_i))
```

`γ` 建议初始化为很小的值，或者作为 learnable scalar，从而让模块一开始接近 identity，降低训练不稳定风险。

## 训练目标

主损失仍然是 MPJPE：

```text
L = L_3D + λ_layout L_layout
```

布局损失统一为：

```text
L_layout = L_root + α L_pair + β L_cycle
```

### Root-Relative Term

```text
target_z_i = normalize_per_frame(Z_i - Z_root)
L_root = SmoothL1(z_i, target_z_i)
```

### Pairwise Ordinal / Gap Term

对选定 pair：

```text
target_ij = tanh(normalize(Z_i - Z_j))
L_pair = SmoothL1(R_ij, target_ij)
```

也可以加 sign loss：

```text
L_ord = BCEWithLogits(raw_R_ij, 1[Z_i > Z_j])
```

### Cycle Consistency Term

深度布局应该具有全局一致性：

```text
R_ij + R_jk ≈ R_ik
L_cycle = mean |R_ij + R_jk - R_ik|
```

这个项只建议用于短骨架链和同一肢体内的 triple，不建议一开始对所有 triple 做。

## 为什么比 PoseAlign / OARDG 更整体

PoseAlign 和 OARDG 里面有很多有用元素，但它们更像把 anchor、relation、topology bias、raw depth contrast、occlusion gate、bone message 分开拼接。

KDLT 只有一个中心对象：

```text
Kinematic Depth Layout Field = {z_i, R_ij, part layout messages}
```

所有子结构都服务于这个对象：

- canonicalization：让 depth residual 显出来；
- `z_i`：提供 root-relative 深度势能；
- `R_ij`：提供 directed pairwise layout；
- part token：提供层级肢体一致性；
- cycle loss：约束全局关系一致性；
- recomposition：把 layout field 注入回 `F_d`。

论文故事也更清晰：

```text
AMS 提取 pose-relevant depth evidence。
KDLT 将这些 evidence 转换成 skeletal depth layout。
融合 Transformer 使用 layout-aware depth token 回归 3D pose。
```

## 预期收益和风险

收益合理性：

- PoseAlign 已经把单 AMS 从 `39.4` 提到约 `38.9`，说明 layout modeling 方向有效。
- 我们的诊断显示，从 `F_i-F_j` 恢复 `ΔZ` 的 R2 达到 `0.965`，可利用信号很强。
- 但原始 `F_d` 特征空间没有按真实 `|ΔZ|` 组织，KDLT 正好补这个表示缺口。

主要风险：

- 如果 fusion transformer 已经隐式利用了这部分信号，KDLT 可能收益有限。
- pairwise loss 太多会过约束，可能损害最终 MPJPE。
- close-pair / occlusion 逻辑如果依赖不稳定 depth map，可能引入噪声。

控制风险：

- residual update，`γ` 小初始化；
- 先用精选 pair，再做 all-pair ablation；
- `λ_layout` 从 `1e-4` 到 `1e-2` 扫；
- 不只看 MPJPE，也看 layout 指标是否改善。

## 消融计划

最低限度应该跑：

1. `AMS only`
2. `AMS + KDLT(no layout loss)`
3. `AMS + KDLT(root only)`
4. `AMS + KDLT(pair only)`
5. `AMS + KDLT(root + pair)`
6. `AMS + KDLT(root + pair + cycle)`
7. `AMS + KDLT(full with part tokens)`

评价指标：

- MPJPE / PA-MPJPE；
- pairwise ordinal accuracy；
- ambiguous-pair ordinal accuracy；
- relation token geometry 和 `|ΔZ|` 的 Spearman；
- wrist / elbow / knee / ankle 等关节的 per-joint MPJPE；
- depth noise 下的鲁棒性。

## 实现建议

先做 compact 版：

```text
canonicalize -> z head -> pair relation head -> relation attention -> residual update
```

一开始不要加复杂 raw-depth occlusion gate。如果 compact 版能接近或超过 posealign，并且 layout 指标更好，再加入 part token、cycle loss、close-pair dynamic edges。

不建议第一版加入 RGB token 或 raw depth gate。它们可能提高局部结果，但会让模块故事重新变碎，并且和 AMS / 后续 multimodal fusion 的职责重叠。
