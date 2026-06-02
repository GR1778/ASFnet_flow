# AMS + DLST 两张图汇报讲解稿

本文档用于配合两张图做汇报：

1. **整体框架图**：Overview of the Proposed AMS + DLST Framework  
2. **DLST 模块细节图**：Depth-Layer Sorting Transformer (DLST)

建议汇报顺序是：先用整体框架图说明 DLST 放在系统哪里、为什么需要它；再用模块细节图说明 DLST 内部如何从高维 joint token 推导相对深度结构。

## 总体讲法

可以先用一句话开场：

```text
AMS 负责从 RGB/depth 特征中为每个关节采集局部 depth-aware evidence；
DLST 进一步把这些高维关节特征组织成有序的 skeleton depth layout，
也就是显式建模关节之间的前后关系。
```

这里要强调一点：

```text
DLST 不是直接比较深度图像素值，也不是直接回归每个关节的标量深度。
它接收的是 AMS 输出的高维 depth-aware joint tokens，并从中学习关节到有序深度层的软分配。
```

---

# 图 1：整体框架图讲解

图 1 的作用是回答：

```text
DLST 为什么要做？它放在整个 AMS/ASFnet pipeline 的什么位置？监督信号从哪里来？
```

## Step 0：Input RGB Image

**图中位置**：最左侧输入 RGB image。

**讲解重点**：

输入是一张 monocular RGB image。单目 3D pose 的核心困难是从 2D 图像恢复 3D 结构，其中深度方向存在天然歧义。尤其在自遮挡、肢体交叉、关节在图像平面接近时，模型容易判断错谁在前谁在后。

**可以这样说**：

```text
输入只有单张 RGB 图像，因此模型必须从 2D pose、RGB appearance 和估计深度图中恢复 3D pose。
DLST 关注的不是普通图像特征增强，而是关节之间的 front-back ordering。
```

## Step 1：2D Pose Estimator / 2D Joint Prior

**图中位置**：RGB 图像上方分支，经过 2D Pose Estimator 得到 2D joint prior。

**对应代码/数据**：

训练中使用 `keypoints_2d_cpn` 和 `keypoints_2d_cpn_crop` 作为 2D 关节输入。

**讲解重点**：

2D pose 提供关节位置先验。AMS 后续会围绕这些 2D joint 位置进行局部采样。这个分支回答的是“在哪里采样”。

**可以这样说**：

```text
2D pose estimator 提供 joint prior，告诉 AMS 每个关节大致在图像中的位置。
这个位置先验决定后续 RGB/depth 特征的采样中心。
```

## Step 2：Monocular Depth Estimator / Relative Depth Cue

**图中位置**：RGB 图像中部分支，经过 depth estimator 得到 relative depth map。

**对应代码/数据**：

训练中输入 `depth_images_batch`，再通过 `self.depth_embed(depth_images.unsqueeze(1))` 得到 depth feature map。

**讲解重点**：

深度图提供额外的几何线索，但它是估计结果，存在噪声、尺度不稳定、遮挡区域不可靠等问题。因此不适合简单地拿两个 joint 的单点深度值直接比较。

**可以这样说**：

```text
Depth estimator 提供的是相对深度 cue，但该深度图不是完美 metric depth。
单点 depth value 容易受遮挡、背景和 2D joint 偏差影响，所以我们不直接比较像素深度，
而是把深度图转为可学习的 depth-aware feature。
```

## Step 3：Visual Backbone / Multi-scale RGB Features

**图中位置**：RGB 图像下方分支，经过 visual backbone 得到多尺度 RGB features。

**对应代码**：

`features_list_hr = self.backbone(images)`

**讲解重点**：

RGB feature 提供外观、肢体边界、纹理和上下文。多尺度特征对不同大小的身体部位都重要。

**可以这样说**：

```text
视觉 backbone 提取多尺度 RGB features，提供外观和语义上下文；
这些特征和 depth feature 会在 AMS 中围绕 joint prior 进行对齐和采样。
```

## Step 4：AMS - Adaptive Multimodal Sampling

**图中位置**：中左绿色 AMS 大框。

**对应代码**：

`DGLifting_dlst.py` 中的 `RGBD_Extraction`：

```python
for blk in self.RGBD_Extraction:
    x = blk(x, ref, features_list_hr)
```

AMS 输出：

```python
depth_joint_tokens = x[:, -1]
```

形状：

```text
F_d ∈ R^{B×J×C}, J=17, C=128
```

**讲解重点**：

AMS 的作用是基于 2D joint prior，从多尺度 RGB/depth feature map 中自适应采样局部证据，得到每个关节的 depth-aware token。这个 token 是高维特征，不是标量深度。

**可以这样说**：

```text
AMS 把 RGB、depth 和 2D joint prior 对齐起来，为每个关节生成一个 128 维 depth-aware token。
但这个 token 仍然是局部且隐式的，它没有显式说明关节之间谁在前、谁在后。
```

**关键过渡**：

```text
因此 DLST 的目标就是把 AMS 采到的局部 depth evidence，整理成全局一致的 skeleton depth layout。
```

## Step 5：DLST - Depth-Layer Sorting Transformer

**图中位置**：中右橙色 DLST 大框。

**作用概括**：

```text
F_d -> ordered depth layers -> assignment A -> R=AΩA^T -> depth-biased attention -> F'_d
```

这一大框在图 1 中拆成 6 个细步骤。它们和图 2 的 3 个阶段一一对应。

### Step 5.1：Learnable Ordered Depth Layers

**图中含义**：

学习 `K` 个有序 depth layer tokens：

```text
L ∈ R^{K×C}
```

这些层从 front 到 back 有固定顺序。

**对应代码**：

```python
self.layer_content
self.layer_order_embed
layer_tokens = self.layer_content + self.layer_order_embed
```

**讲解重点**：

这不是固定人体部位层，而是每一帧自适应的深度容器。比如某一帧 foreground layer 可能包含手腕，另一帧可能包含膝盖。

**可以这样说**：

```text
DLST 先定义少量有序 depth layers。它们不是人体语义部位，而是可学习的 front-to-back 容器。
这样模型不需要预测连续 metric depth，只需要判断关节落在哪个相对深度层。
```

### Step 5.2：Layer Aggregation

**图中含义**：

Depth layer tokens 通过 cross-attention 从 joint tokens 中聚合当前样本的全局骨架信息：

```text
L' = CrossAttn(L, F_d)
```

**对应代码**：

`LayerGatherBlock.forward(layer_tokens, joint_tokens)`

**讲解重点**：

深度层不是静态模板，而是根据当前人体姿态和深度证据动态更新。

**可以这样说**：

```text
Layer aggregation 让 depth layers 先读取所有关节的 depth-aware evidence，
从而形成适配当前姿态的 ordered layer representation。
```

### Step 5.3：Soft Joint-to-Layer Assignment

**图中含义**：

每个关节对 `K` 个 depth layers 做软分配：

```text
A = Softmax(Q(F_d)K(L')^T / sqrt(C))
A ∈ R^{J×K}
```

**对应代码**：

```python
joint_q = self.assign_joint(self.joint_norm(joint_tokens))
layer_k = self.assign_layer(self.layer_norm(layer_tokens))
assign_logits = (joint_q @ layer_k.transpose(-1, -2)) / math.sqrt(c)
assign = F.softmax(assign_logits / self.assignment_temperature, dim=-1)
```

**讲解重点**：

这里不是硬分类，而是 soft assignment。软分配更适合人体关节，因为有些关节本来就在层边界附近。

**可以这样说**：

```text
每个 joint 都会得到一个对 K 个 depth layers 的概率分布。
这相当于把高维 joint token 映射到一个可解释的 depth-layer coordinate。
```

### Step 5.4：Relative Depth Matrix

**图中含义**：

通过固定层间顺序矩阵 `Ω` 推导关节相对深度矩阵：

```text
R = AΩA^T
Ω_ab = tanh((b-a)/τ)
```

**对应代码**：

```python
omega = torch.tanh((idx[None, :] - idx[:, None]) / omega_temperature)
rel_depth = assign @ self.omega @ assign.transpose(-1, -2)
```

**讲解重点**：

`R_ij > 0` 表示 joint `i` 更可能在 joint `j` 前面。由于 `Ω` 是反对称矩阵，`R` 也天然反对称：

```text
R_ij = -R_ji
```

这保证了 pairwise relation 的基本几何一致性。

**可以这样说**：

```text
相比直接预测任意 pairwise relation，AΩA^T 让相对深度关系来自少量有序层。
这样得到的 R 具有反对称性和全局一致性，也便于可视化。
```

### Step 5.5：Depth-Biased Self-Attention

**图中含义**：

用 `R` 作为 attention bias 引导 depth token 交互：

```text
Attn = Softmax(QK^T / sqrt(d) + αR)V
```

**对应代码**：

```python
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = attn + self.depth_gate * rel_depth.unsqueeze(1)
attn = attn.softmax(dim=-1)
```

**讲解重点**：

这一步不是只把 `R` 输出出来做辅助监督，而是把它注入到 feature refinement 中，让前后关系影响 joint token 的交互。

**可以这样说**：

```text
R 不只是一个中间诊断量。它会作为 attention bias 参与 token refinement，
使模型在融合关节信息时显式感知 front-back ordering。
```

### Step 5.6：Depth-Ordered Joint Tokens

**图中含义**：

输出 refined depth-aware tokens：

```text
F'_d ∈ R^{J×C}
```

同时输出：

```text
R ∈ R^{J×J}
A ∈ R^{J×K}
```

**对应代码**：

```python
return out, rel_depth, assign
```

**讲解重点**：

`F'_d` 进入后续 pose regression pipeline；`R` 和 `A` 用于监督、诊断和可视化。

**可以这样说**：

```text
最终 DLST 输出三个东西：refined tokens 用于提升 3D pose，R 用于表示相对深度关系，A 用于解释每个关节属于哪个 depth layer。
```

## Step 6：Multimodal Fusion Transformer

**图中位置**：DLST 之后右侧第一个模块。

**对应代码**：

```python
x = torch.cat((x[:, :-1], depth_joint_tokens.unsqueeze(1)), dim=1)
for blk in self.Features_Fusion:
    x = blk(x)
```

**讲解重点**：

DLST 只更新 depth token。更新后的 depth token 会和 pose/RGB/depth 多模态 tokens 一起融合。

**可以这样说**：

```text
DLST refined depth token 会被放回原来的 multimodal token set 中，
再通过 fusion transformer 与 pose token 和 RGB tokens 交互。
```

## Step 7：Spatial Reasoning Transformer

**图中位置**：右侧第二个模块。

**对应代码**：

```python
for blk in self.Spatial_Transformer:
    x = blk(x)
```

**讲解重点**：

这一步在 joint 维度上建模全身空间结构。

**可以这样说**：

```text
Spatial transformer 负责建模 17 个关节之间的全身结构关系，
DLST 提供的 depth-ordered tokens 会在这里进一步影响最终 pose 表达。
```

## Step 8：3D Pose Head

**图中位置**：最右侧输出 predicted 3D human pose。

**对应代码**：

```python
x = self.head(x).view(b, 1, p, -1)
```

**讲解重点**：

最终输出 3D pose，并用 `L_3D` 监督。

**可以这样说**：

```text
最终 pose head 回归每个关节的 3D 坐标，主监督仍然是标准 3D pose loss。
DLST 的排序监督只是辅助约束，用来让中间深度结构可学习。
```

## Bottom Supervision：监督分支

**图中位置**：底部 dashed supervision band。

### 3D Pose Loss

```text
L_3D
```

用于监督最终预测 3D pose。

### Depth Ordering Loss

```text
L_order
```

从 GT 3D pose 的 z 坐标构造 pairwise ordering target：

```text
y_ij = sign(z_i - z_j)
```

只统计真实深度差超过阈值的 pair：

```text
|z_i - z_j| > 0.05m
```

总损失：

```text
L = L_3D + λ_order L_order
```

当前：

```text
λ_order = 0.001
```

**可以这样说**：

```text
Depth ordering loss 不要求 DLST 回归绝对深度值，只要求 R 的符号和 GT 前后顺序一致。
这样避免了 metric depth 尺度不稳定的问题，同时提供了明确的结构监督。
```

---

# 图 2：DLST 模块细节图讲解

图 2 的作用是回答：

```text
DLST 内部到底怎么从高维 F_d 得到相对深度结构？
```

图 2 把图 1 中 DLST 的 6 个小步骤概括成 3 个阶段。

## 图 2 Stage 1：Ordered Depth-Layer Aggregation

**对应图 1**：

```text
Step 5.1 Learnable Ordered Depth Layers
Step 5.2 Layer Aggregation
```

**输入**：

```text
F_d ∈ R^{B×J×C}
```

**核心计算**：

```text
L' = CrossAttn(L, F_d)
```

**讲解重点**：

这一步先建立“有序深度层”这个中间结构。`K` 个 layers 从 front 到 back 排列，cross-attention 让这些层读取当前样本的 joint evidence。

**推荐讲法**：

```text
第一阶段不是直接对 joint 做 pairwise 判断，而是先构造一组有序 depth layers。
这些 layers 通过 cross-attention 聚合所有 joint tokens，从而得到适配当前人体姿态的 depth-layer representation。
```

## 图 2 Stage 2：Soft Layer Assignment & Relative Depth Matrix

**对应图 1**：

```text
Step 5.3 Soft Joint-to-Layer Assignment
Step 5.4 Relative Depth Matrix
```

**核心计算**：

```text
A = Softmax(Q(F_d)K(L')^T / sqrt(C))
R = AΩA^T
Ω_ab = tanh((b-a)/τ)
```

**讲解重点**：

这一阶段是 DLST 的核心：先得到 joint-to-layer assignment `A`，再通过固定层间顺序 `Ω` 得到 pairwise relative depth matrix `R`。

**推荐讲法**：

```text
第二阶段把每个 joint token 投影到有序 depth layers 上。
每个关节不是被硬分到一个层，而是得到一个 K 维概率分布 A。
然后由 A 和固定的层间顺序 Ω 推导 R。
因此 R 不是任意学出来的 pairwise matrix，而是来自全局 depth-layer ordering。
```

**强调优势**：

```text
1. R 天然反对称；
2. pairwise relation 来自少量有序层，减少不一致关系；
3. A 和 R 都可以可视化，解释性强。
```

## 图 2 Stage 3：Relative-Depth-Guided Refinement

**对应图 1**：

```text
Step 5.5 Depth-Biased Self-Attention
Step 5.6 Depth-Ordered Joint Tokens
```

**核心计算**：

```text
Attn = Softmax(QK^T / sqrt(d) + αR)V
```

**输出**：

```text
F'_d ∈ R^{B×J×C}
R ∈ R^{B×J×J}
A ∈ R^{B×J×K}
```

**讲解重点**：

相对深度关系 `R` 不是只用于 loss，而是进入 attention logits，直接影响 token refinement。

**推荐讲法**：

```text
第三阶段把相对深度矩阵 R 注入 self-attention。
这样 joint token 在交互时不仅看 appearance similarity，也会参考前后深度关系。
最终得到 depth-ordered joint tokens F'_d，并继续送入后续 pose regression pipeline。
```

---

# 两张图之间的关系

两张图并不冲突，只是抽象粒度不同：

| 图 1 的 6 个小步骤 | 图 2 的 3 个阶段 | 说明 |
|---|---|---|
| 1 Learnable Ordered Depth Layers | Stage 1 | 初始化有序深度层 |
| 2 Layer Aggregation | Stage 1 | 层 token 聚合 joint evidence |
| 3 Soft Joint-to-Layer Assignment | Stage 2 | 得到 assignment `A` |
| 4 Relative Depth Matrix | Stage 2 | 得到 `R=AΩA^T` |
| 5 Depth-Biased Attention | Stage 3 | 用 `R` 引导 attention |
| 6 Depth-Ordered Joint Tokens | Stage 3 | 输出 refined token `F'_d` |

汇报时可以这样衔接：

```text
整体框架图中为了展示完整 pipeline，我们把 DLST 展开成 6 个计算步骤；
模块图中为了突出核心思想，我们把这 6 个步骤归纳为 3 个阶段：
depth-layer aggregation、ordering inference、depth-guided refinement。
```

---

# 老师可能会问的问题

## Q1：AMS 输出不是标量深度，为什么能做 depth sorting？

**回答**：

AMS 输出的是高维 depth-aware token，不是标量 depth。但这个 token 来自 RGB/depth feature 的 joint-centered sampling，包含隐式深度证据。DLST 不是直接比较像素深度，而是通过监督学习把这些高维 token 映射到有序 depth layers，再推导 pairwise ordering。

## Q2：为什么不直接比较 depth map 上两个 joint 的深度值？

**回答**：

单点 depth value 容易受 2D joint 偏差、遮挡、背景和深度估计噪声影响。尤其被遮挡关节处的像素深度可能对应遮挡物表面，而不是关节本身。DLST 使用 AMS 后的高维 token，可以利用局部上下文和多尺度证据，比直接 pixel depth comparison 更鲁棒。

## Q3：为什么用少量 depth layers，而不是直接预测所有 pairwise depth？

**回答**：

直接预测任意 pairwise matrix 容易出现不一致关系，比如 `i` 在 `j` 前、`j` 在 `k` 前，但 `k` 又在 `i` 前。DLST 用少量有序层作为中间结构，`R=AΩA^T` 天然反对称，并且来自全局一致的 layer ordering。

## Q4：`dlst_rel_sign_acc` 高说明什么？

**回答**：

它说明 DLST 输出的 `R` 在验证集上与 GT 前后顺序高度一致。当前训练第 7 个 epoch `val dlst_rel_sign_acc≈0.953`，说明中间深度排序结构学得很好。但它是机制指标，不等价于最终 MPJPE，最终还要看完整训练和消融。

## Q5：这个模块和 UDE 的区别是什么？

**回答**：

UDE 主要解决 depth feature 的不确定性和噪声抑制问题；DLST 解决的是 depth-aware tokens 缺少全局前后组织的问题。二者的动机不同：UDE 是 reliability modeling，DLST 是 front-back layout modeling。

---

# 一分钟汇报版本

如果时间很短，可以这样讲：

```text
我们的 DLST 放在 AMS 之后。AMS 已经能为每个关节采集 RGB/depth 局部证据，但输出是高维 token，
没有显式表达关节之间的前后关系。DLST 的核心是把这些 joint tokens 软分配到少量有序 depth layers，
再通过 R=AΩA^T 得到全局一致的 relative depth matrix。这个 R 天然反对称，表示 joint i 是否在 joint j 前面，
并进一步作为 attention bias 指导 depth token refinement。

训练时我们不用绝对深度回归，而是用 GT 3D pose 的 z 坐标构造 pairwise ordering loss。
当前第 7 个 epoch 验证集 dlst_rel_sign_acc 约 0.953，assignment 没有塌缩，4 个 layer 都被使用；
MPJPE 已降到 39.0mm，训练还没结束但趋势比较积极。
```
