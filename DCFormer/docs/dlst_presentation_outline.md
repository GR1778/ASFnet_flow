# DLST 汇报提纲

这份提纲用于给老师汇报 DLST。核心原则是：先讲清楚“为什么需要”，再讲“怎么做”，最后讲“现在有什么证据”。不要一开始陷入公式和代码细节。

## 汇报主线

一句话主线：

```text
AMS 已经能采到关节附近的 RGB-depth 局部证据，但这些证据仍是高维、隐式、局部的；
DLST 的目标是把这些 depth-aware joint tokens 进一步整理成显式、全局一致的骨架前后层次。
```

要避免的说法：

```text
AMS 输出 coarse depth。
DLST 是直接比较深度图像素值。
DLST 只是替代 UDE 做去噪。
```

更准确的说法：

```text
AMS 输出高维 depth-aware token F_d。
原 ASFnet/UDE 分支会从 F_d 预测 coarse depth 和 uncertainty。
DLST 从 F_d 学 joint-to-layer assignment，并推导相对深度矩阵 R。
```

---

# Slide 1：问题背景

## 标题

```text
问题：深度特征被采到了，但前后结构没有显式组织
```

## 页面内容

```text
单目 3D HPE 的关键歧义之一：
图像上接近或重叠的关节，真实 3D 中可能有明显前后关系。

典型困难：
手腕 vs 躯干
左右腿交叉
肘/膝/踝等末端关节遮挡
```

## 讲解稿

```text
ASFnet 通过 AMS 引入深度图和多尺度 RGB 特征，已经能围绕关节采样局部 depth-aware evidence。
但 AMS 的输出是每个关节一个 128 维 token，不是显式的前后关系。
所以问题不是“有没有深度信息”，而是“这些深度信息有没有被组织成全身一致的 front-back layout”。
```

---

# Slide 2：概念区分

## 标题

```text
先区分三个概念：AMS token、coarse depth、DLST relation
```

## 页面内容

| 名称 | 来源 | 形状 | 含义 |
|---|---|---:|---|
| AMS depth-aware token `F_d` | AMS 输出 | `[B,17,128]` | 高维局部 RGB-depth 证据 |
| 原 ASFnet/UDE coarse depth `μ` | UDE depth head | `[B,17,1]` | 从 `F_d` 回归的标量粗深度 |
| DLST `R` | DLST 推导 | `[B,17,17]` | 关节两两前后关系 |

## 讲解稿

```text
这里容易混淆：coarse depth 不是 AMS 直接输出，而是原 ASFnet/UDE 分支接在 AMS 后面的一个小 depth head 预测出来的。
DLST 也不是直接比较 depth map 像素，而是接收 AMS 的高维 token F_d，从中学习关节到有序深度层的软分配。
```

---

# Slide 3：测试动机

## 标题

```text
测试动机：DLST 是否有必要？
```

## 页面内容

```text
我们先不直接堆模块，而是用诊断脚本回答三个问题：

Q1：少量有序深度层是否足以表达真实关节前后关系？
Q2：AMS 的高维 token 中是否存在可解码的深度层次线索？
Q3：原 ASFnet/UDE 的标量 coarse depth 是否已经解决易混淆关节的前后判断？
```

## 讲解稿

```text
如果 AMS 本身已经显式组织好了深度结构，那么原始 F_d 的特征距离应该和真实深度差高度相关；如果原 ASFnet/UDE 的标量深度头已经足够，coarse depth 也应该能很好判断困难关节对的前后顺序。
所以我们先做诊断，验证 DLST 的设计是不是有数据依据。
```

---

# Slide 4：测试结果

## 标题

```text
诊断结果：AMS 有深度线索，但仍缺少显式结构
```

## 页面内容

4096 个 Human3.6M 验证样本：

| 诊断项 | 结果 | 结论 |
|---|---:|---|
| 4 个有序深度层的理论前后排序准确率 | **0.872** | 少量层足以近似真实前后关系 |
| 易混淆关节对的理论前后排序准确率 | **0.859** | 分层结构对困难关节仍有效 |
| 从 AMS 特征预测相邻深度层准确率 | **0.804** | AMS 中有可解码的层次线索 |
| 从 AMS 特征推导相对深度矩阵准确率 | **0.749** | 简单 probe 能抽出部分关系 |
| 原 ASFnet/UDE coarse depth 在易混淆关节对上的准确率 | **0.551** | 标量粗深度接近随机 |
| 原始 AMS 特征距离与真实深度差相关性 | **0.103** | 原始特征空间没有自然按深度组织 |

## 讲解稿

```text
这组结果不是证明 AMS 已经足够好，而是证明 AMS 有可用的深度原材料。
但是原始 F_d 和真实深度差的相关性只有 0.103，说明它没有自然形成深度几何空间；
原 ASFnet/UDE 的 coarse depth 在易混淆关节对上只有 0.551，接近随机。
因此 DLST 的必要性是把 AMS 的隐式 depth evidence 转成显式前后结构。
```

## 本页结论

```text
AMS 解决“有没有局部深度证据”；
DLST 解决“如何把局部证据组织成全身前后关系”。
```

---

# Slide 5：整体框架图

## 标题

```text
AMS + DLST 整体框架
```

## 页面放图

放整体框架图：

```text
Input RGB
→ 2D pose / depth / RGB features
→ AMS
→ DLST
→ multimodal fusion
→ spatial transformer
→ 3D pose
```

## 讲解稿

```text
整体上，2D pose 提供采样位置，depth estimator 提供相对深度 cue，visual backbone 提供 RGB 多尺度特征。
AMS 负责围绕关节采集局部 RGB-depth evidence，输出 depth-aware joint tokens F_d。
DLST 插在 AMS 之后，把 F_d 组织成有序 depth layers 和相对深度矩阵，再把 refined depth token 送回后续 fusion 和 pose regression。
```

---

# Slide 6：DLST 模块图

## 标题

```text
DLST：从高维 token 到骨架前后布局
```

## 页面放图

放 DLST 模块细节图。

## 三阶段讲法

```text
1. Depth-layer aggregation
   构造 K 个有序 depth layers，并通过 cross-attention 聚合 joint evidence。

2. Ordering inference
   每个 joint 软分配到 K 个 depth layers，得到 assignment A；
   再由固定层间顺序 Ω 推导 R=AΩA^T。

3. Depth-guided refinement
   将 R 加入 self-attention logits，引导 depth token 交互，输出 F'_d。
```

## 讲解稿

```text
DLST 的关键不是直接回归绝对深度，而是构造一个中间结构：有序 depth layers。
每个关节对这些层做软分配，得到 A。
由于层之间有固定前后顺序 Ω，所以可以推导出全局一致的相对深度矩阵 R=AΩA^T。
最后 R 作为 attention bias 参与 token refinement。
```

---

# Slide 7：DLST 的结构优势

## 标题

```text
为什么用 AΩA^T，而不是直接预测 pairwise relation？
```

## 页面内容

```text
直接预测任意 R：
- 容易出现 pairwise 不一致
- 缺少全局层次约束
- 可解释性弱

DLST 的 R=AΩA^T：
- 由少量有序 depth layers 推导
- 天然反对称：R_ij = -R_ji
- 全局一致，减少循环矛盾
- A 和 R 都可视化
```

## 讲解稿

```text
如果直接对每一对关节预测前后关系，模型可能学出互相矛盾的关系。
DLST 用少量有序层作为中间结构，所有 pairwise relation 都从这些层推导出来。
因此它不是普通 pairwise graph，而是一个可微的 skeleton z-buffer。
```

---

# Slide 8：监督方式

## 标题

```text
Depth Ordering Loss：监督前后顺序，而不是绝对深度
```

## 页面内容

```text
GT 3D pose 提供 z 坐标
y_ij = sign(z_i - z_j)

只统计明显前后关系：
|z_i - z_j| > 0.05m

L = L_3D + λ_order L_order
λ_order = 0.001
```

## 讲解稿

```text
我们不要求 DLST 回归绝对 metric depth，因为单目深度尺度本身不稳定。
这里只监督 R 的符号是否和 GT 前后顺序一致。
0.05m 是用来过滤几乎同层的关节对，避免把前后不明显的 pair 加进监督。
```

---

# Slide 9：当前训练结果

## 标题

```text
阶段性结果：排序结构已稳定学到，MPJPE 趋势积极
```

## 页面内容

当前日志：

```text
logs/ConPose@07.05.2026-10:46:00/out.txt
```

| Epoch | P1 MPJPE | P2 | val order loss | val sign acc | entropy | usage min/max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46.3 | 38.1 | 0.1957 | 0.9356 | 0.3785 | 0.205 / 0.298 |
| 3 | 40.4 | 33.8 | 0.1699 | 0.9487 | 0.3533 | 0.206 / 0.297 |
| 7 | 39.0 | 32.4 | 0.1609 | 0.9528 | 0.3589 | 0.216 / 0.287 |

## 讲解稿

```text
当前训练还没有完成，但内部诊断已经很稳定。
val sign acc 从第 1 个 epoch 就超过 0.93，第 7 个 epoch 达到约 0.953，说明 R 和 GT 前后关系高度一致。
entropy 和 layer usage 都正常，说明 assignment 没有塌缩。
MPJPE 第 7 个 epoch 已经到 39.0mm，接近本地 ASFnet 复现线，后续需要等完整训练。
```

---

# Slide 10：当前结论与下一步

## 标题

```text
当前结论与后续验证
```

## 当前可以说

```text
DLST 的中间结构确实学到了稳定的关节前后排序。
AMS 中的隐式 depth evidence 可以被组织成显式 skeleton front-back layout。
```

## 还不能说满

```text
DLST 最终一定超过 UDE / PoseAlign。
训练尚未完成，仍需完整主表和消融。
```

## 下一步

```text
1. 跑完 30 epoch，目标 P1 <= 38.8mm
2. w/o order loss：验证排序监督是否必要
3. rel_depth zero / shuffle：验证 R 是否真的被 pose branch 使用
4. per-joint MPJPE：重点看 wrist / elbow / knee / ankle
5. per-action MPJPE：重点看遮挡动作
```

## 结束语

```text
DLST 的贡献点不是“再加一个 Transformer”，而是提出一个显式、可监督、可解释的 depth-layer sorting 中间结构，
把 AMS 的局部深度证据转化为全身一致的前后布局。
```

---

# 最短 90 秒讲稿

```text
我们这次想解决的问题是：AMS 虽然能为每个关节采到 RGB-depth 局部证据，但它输出的是 128 维高维 token，并没有显式表达关节之间谁在前谁在后。

为了验证这个问题，我们先做了诊断脚本。结果显示，4 个有序深度层理论上可以覆盖 87.2% 的真实前后关系；AMS token 中确实有可解码的层次线索，相邻层准确率达到 80.4%；但原始 F_d 特征距离和真实深度差相关性只有 0.103，原 ASFnet/UDE 的 coarse depth 在易混淆关节对上只有 0.551，接近随机。

所以 DLST 的动机不是说明 AMS 不行，而是说明 AMS 有深度原材料，但缺少结构化组织。DLST 做的是把 joint tokens 软分配到少量有序 depth layers，再通过 R=AΩA^T 得到全局一致的相对深度矩阵，并把 R 作为 attention bias 引导 depth token refinement。

目前训练还没完成，但第 7 个 epoch 验证集 sign accuracy 已经到 0.953，assignment 没有塌缩，MPJPE 到 39.0mm，趋势比较积极。下一步会补完整训练、w/o order loss 和 rel_depth zero/shuffle 消融，证明这个排序结构确实对最终 pose 有贡献。
```
