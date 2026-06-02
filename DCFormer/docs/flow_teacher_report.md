# 光流路线阶段性汇报整理

更新时间：2026-05-29

本文只整理光流这一条路线，不展开 ASFNet 的 depth/AMS/UDE 线。汇报主线建议定为：

> CAPF-style RGB context 主要补充当前帧的空间外观信息，但没有显式建模相邻帧之间的短期运动对应关系。光流能提供这种 pixel-level correspondence cue；真正的问题不是“加一个 flow token”，而是如何在 2D 关节点有误差、局部存在运动边界和遮挡时，从关节邻域中取到可靠的 flow evidence。

## 1. 一句话结论

目前可以向老师汇报：

1. 光流模态本身是有用的。单点 flow token / 简单 flow fusion 已经能把结果推到约 `40.1` 左右，部分变体最好到 `39.8-39.9`。
2. 直接把 RGB/depth 里的 deformable sampling 思路照搬到 flow 上并不稳定。诊断显示，多点采样会在 motion boundary 和局部 flow variance 大的区域混入不同运动来源，尤其影响 wrist、elbow、ankle 等末端关节。
3. 当前更合理的下一步不是继续堆小组件，而是把模块收束为“flow correspondence reliability / consistency sampling”：在当前帧关节邻域内选取运动一致的候选点，并对不可靠 flow 做抑制或回退。

## 2. 已尝试的光流变体

注意：以下实验的训练长度、配置细节还没有完全统一，适合做阶段汇报，不适合直接当最终论文主表。

| 方向 | 核心做法 | 最好结果 | 当前判断 |
|---|---|---:|---|
| CAPF + single-point flow token | flow 经 `Conv2d(2,128)` 编码后，在 2D joint 处单点采样，再 append 到 RGB context tokens | `P1≈40.1`；clip10 rerun 为 `40.7/33.9` | 证明光流模态有价值，但单点受 2D 误差和背景/边界污染 |
| AOFS | 在固定局部 flow 网格上做 joint-conditioned attention 汇聚 | `40.4/33.3` | 比单点更复杂，但没有稳定提升 |
| MFCE unified | 把 flow feature 放进统一 DCE-style deformable sampling | `40.3/33.0` | 能工作，但动机不够干净 |
| MFCE separate | flow 单独一条 DCE-style 分支，再和 RGB/pose 融合 | `39.8/33.0` | 当前数字最好，但诊断显示采样机制仍可能混入不一致 flow |
| LMRS norm5/norm10 | 在多半径局部候选中做 motion-region selection，显式利用 raw flow 与局部区域关系 | `39.9/33.2` | 方向更贴近 flow 特性，但选择分布偏散，还没真正学成尖锐可靠区域 |

对应日志：

```text
logs_flow_capf_clip10/ConPose@26.05.2026-15:47:47/out.txt
logs_aofs/ConPose@26.05.2026-15:45:50/out.txt
logs_mfce_separate/ConPose@25.05.2026-17:33:39/out.txt
logs_mfce_unified/ConPose@25.05.2026-17:34:08/out.txt
logs_lmrs_norm5/ConPose@27.05.2026-15:07:01/out.txt
logs_lmrs_norm10/ConPose@27.05.2026-15:07:46/out.txt
```

## 3. 关键诊断证据

### 3.1 光流确实提供有效运动线索

`debug_vis/flow_reliability_probe_val.json` 显示：

| 指标 | 数值 | 含义 |
|---|---:|---|
| GT joint 位置 raw flow error mean / p50 | `0.824 / 0.333 px` | 真实关节位置处的 flow 与关节运动较接近 |
| CPN joint 位置 raw flow error mean / p50 | `1.366 / 0.926 px` | 检测点偏差会明显损害单点 flow |
| GT motion 与 flow error 相关 | `0.718` | 快速运动更容易出错 |
| patch flow variance 与 error 相关 | `0.359` | 局部运动不一致会降低可靠性 |
| flow edge 与 error 相关 | `0.255` | 运动边界是风险来源 |

最差关节主要是 `RWrist/LWrist/LAnkle/RElbow/RAnkle`，符合“末端关节、高运动、遮挡边界更难”的预期。

### 3.2 单点 flow 的失败模式很明确

可视化目录：

```text
debug_vis/single_point_flow_failure_clip10_vis/
debug_vis/single_point_flow_zero_bg_clip10_vis/
```

典型失败有两类：

1. 2D 关节点落在错误运动区域，采到相邻肢体或背景 flow。
2. 2D 关节点落在背景/无效区域，采到接近 zero flow。

例子：

| Case | 关节 | 单点 flow error | GT-site flow error | 说明 |
|---|---|---:|---:|---|
| Purchases-2, RElbow | RElbow | `8.42` | `1.05` | 2D 点偏离后采到错误运动 |
| Directions-2, RAnkle | RAnkle | `9.38` | `1.86` | 2D 点落到 zero/background flow |

这说明我们需要的是“关节邻域内可靠 correspondence 的选择”，不是盲目信任 detected joint 单点。

### 3.3 直接多点采样会混入不一致运动

MFCE/DCE-style 采样诊断：

```text
debug_vis/flow_feature_sampling_separate_epoch3.json
debug_vis/flow_feature_sampling_unified_epoch3.json
```

关键结果：

| 指标 | separate | unified | 含义 |
|---|---:|---:|---|
| center raw flow error mean | `0.1277` | `0.1277` | 中心点 raw flow 已经较稳定 |
| multi-sampled raw flow error mean | `0.1317` | `0.1317` | 普通多点采样没有比中心更准 |
| multi - center | `+0.0040` | `+0.0040` | 平均还略微变差 |
| feature delta vs patch flow variance | `0.822` | `0.809` | 特征扰动强烈绑定局部 flow variance |
| feature spread vs patch flow variance | `0.835` | `0.831` | 局部运动不一致会放大采样特征分散 |
| feature spread vs flow edge | `0.526` | `0.518` | motion boundary 会导致采样 token 不稳定 |

这组诊断支持：flow 不是普通 RGB/depth feature map。普通 deformable sampling 在 motion boundary 附近可能越采越乱。

### 3.4 3x3 flow embedding 本身也在混邻域

`debug_vis/flow_conv_kernel_effect_mfce_separate.json` 显示：

| 指标 | 数值 | 含义 |
|---|---:|---|
| neighbor / center kernel L2 | `6.63` | `Conv2d(2,128,3)` 强依赖邻域而非中心 |
| full vs center-only feature L2 mean | `0.248` | 仅卷积邻域就会显著改变 flow token |
| full spread increase vs patch flow variance corr | `0.815` | 邻域混合在 flow variance 大时尤其明显 |

这解释了为什么单纯“把 flow 卷成 128 维再采样”会丢掉部分 raw flow 的清晰物理含义。

### 3.5 局部候选有上限，但选择机制还没学好

`debug_vis/flow_sampling_oracle_clip10_4096.json` 显示：

| 指标 | 数值 | 含义 |
|---|---:|---|
| center error mean | `0.721` | 当前单点采样误差 |
| best candidate error mean | `0.370` | 局部候选中存在更好的点 |
| oracle gain mean | `0.350` | 理想选择能显著改善 flow evidence |
| best better than center | `77.0%` | 多数样本存在比中心更好的候选 |
| best scheme counts | local_ring 最多 | 好点多数来自局部环形候选 |

但 LMRS 的选择诊断 `debug_vis/lmrs_alpha_norm5_best.json` 显示：

| 指标 | 数值 | 含义 |
|---|---:|---|
| center weight mean | `0.005` | 模型并不依赖中心 |
| entropy_norm mean | `0.848` | 分布仍偏散 |
| effective_k mean | `55 / 97` | 没有形成清晰 top-k 可靠选择 |
| top1 center frac | `0` | 选择机制完全偏离中心，但未必更可靠 |

所以：候选点空间是有潜力的，问题在于如何用更强的 consistency/reliability 约束学会“选对”。

## 4. 可视化材料建议

汇报时建议放 4 类图：

1. 总体框架图：`output/flow_guided_context_overview.png`
2. 当前 LMRS/flow sampling 模块图：`paper/lmrs_module_ams_style_final_v3.png`
3. 单点 flow 失败 contact sheet：`debug_vis/single_point_flow_failure_clip10_vis/contact_sheet_top8.png`
4. zero/background flow 失败 contact sheet：`debug_vis/single_point_flow_zero_bg_clip10_vis/contact_sheet_top8.png`

可补充：

```text
debug_vis/flow_offset_strategy_vis_val/*.png
debug_vis/learned_flow_dce_sampling_mfce_separate_vis/*.png
debug_vis/flow_mfce_forward_hook_sampling_vis/*.png
debug_vis/flow_conv_effect_s09_09_01_ca02_f1202.png
```

## 5. 给老师汇报时的推荐讲法

可以这样讲：

> 我们现在只沿光流路线看。最开始的想法是把 optical flow 作为短期运动线索接到 CAPF-style RGB context pipeline 里，因为 RGB context 主要描述当前帧外观，而 flow 能提供当前帧到历史帧的像素级对应关系。简单单点 flow token 已经能带来收益，说明 flow 模态本身是有价值的。
>
> 但后面直接做 flow 版 DCE/AMS-style 多点采样时，发现它和 RGB/depth 不一样。flow 是位移向量场，局部邻域可能同时包含当前肢体、相邻肢体、背景、遮挡和运动边界。诊断结果显示，多点采样后的 raw flow error 没有比中心点更好，feature disturbance 和 patch flow variance 的相关性达到 0.8 以上，最受影响的是 wrist、elbow、ankle 这些末端关节。
>
> 所以目前的判断是：光流分支的核心贡献不应该写成“又做了一个 deformable sampling”，而应该写成“在当前关节邻域内做 correspondence-consistent sampling / reliability modeling”。也就是说，候选点是有上限的，oracle 诊断显示 77% 的样本局部有比中心更好的 flow 点；但我们现在的 learned selector 还不够尖锐，LMRS 的有效选择数仍然偏大。下一步要把模块收束到 raw-flow consistency、局部 motion variance 和 reliability gate 上，让模型学会选可靠的 flow，而不是把不同运动混在一起。

## 6. 当前可以讲和不能讲

可以讲：

1. 光流模态有效，能作为 RGB context 的短期 motion correspondence 补充。
2. 单点 flow 的主要问题来自 2D 关节点误差、zero/background flow、motion boundary。
3. 普通 DCE/AMS-style flow sampling 不稳定，诊断已经支持这一点。
4. 局部候选存在明显 oracle 上限，下一步应围绕 consistency/reliability 设计模块。

暂时不要讲：

1. 不要说已经最终优于所有 baseline。当前实验设置还未完全统一。
2. 不要说这是严格 single-frame 方法。光流使用历史帧，更准确是 target-frame lifting with short-term causal motion cue。
3. 不要把 `ref + flow` 写成当前 flow map 上的采样坐标，除非后续真的引入历史帧 RGB feature 做跨帧 warping。
4. 不要把 flow reliability 写成 depth uncertainty 的复制。我们这里建模的是 correspondence reliability。

## 7. 下一步建议

1. 统一训练设置，重跑或补齐：
   - RGB-only CAPF baseline
   - single-point flow token
   - MFCE separate
   - LMRS / consistency sampling
2. 模块收束为 FCAS/CCS：
   - 当前帧关节邻域局部候选；
   - raw flow consistency weighting；
   - patch flow variance / flow edge / candidate consistency 作为 reliability cue；
   - 不可靠时回退到 RGB context 或 pose token。
3. 指标补充：
   - per-joint：重点看 wrist/elbow/ankle；
   - per-action：重点看 Greeting、Sitting、Purchases、Directions、Phoning 等失败案例多的动作；
   - 如果要强调运动稳定性，再补 MPJVE 或 acceleration error。

## 8. 90 秒版本

> 我们这条线现在只看光流。最初我们把 flow 接到 CAPF-style pipeline 里，是因为 RGB context 只能补当前帧空间外观，而 flow 能提供相邻帧的短期对应关系。简单单点 flow token 已经能把结果推到约 40.1 左右，说明 flow 模态是有价值的。
>
> 但直接把 RGB/depth 的 deformable sampling 照搬到 flow 上不稳定。flow 是二维位移场，不是普通特征图；关节附近可能混有相邻肢体、背景、遮挡和运动边界。我们的诊断显示，多点采样的 raw flow error 平均没有优于中心点，feature disturbance 与 patch flow variance 的相关性超过 0.8，最明显的是 wrist、elbow、ankle。单点失败可视化也能看到，2D 点偏一点就可能采到错误运动或 zero background。
>
> 所以当前结论是：光流路线的核心问题不是“再加一个采样模块”，而是“如何在关节邻域内选择可靠的 motion correspondence”。Oracle 诊断显示，77% 的样本局部存在比中心更好的候选，说明上限存在；但 LMRS 的选择分布还偏散，没有真正学会稳定选点。下一步我们准备把方法收束到 correspondence-consistent sampling 和 reliability gate，用 raw flow consistency、局部 variance 和 flow edge 来约束采样与融合。
