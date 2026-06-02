# UDE 设计缺陷诊断报告

- 时间: `2026-05-07T07:46:43`
- 样本数: `512`
- 检查点: `checkpoint/h36m_v2b.bin`

## 关键指标（均值）
- `attn_diag_ratio`: 1.0419
- `attn_diag_weight`: 0.0611
- `attn_offdiag_weight`: 0.0587
- `delta_mu_shuffle_mm`: 0.0704
- `delta_s_shuffle_mm`: 1.9252
- `delta_s_zero_mm`: 1.0155
- `mpjpe_base_mm`: 39.9350
- `mpjpe_mu_shuffle_mm`: 40.0054
- `mpjpe_s_shuffle_mm`: 41.8602
- `mpjpe_s_zero_mm`: 40.9506
- `s_error_corr`: 0.0120
- `s_gate_corr`: 0.4277

## 设计缺陷命中
- [高] `不确定性单调抑制缺失`: s-门控相关系数=0.428，未体现“高不确定性应更强抑制”的负相关趋势。
- [中] `不确定性与误差关联弱`: s-最终误差相关系数=0.012，不确定性信息性较弱。

## 解读
- `delta_s_shuffle_mm / delta_s_zero_mm` 越小，说明 s 对最终输出越不具因果性。
- `delta_mu_shuffle_mm` 远大于 `delta_s_shuffle_mm`，说明系统更依赖 mu 而非 s。
- `s_gate_corr` 若不显著为负，说明“高不确定->低权重”的结构单调性不足。
- `s_error_corr` 接近 0 表明不确定性未学成有效误差代理。
