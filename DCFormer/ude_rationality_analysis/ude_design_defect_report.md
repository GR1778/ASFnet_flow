# UDE 设计缺陷诊断报告

- 时间: `2026-04-22T04:49:35`
- 样本数: `256`
- 检查点: `/home/SIMON/26HPE/ASFnet/DCFormer/checkpoint/h36m_v2b.bin`

## 关键指标（均值）
- `attn_diag_ratio`: 1.0454
- `attn_diag_weight`: 0.0613
- `attn_offdiag_weight`: 0.0587
- `delta_mu_shuffle_mm`: 0.1637
- `delta_s_shuffle_mm`: 2.1029
- `delta_s_zero_mm`: 1.1863
- `mpjpe_base_mm`: 41.4036
- `mpjpe_mu_shuffle_mm`: 41.5673
- `mpjpe_s_shuffle_mm`: 43.5064
- `mpjpe_s_zero_mm`: 42.5899
- `s_error_corr`: 0.0192
- `s_gate_corr`: 0.3924

## 设计缺陷命中
- [高] `不确定性单调抑制缺失`: s-门控相关系数=0.392，未体现“高不确定性应更强抑制”的负相关趋势。
- [中] `不确定性与误差关联弱`: s-最终误差相关系数=0.019，不确定性信息性较弱。

## 解读
- `delta_s_shuffle_mm / delta_s_zero_mm` 越小，说明 s 对最终输出越不具因果性。
- `delta_mu_shuffle_mm` 远大于 `delta_s_shuffle_mm`，说明系统更依赖 mu 而非 s。
- `s_gate_corr` 若不显著为负，说明“高不确定->低权重”的结构单调性不足。
- `s_error_corr` 接近 0 表明不确定性未学成有效误差代理。
