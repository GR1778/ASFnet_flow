# UDE 正常推理诊断报告

- 结论: **不合理**
- 综合分数: **25.70/100**
- 时间: `2026-04-22T04:44:01`

## 执行信息
- 命令: `python3 tools/analyze_depth_utilization.py --config experiments/human36m/human36m_single.yaml --checkpoint /home/SIMON/26HPE/ASFnet/DCFormer/checkpoint/h36m_v2b.bin --num_samples 256 --batch_size 32 --device cuda:0`
- 返回码: `0`

## 关键原始指标
- `affine_r2_mean`: 0.233700
- `alpha_mean`: 1.291800
- `alpha_std`: 1.809200
- `beta_mean`: -4.210500
- `beta_std`: 8.379200
- `depth_at_joint_mean`: 69.383100
- `grad_mag_mean`: 13.938500
- `grad_mag_std`: 33.435600
- `joint_depth_variance`: 0.482800
- `metric_bone_depth_error_mm`: 0.277500
- `ordinal_accuracy`: 0.605000

## 维度评分
- `uncertainty_calibration`: 0.00
  - corr=0.000, high_unc_err=0.0000, low_unc_err=0.0000
- `ordinal_metric_consistency`: 23.00
  - ordinal_acc=0.605, affine_r2=0.234, metric_bone_depth_error=0.2775
- `affine_stability`: 19.00
  - alpha=1.292±1.809, beta=-4.210±8.379
- `joint_depth_structure`: 100.00
  - grad_mean=13.9385, grad_std=33.4356, joint_depth_variance=0.4828, depth_at_joint_mean=69.3831

## 代码一致性风险
- [high] `mvn/models/DGLifting.py` `umap_softmax_dim_over_joints`: UMap uses softmax over joint dimension (dim=1). Paper-style UMap usually normalizes feature channel dimension.
- [high] `mvn/models/loss.py` `missing_log_variance_clamp`: No explicit clamp for log-variance s was found in depth uncertainty loss.
- [medium] `tools/analyze_depth_utilization.py` `sigma_conversion_mismatch`: Diagnostic computes sigma as exp(s), while common heteroscedastic form uses sigma=exp(s/2).
- [medium] `train.py` `hardcoded_depth_loss_weight`: Depth uncertainty loss weight is hard-coded (1e-5) in training loop.

## 判定建议
- 本报告仅基于真实输入下的正常推理，不包含随机/零深度测试。
- 若 `uncertainty_error_correlation <= 0`，说明 UDE 不确定性分支未校准。
- 若 `affine_r2_mean` 偏低且 `alpha_std` 偏高，说明深度映射稳定性不足。
