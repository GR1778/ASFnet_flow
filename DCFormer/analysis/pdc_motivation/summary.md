# PDC Motivation Diagnostics

- config: `experiments/human36m/human36m_single_dlst.yaml`
- checkpoint: `logs/ConPose@07.05.2026-10:46:00/checkpoints/best_epoch.bin`
- samples: 1024
- stages: 5

## Stage Disagreement

- vs joint MPJPE spearman: 0.1673
- vs joint depth error spearman: 0.2042
- vs DLST order error spearman: 0.0988
- high-low joint MPJPE: +14.0018 mm

## First-Last Shift

- vs joint MPJPE spearman: 0.2203
- high-low joint MPJPE: +22.2772 mm
