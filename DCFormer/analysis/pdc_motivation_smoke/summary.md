# PDC Motivation Diagnostics

- config: `experiments/human36m/human36m_single_dlst.yaml`
- checkpoint: `logs/ConPose@07.05.2026-10:46:00/checkpoints/best_epoch.bin`
- samples: 64
- stages: 5

## Stage Disagreement

- vs joint MPJPE spearman: 0.1148
- vs joint depth error spearman: 0.1565
- vs DLST order error spearman: -0.0086
- high-low joint MPJPE: +4.2989 mm

## First-Last Shift

- vs joint MPJPE spearman: 0.1594
- high-low joint MPJPE: +12.1737 mm
