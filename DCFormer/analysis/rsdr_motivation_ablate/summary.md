# RSDR Motivation Diagnostics

- config: `experiments/human36m/human36m_single_dlst.yaml`
- checkpoint: `logs/ConPose@07.05.2026-10:46:00/checkpoints/best_epoch.bin`
- samples: 1024
- groups: 8

## Group Probe Spread

- pairwise all ordinal acc range: 0.0393
- absolute z R2 range: 0.0366

## Channel Group Ablation

- baseline MPJPE: 40.481 mm
- max delta MPJPE: +1.287 mm
- min delta MPJPE: +0.058 mm
