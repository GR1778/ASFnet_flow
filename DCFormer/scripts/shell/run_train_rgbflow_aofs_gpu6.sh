#!/usr/bin/env bash
set -euo pipefail

cd /home/SIMON/26HPE/ASFnet/DCFormer

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=2345 \
  train.py \
  --config experiments/human36m/human36m_single_rgbflow_aofs_clip10.yaml \
  --logdir ./logs_aofs
