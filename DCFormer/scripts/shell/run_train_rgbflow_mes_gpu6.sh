#!/usr/bin/env bash
set -euo pipefail

cd /home/SIMON/26HPE/ASFnet/DCFormer

CUDA_VISIBLE_DEVICES=6 python train.py \
  --config experiments/human36m/human36m_single_rgbflow_mes_clip10.yaml \
  --logdir logs_mes
