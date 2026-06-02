#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
CONFIG="${CONFIG:-experiments/human36m/human36m_single_rgbflow_mfce_separate.yaml}"
LOGDIR="${LOGDIR:-logs_mfce_separate}"
MASTER_PORT="${MASTER_PORT:-2346}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  --logdir "${LOGDIR}"
