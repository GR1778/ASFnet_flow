#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
CONFIG="${CONFIG:-experiments/human36m/human36m_single_rgbflow_mfas_clip10.yaml}"
LOGDIR="${LOGDIR:-logs_mfas}"
MASTER_PORT="${MASTER_PORT:-2356}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  --logdir "${LOGDIR}"
