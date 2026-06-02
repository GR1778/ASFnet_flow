#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-1,2}"
CONFIG="${CONFIG:-experiments/human36m/human36m_single_rgbflow_cmff_accum2.yaml}"
LOGDIR="${LOGDIR:-logs_cmff_aligned_accum2_gpu12}"
MASTER_PORT="${MASTER_PORT:-2348}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  --logdir "${LOGDIR}"
