#!/usr/bin/env bash
set -euo pipefail

cd /home/SIMON/26HPE/ASFnet/DCFormer

GPU="${GPU:-6}"
CONFIG="${CONFIG:-experiments/human36m/human36m_single_rgbflow_mces_clip10.yaml}"
LOGDIR="${LOGDIR:-logs_mces}"
MASTER_PORT="${MASTER_PORT:-2347}"
PYTHON_BIN="${PYTHON_BIN:-/home/SIMON/miniconda3/envs/asfnet/bin/python}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG}" \
  --logdir "${LOGDIR}"
