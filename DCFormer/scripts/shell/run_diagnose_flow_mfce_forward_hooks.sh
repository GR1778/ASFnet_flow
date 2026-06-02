#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
CONFIG="${CONFIG:-experiments/human36m/human36m_single_rgbflow_mfce_separate.yaml}"
CHECKPOINT="${CHECKPOINT:-logs_mfce_separate/ConPose@24.05.2026-08:43:51/checkpoints/best_epoch.bin}"
FLOW_DIR="${FLOW_DIR:-../H36M-Toolbox/flow_images_float}"
IMAGE_ROOT="${IMAGE_ROOT:-../H36M-Toolbox/images_crop}"
OUT="${OUT:-debug_vis/flow_mfce_forward_hook_sampling.json}"
VIS_DIR="${VIS_DIR:-debug_vis/flow_mfce_forward_hook_sampling_vis}"
MAX_FRAMES="${MAX_FRAMES:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FLOW_CLIP="${FLOW_CLIP:-5}"
FLOW_NORM="${FLOW_NORM:-5}"

CUDA_VISIBLE_DEVICES="${GPU}" python tools/diagnose_flow_mfce_forward_hooks.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --flow-dir "${FLOW_DIR}" \
  --image-root "${IMAGE_ROOT}" \
  --flow-clip "${FLOW_CLIP}" \
  --flow-norm "${FLOW_NORM}" \
  --max-frames "${MAX_FRAMES}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --out "${OUT}" \
  --vis-dir "${VIS_DIR}"
