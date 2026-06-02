#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
CONFIG="${CONFIG:-experiments/human36m/human36m_single.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoint/h36m_v2b.bin}"
IMAGE_ROOT="${IMAGE_ROOT:-../H36M-Toolbox/images_crop}"
DEPTH_ROOT="${DEPTH_ROOT:-../H36M-Toolbox/depth_images}"
DEPTH_FORMAT="${DEPTH_FORMAT:-image}"
OUT="${OUT:-debug_vis/asfnet_ams_forward_hook_sampling.json}"
VIS_DIR="${VIS_DIR:-debug_vis/asfnet_ams_forward_hook_sampling_vis}"
MAX_FRAMES="${MAX_FRAMES:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"

CUDA_VISIBLE_DEVICES="${GPU}" python tools/diagnose_asfnet_ams_forward_hooks.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --image-root "${IMAGE_ROOT}" \
  --depth-root "${DEPTH_ROOT}" \
  --depth-format "${DEPTH_FORMAT}" \
  --max-frames "${MAX_FRAMES}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --out "${OUT}" \
  --vis-dir "${VIS_DIR}"
