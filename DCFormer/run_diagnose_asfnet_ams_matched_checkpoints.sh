#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
MAX_FRAMES="${MAX_FRAMES:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"

for item in \
  "h36m_v2b:checkpoint/h36m_v2b.bin" \
  "h36m_v2l:checkpoint/h36m_v2l.bin" \
  "h36m_v2s:checkpoint/h36m_v2s.bin" \
  "log_21may:logs/ConPose@21.05.2026-01:16:34/checkpoints/best_epoch.bin" \
  "log_28mar:logs/ConPose@28.03.2026-13:20:15/checkpoints/best_epoch.bin"
do
  name="${item%%:*}"
  ckpt="${item#*:}"
  CHECKPOINT="${ckpt}" \
  OUT="debug_vis/asfnet_ams_forward_hook_sampling_${name}.json" \
  VIS_DIR="debug_vis/asfnet_ams_forward_hook_sampling_${name}_vis" \
  GPU="${GPU}" \
  MAX_FRAMES="${MAX_FRAMES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  bash run_diagnose_asfnet_ams_forward_hooks.sh
done
