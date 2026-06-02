#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/SIMON/26HPE/ASFnet"
python_bin="/home/SIMON/miniconda3/envs/asfnet/bin/python"
crop_cmd="generate_h36m_crop_val_only.py"
crop_dir="$repo_root/H36M-Toolbox/images_crop"
depth_dir="$repo_root/H36M-Toolbox/depth_images"
depth_anything_root="$repo_root/Depth-Anything-V2"
target_count="${1:-543344}"
gpu_id="${2:-6}"
depth_batch_size="${DEPTH_BATCH_SIZE:-0}"
depth_num_workers="${DEPTH_NUM_WORKERS:-8}"
depth_amp="${DEPTH_USE_AMP:-1}"

while pgrep -af "$crop_cmd" >/dev/null 2>&1; do
  count=$(find "$crop_dir" -type f | wc -l)
  printf "%s waiting_for_crop count=%s target=%s\n" "$(date -u +%F_%T)" "$count" "$target_count"
  sleep 30
done

final_count=$(find "$crop_dir" -type f | wc -l)
printf "%s crop_finished count=%s target=%s\n" "$(date -u +%F_%T)" "$final_count" "$target_count"

if [ "$final_count" -lt "$target_count" ]; then
  printf "%s crop_incomplete aborting_depth\n" "$(date -u +%F_%T)"
  exit 1
fi

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"

printf "%s starting_depth gpu=%s batch_size=%s num_workers=%s amp=%s\n" \
  "$(date -u +%F_%T)" "$gpu_id" "$depth_batch_size" "$depth_num_workers" "$depth_amp"

depth_cmd=(
  "$python_bin" -u DCFormer/generate_h36m_depth_v2_adapter.py
  --input-dir "$crop_dir"
  --output-dir "$depth_dir"
  --depth-anything-root "$depth_anything_root"
  --encoder vitb
  --skip-existing
  --batch-size "$depth_batch_size"
  --num-workers "$depth_num_workers"
)

if [ "$depth_amp" = "1" ]; then
  depth_cmd+=(--amp)
fi

exec "${depth_cmd[@]}"
