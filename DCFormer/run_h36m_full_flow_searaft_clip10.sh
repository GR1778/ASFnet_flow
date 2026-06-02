#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/SIMON/26HPE/ASFnet"
dcformer_root="$repo_root/DCFormer"
python_bin="/home/SIMON/miniconda3/envs/asfnet/bin/python"

raw_image_dir="$repo_root/H36M-Toolbox/images"
flow_dir="${FLOW_OUTPUT_DIR:-$repo_root/H36M-Toolbox/flow_images_float_clip10}"
vis_dir="${FLOW_VIS_DIR:-}"

sea_raft_root="$repo_root/third_party/SEA-RAFT"
sea_raft_cfg="$sea_raft_root/config/eval/spring-M.json"
sea_raft_ckpt="$sea_raft_root/models/Tartan-C-T-TSKH-spring540x960-M.pth"

train_labels="$dcformer_root/data/h36m_train.pkl"
val_labels="$dcformer_root/data/h36m_validation.pkl"

gpu="${FLOW_GPU:-6}"
batch_size="${FLOW_BATCH_SIZE:-4}"
amp="${FLOW_USE_AMP:-1}"
skip_existing="${FLOW_SKIP_EXISTING:-1}"
frame_gap="${FLOW_FRAME_GAP:-4}"
suppress_threshold="${FLOW_SUPPRESS_THRESHOLD:-0.2}"
clip_flow="${FLOW_CLIP_FLOW:-10.0}"
dtype="${FLOW_DTYPE:-float16}"
output_format="${FLOW_OUTPUT_FORMAT:-npy}"
log_dir="${FLOW_LOG_DIR:-$dcformer_root/logs_flow_searaft_full_clip10}"

mkdir -p "$log_dir"

common_args=(
  --backend searaft
  --sea-raft-dir "$sea_raft_root"
  --sea-raft-cfg "$sea_raft_cfg"
  --sea-raft-ckpt "$sea_raft_ckpt"
  --input-dir "$raw_image_dir"
  --output-dir "$flow_dir"
  --output-format "$output_format"
  --dtype "$dtype"
  --frame-gap "$frame_gap"
  --suppress-small-flow
  --suppress-threshold "$suppress_threshold"
  --clip-flow "$clip_flow"
  --batch-size "$batch_size"
)

if [ "$amp" = "1" ]; then
  common_args+=(--amp)
fi

if [ "$skip_existing" = "1" ]; then
  common_args+=(--skip-existing)
fi

if [ -n "$vis_dir" ]; then
  common_args+=(--vis-dir "$vis_dir" --vis-stride "${FLOW_VIS_STRIDE:-5000}")
fi

run_flow() {
  local name="$1"
  local labels="$2"
  shift 2
  local subjects=("$@")
  local log_path="$log_dir/${name}.log"

  printf "%s start name=%s gpu=%s labels=%s subjects=%s log=%s\n" \
    "$(date -u +%F_%T)" "$name" "$gpu" "$labels" "${subjects[*]}" "$log_path"

  (
    cd "$dcformer_root"
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$python_bin" -u "$dcformer_root/generate_h36m_flow_samecrop_adapter.py" \
      --labels "$labels" \
      "${common_args[@]}" \
      --subjects "${subjects[@]}"
  ) >"$log_path" 2>&1

  printf "%s finished name=%s\n" "$(date -u +%F_%T)" "$name"
}

run_flow "train_s1_s5_s6_s7_s8" "$train_labels" 1 5 6 7 8
run_flow "val_s9_s11" "$val_labels" 9 11

printf "%s finished all SEA-RAFT clip10 flow jobs. logs=%s output=%s\n" \
  "$(date -u +%F_%T)" "$log_dir" "$flow_dir"
printf "Use: tail -f %s/*.log\n" "$log_dir"
