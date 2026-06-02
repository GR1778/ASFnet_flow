#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/SIMON/26HPE/ASFnet"
dcformer_root="$repo_root/DCFormer"
python_bin="/home/SIMON/miniconda3/envs/asfnet/bin/python"

raw_image_dir="$repo_root/H36M-Toolbox/images"
flow_dir="${FLOW_OUTPUT_DIR:-$repo_root/H36M-Toolbox/flow_images_float}"
vis_dir="${FLOW_VIS_DIR:-}"

sea_raft_root="$repo_root/third_party/SEA-RAFT"
sea_raft_cfg="$sea_raft_root/config/eval/spring-M.json"
sea_raft_ckpt="$sea_raft_root/models/Tartan-C-T-TSKH-spring540x960-M.pth"

train_labels="$dcformer_root/data/h36m_train.pkl"
val_labels="$dcformer_root/data/h36m_validation.pkl"

gpu_train_a="${FLOW_GPU_TRAIN_A:-2}"
gpu_train_b="${FLOW_GPU_TRAIN_B:-3}"
gpu_train_c="${FLOW_GPU_TRAIN_C:-4}"
gpu_val="${FLOW_GPU_VAL:-4}"

batch_size="${FLOW_BATCH_SIZE:-4}"
amp="${FLOW_USE_AMP:-1}"
skip_existing="${FLOW_SKIP_EXISTING:-1}"
frame_gap="${FLOW_FRAME_GAP:-4}"
suppress_threshold="${FLOW_SUPPRESS_THRESHOLD:-0.2}"
clip_flow="${FLOW_CLIP_FLOW:-5.0}"
dtype="${FLOW_DTYPE:-float16}"
output_format="${FLOW_OUTPUT_FORMAT:-npy}"
log_dir="${FLOW_LOG_DIR:-$dcformer_root/logs_flow_searaft_full}"

mkdir -p "$log_dir"
started_pids=()

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

start_flow() {
  local name="$1"
  local gpu="$2"
  local labels="$3"
  shift 3
  local subjects=("$@")
  local log_path="$log_dir/${name}.log"

  local cmd=(
    "$python_bin" -u "$dcformer_root/generate_h36m_flow_samecrop_adapter.py"
    --labels "$labels"
    "${common_args[@]}"
    --subjects "${subjects[@]}"
  )

  printf "%s start name=%s gpu=%s labels=%s subjects=%s log=%s\n" \
    "$(date -u +%F_%T)" "$name" "$gpu" "$labels" "${subjects[*]}" "$log_path"

  (
    cd "$dcformer_root"
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "${cmd[@]}"
  ) >"$log_path" 2>&1 &
  local pid="$!"
  started_pids+=("$pid")
  printf "%s pid=%s name=%s\n" "$(date -u +%F_%T)" "$pid" "$name"
}

start_flow "train_s1_s5" "$gpu_train_a" "$train_labels" 1 5
start_flow "train_s6_s7" "$gpu_train_b" "$train_labels" 6 7
start_flow "train_s8" "$gpu_train_c" "$train_labels" 8

printf "%s waiting for train flow jobs: %s\n" "$(date -u +%F_%T)" "${started_pids[*]}"
wait "${started_pids[@]}"
printf "%s train flow jobs finished\n" "$(date -u +%F_%T)"

started_pids=()
start_flow "val_s9_s11" "$gpu_val" "$val_labels" 9 11
printf "%s waiting for validation flow job: %s\n" "$(date -u +%F_%T)" "${started_pids[*]}"
wait "${started_pids[@]}"
printf "%s validation flow job finished\n" "$(date -u +%F_%T)"

printf "%s launched all SEA-RAFT flow jobs. logs=%s output=%s\n" \
  "$(date -u +%F_%T)" "$log_dir" "$flow_dir"
printf "Use: tail -f %s/*.log\n" "$log_dir"
