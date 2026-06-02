#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/SIMON/26HPE/ASFnet"
dcformer_root="$repo_root/DCFormer"
python_bin="/home/SIMON/miniconda3/envs/asfnet/bin/python"
crop_dir="$repo_root/H36M-Toolbox/images_crop"
depth_dir="$repo_root/H36M-Toolbox/depth_images"
depth_anything_root="$repo_root/Depth-Anything-V2"
gpu_id="${1:-6}"
depth_batch_size="${DEPTH_BATCH_SIZE:-64}"
depth_num_workers="${DEPTH_NUM_WORKERS:-8}"
depth_amp="${DEPTH_USE_AMP:-1}"
poll_seconds="${POLL_SECONDS:-60}"
val_target_count="${VAL_TARGET_COUNT:-0}"
full_target_count="${FULL_TARGET_COUNT:-0}"
val_subjects="${VAL_SUBJECTS:-9,11}"

chain_log="${CHAIN_LOG:-$dcformer_root/logs_h36m_full_auto_chain.log}"
crop_log="${CROP_LOG:-$dcformer_root/logs_h36m_train_crop_auto.log}"
eval_log="${EVAL_LOG:-$dcformer_root/logs_h36m_official_eval_auto.log}"
eval_config_root="${EVAL_CONFIG_ROOT:-/tmp/asfnet_official_eval_configs}"
eval_master_port="${EVAL_MASTER_PORT:-2345}"
eval_batch_sizes="${EVAL_BATCH_SIZES:-256 128 64 32 16}"
val_depth_log="${VAL_DEPTH_LOG:-$dcformer_root/logs_h36m_val_depth_auto.log}"
full_depth_log="${FULL_DEPTH_LOG:-$dcformer_root/logs_h36m_full_depth_auto.log}"

mkdir -p "$(dirname "$chain_log")"

log() {
  printf "%s %s\n" "$(date -u +%F_%T)" "$*" | tee -a "$chain_log" >&2
}

count_files() {
  find "$1" -type f | wc -l
}

count_files_for_subjects() {
  local root="$1"
  local subjects_csv="${2:-}"

  if [ -z "$subjects_csv" ]; then
    count_files "$root"
    return
  fi

  local find_args=("$root" -type f "(")
  local first=1
  local subject
  IFS=',' read -r -a subject_array <<< "$subjects_csv"
  for subject in "${subject_array[@]}"; do
    local subject_int
    subject_int=$((subject))
    local subject_pattern
    printf -v subject_pattern "*/s_%02d_*" "$subject_int"
    if [ "$first" -eq 0 ]; then
      find_args+=(-o)
    fi
    find_args+=(-path "$subject_pattern")
    first=0
  done
  find_args+=(")")

  find "${find_args[@]}" | wc -l
}

depth_log_for_stage() {
  local stage_name="$1"
  if [ "$stage_name" = "val_stage" ]; then
    echo "$val_depth_log"
    return
  fi
  echo "$full_depth_log"
}

start_depth_process() {
  local stage_name="$1"
  local subjects_csv="${2:-}"
  local depth_log
  depth_log="$(depth_log_for_stage "$stage_name")"

  local depth_cmd=(
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

  if [ -n "$subjects_csv" ]; then
    local subject
    IFS=',' read -r -a subject_array <<< "$subjects_csv"
    depth_cmd+=(--subjects)
    for subject in "${subject_array[@]}"; do
      depth_cmd+=("$((subject))")
    done
  fi

  : > "$depth_log"
  (
    cd "$repo_root"
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    exec "${depth_cmd[@]}"
  ) >"$depth_log" 2>&1 &
  local depth_pid=$!
  log "starting_depth stage=$stage_name pid=$depth_pid gpu=$gpu_id batch_size=$depth_batch_size num_workers=$depth_num_workers amp=$depth_amp subjects=${subjects_csv:-all} log=$depth_log"
}

ensure_depth_at_least() {
  local target_count="$1"
  local stage_name="$2"
  local subjects_csv="${3:-}"

  while true; do
    local current_depth_count
    local current_crop_count
    current_depth_count="$(count_files_for_subjects "$depth_dir" "$subjects_csv")"
    current_crop_count="$(count_files_for_subjects "$crop_dir" "$subjects_csv")"

    log "$stage_name depth_count=$current_depth_count target=$target_count crop_count=$current_crop_count subjects=${subjects_csv:-all}"

    if [ "$current_depth_count" -ge "$target_count" ]; then
      if pgrep -af "DCFormer/generate_h36m_depth_v2_adapter.py" >/dev/null 2>&1; then
        log "$stage_name target_reached_waiting_depth_process_exit"
        sleep "$poll_seconds"
        continue
      fi
      log "$stage_name depth_complete"
      break
    fi

    if pgrep -af "DCFormer/generate_h36m_depth_v2_adapter.py" >/dev/null 2>&1; then
      sleep "$poll_seconds"
      continue
    fi

    start_depth_process "$stage_name" "$subjects_csv"
    sleep "$poll_seconds"
  done
}

start_train_crop() {
  if pgrep -af "generate_h36m_crop_full_adapter.py.*--splits train" >/dev/null 2>&1; then
    local existing_pid
    existing_pid="$(pgrep -f "generate_h36m_crop_full_adapter.py.*--splits train" | head -n 1)"
    log "train_crop_already_running pid=$existing_pid"
    echo "$existing_pid"
    return
  fi

  log "starting_train_crop"
  (
    cd "$dcformer_root"
    exec "$python_bin" -u generate_h36m_crop_full_adapter.py --splits train >"$crop_log" 2>&1
  ) &
  local crop_pid=$!
  log "train_crop_started pid=$crop_pid log=$crop_log"
  echo "$crop_pid"
}

ensure_crop_at_least() {
  local target_count="$1"
  local stage_name="$2"

  while true; do
    local current_crop_count
    current_crop_count="$(count_files "$crop_dir")"
    log "$stage_name crop_count=$current_crop_count target=$target_count"

    if [ "$current_crop_count" -ge "$target_count" ]; then
      log "$stage_name crop_complete"
      break
    fi

    if pgrep -af "generate_h36m_crop_full_adapter.py.*--splits train" >/dev/null 2>&1; then
      sleep "$poll_seconds"
      continue
    fi

    log "$stage_name crop_process_missing restarting"
    start_train_crop >/dev/null
    sleep "$poll_seconds"
  done
}

run_official_eval() {
  mkdir -p "$eval_config_root"
  touch "$eval_log"

  local base_config="$dcformer_root/experiments/human36m/human36m_single.yaml"
  local batch_sizes=($eval_batch_sizes)

  for batch_size in "${batch_sizes[@]}"; do
    local config_path="$eval_config_root/human36m_single_eval_bs${batch_size}.yaml"
    "$python_bin" - "$base_config" "$config_path" "$batch_size" <<'PY'
import sys

src, dst, batch_size = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []
in_val = False
updated = False
for line in lines:
    stripped = line.lstrip()
    if line.startswith("val:"):
        in_val = True
        out.append(line)
        continue

    if in_val and line and not line.startswith(" ") and not line.startswith("\t"):
        in_val = False

    if in_val and stripped.startswith("batch_size:"):
        indent = line[: len(line) - len(stripped)]
        out.append(f"{indent}batch_size: {batch_size}\n")
        updated = True
        continue

    out.append(line)

if not updated:
    raise RuntimeError("Failed to update val.batch_size in config")

with open(dst, "w", encoding="utf-8") as f:
    f.writelines(out)
PY

    log "starting_official_eval gpu=$gpu_id batch_size=$batch_size config=$config_path log=$eval_log launch=torch.distributed.launch master_port=$eval_master_port"
    if (
      cd "$dcformer_root"
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      "$python_bin" -u -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port="$eval_master_port" \
        train.py \
        --config "$config_path" \
        --logdir "./logs_auto_official_eval_bs${batch_size}" \
        --eval
    ) >>"$eval_log" 2>&1; then
      log "official_eval_finished batch_size=$batch_size log=$eval_log"
      return 0
    fi

    log "official_eval_attempt_failed batch_size=$batch_size log=$eval_log"
  done

  log "official_eval_failed_all_attempts log=$eval_log"
  return 1
}

main() {
  log "chain_start gpu=$gpu_id"

  local initial_crop_count
  initial_crop_count="$(count_files "$crop_dir")"
  log "initial_crop_count=$initial_crop_count"

  local val_stage_target="$initial_crop_count"
  if [ "$val_target_count" -gt 0 ]; then
    val_stage_target="$val_target_count"
  fi
  log "val_stage_target=$val_stage_target"

  ensure_depth_at_least "$val_stage_target" "val_stage" "$val_subjects"

  if ! run_official_eval; then
    log "continuing_after_eval_failure"
  fi

  local full_stage_target
  full_stage_target="$(count_files "$crop_dir")"
  if [ "$full_target_count" -gt 0 ]; then
    full_stage_target="$full_target_count"
  fi
  log "full_stage_target=$full_stage_target"

  ensure_crop_at_least "$full_stage_target" "train_crop"

  local full_crop_count="$full_stage_target"
  log "full_crop_count=$full_crop_count"

  ensure_depth_at_least "$full_crop_count" "full_stage"

  log "chain_done"
}

main "$@"
