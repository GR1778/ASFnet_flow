#!/usr/bin/env bash
set -euo pipefail

target="${1:-543344}"
pid="${2:-2418145}"
interval="${3:-15}"
log_file="${4:-/home/SIMON/26HPE/ASFnet/DCFormer/logs_h36m_stop_at_val.log}"
crop_dir="${5:-/home/SIMON/26HPE/ASFnet/H36M-Toolbox/images_crop}"

mkdir -p "$(dirname "$log_file")"

while kill -0 "$pid" 2>/dev/null; do
  count=$(find "$crop_dir" -type f | wc -l)
  printf "%s count=%s target=%s\n" "$(date -u +%F_%T)" "$count" "$target" >> "$log_file"

  if [ "$count" -ge "$target" ]; then
    printf "%s stopping PID=%s at count=%s\n" "$(date -u +%F_%T)" "$pid" "$count" >> "$log_file"
    kill -INT "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    exit 0
  fi

  sleep "$interval"
done

printf "%s monitor_exit pid_not_running\n" "$(date -u +%F_%T)" >> "$log_file"
