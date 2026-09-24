#!/usr/bin/env bash
# Queue the label-free Stage-A MAE runs after the shared H20 is genuinely idle.
set -euo pipefail

ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
OUT_ROOT="$ROOT/outputs/mae_20260924"
LOG_DIR="$OUT_ROOT/logs"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
mkdir -p "$LOG_DIR" "$ROOT/outputs/tmp"
export PYTHONPATH="$ROOT/src"
export TMPDIR="$ROOT/outputs/tmp"
export XDG_CACHE_HOME="$ROOT/outputs/tmp"

wait_for_idle_gpu() {
  local stable=0
  while true; do
    read -r utilization memory <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n 1 | tr ',' ' ')"
    if [[ "${utilization// /}" -le 25 && "${memory// /}" -le 2048 ]]; then
      stable=$((stable + 1))
      if [[ "$stable" -ge 2 ]]; then
        return 0
      fi
    else
      stable=0
    fi
    printf '%s waiting_for_idle_gpu utilization=%s memory_mib=%s\n' "$(date --iso-8601=seconds)" "$utilization" "$memory"
    sleep 300
  done
}

run_one() {
  local modality="$1"
  local protocol="$2"
  printf '%s starting modality=%s protocol=%s\n' "$(date --iso-8601=seconds)" "$modality" "$protocol"
  "$PY" scripts/multilabel/112_run_modality_mae.py \
    --modality "$modality" --protocol "$protocol" --out-root "$OUT_ROOT" --device cuda
  printf '%s completed modality=%s protocol=%s\n' "$(date --iso-8601=seconds)" "$modality" "$protocol"
}

cd "$ROOT"
wait_for_idle_gpu
run_one eeg cross_day
run_one eeg within_subject_day
run_one wear cross_day
run_one wear within_subject_day
printf '%s stage_a_complete\n' "$(date --iso-8601=seconds)" > "$OUT_ROOT/STAGE_A_COMPLETE"
