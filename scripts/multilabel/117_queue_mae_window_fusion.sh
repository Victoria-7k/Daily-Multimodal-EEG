#!/usr/bin/env bash
set -euo pipefail

ROOT="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/outputs/mae_window_fusion_20260928"
LOG="$OUT/logs/117_queue_${STAMP}.log"
mkdir -p "$OUT/logs" "$OUT/smoke" "$OUT/formal" "$OUT/predictions"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run() {
  printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

printf 'queue_started_utc=%s\n' "$(date -u +%FT%TZ)" | tee -a "$LOG"
printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"

# Phase 0: all three matched token compositions, one fixed downstream seed.
run "$PY" "$ROOT/scripts/multilabel/116_run_mae_window_fusion.py" \
  --protocols cross_day --conditions B0,E1,W1 --seeds 240800 \
  --epochs 3 --patience 3 --batch-size 256 --device cuda \
  --out-json "$OUT/smoke/results.json" --out-md "$OUT/smoke/results.md" \
  --predictions-dir "$OUT/smoke/predictions"
touch "$OUT/SMOKE_COMPLETE"

# Phase 1: matched 3-seed formal matrix.  No VideoMAE-dependent conditions run here.
run "$PY" "$ROOT/scripts/multilabel/116_run_mae_window_fusion.py" \
  --protocols cross_day,within_subject_day --conditions B0,E1,W1 --seeds 240800,240801,240802 \
  --epochs 80 --patience 15 --batch-size 256 --device cuda \
  --out-json "$OUT/formal/results.json" --out-md "$OUT/formal/results.md" \
  --predictions-dir "$OUT/predictions"
touch "$OUT/FORMAL_COMPLETE"
printf 'queue_completed_utc=%s\n' "$(date -u +%FT%TZ)" | tee -a "$LOG"
