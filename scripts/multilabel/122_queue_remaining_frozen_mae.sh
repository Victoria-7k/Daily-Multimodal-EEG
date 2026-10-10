#!/usr/bin/env bash
set -euo pipefail
ROOT="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
STAGING="${MAE_REMAINING_STAGING:-/home/wangzw/mae_remaining_staging}"
OUT="${MAE_REMAINING_OUT:-/home/wangzw/outputs/mae_remaining_20261007}"
VIDEO_ROOT="${MAE_VIDEO_ROOT:-/home/wangzw/outputs/video_mae_20261007}"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/122_queue_$(date -u +%Y%m%dT%H%M%SZ).log"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
common=(--root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" --mae-root "$ROOT/outputs/mae_20260924" --video-mae-root "$VIDEO_ROOT" --conditions M3,M4,M5-F --device cuda)
run() { "$@" 2>&1 | tee -a "$LOG"; }
run "$PY" "$STAGING/118_run_mae_mt11_event_ablation.py" "${common[@]}" --out-root "$OUT/frozen_smoke" --protocols cross_day --seeds 240800 --epochs 3 --patience 3
touch "$OUT/FROZEN_SMOKE_COMPLETE"
run "$PY" "$STAGING/118_run_mae_mt11_event_ablation.py" "${common[@]}" --out-root "$OUT/frozen_formal" --protocols cross_day,within_subject_day --seeds 240800,240801,240802 --epochs 80 --patience 15
touch "$OUT/FROZEN_FORMAL_COMPLETE"
