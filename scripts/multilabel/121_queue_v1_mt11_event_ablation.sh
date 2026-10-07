#!/usr/bin/env bash
set -euo pipefail
ROOT="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
RUNNER="${V1_RUNNER:-/home/wangzw/v1_staging/118_run_mae_mt11_event_ablation.py}"
VIDEO_ROOT="${V1_VIDEO_ROOT:-/home/wangzw/outputs/video_mae_20261007}"
OUT="${V1_OUT:-/home/wangzw/outputs/mae_mt11_v1_20261007}"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/121_queue_$(date -u +%Y%m%dT%H%M%SZ).log"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
test -f "$VIDEO_ROOT/VIDEO_STAGE_A_COMPLETE"
common=(--root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" --video-mae-root "$VIDEO_ROOT" --conditions V1 --device cuda)
run() { "$@" 2>&1 | tee -a "$LOG"; }
run "$PY" "$RUNNER" "${common[@]}" --out-root "$OUT/smoke" --protocols cross_day --seeds 240800 --epochs 3 --patience 3
touch "$OUT/SMOKE_COMPLETE"
run "$PY" "$RUNNER" "${common[@]}" --out-root "$OUT/formal" --protocols cross_day,within_subject_day --seeds 240800,240801,240802 --epochs 80 --patience 15
touch "$OUT/FORMAL_COMPLETE"
