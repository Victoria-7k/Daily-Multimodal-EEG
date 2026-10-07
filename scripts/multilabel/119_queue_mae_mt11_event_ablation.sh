#!/usr/bin/env bash
set -euo pipefail
ROOT="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
OUT="$ROOT/outputs/mae_mt11_window_20260928"
LOG="$OUT/logs/119_queue_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$OUT/logs" "$OUT/smoke" "$OUT/formal"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
run() { "$@" 2>&1 | tee -a "$LOG"; }
run "$PY" "$ROOT/scripts/multilabel/118_run_mae_mt11_event_ablation.py" --out-root "$OUT/smoke" --protocols cross_day --seeds 240800 --epochs 3 --patience 3 --device cuda
touch "$OUT/SMOKE_COMPLETE"
run "$PY" "$ROOT/scripts/multilabel/118_run_mae_mt11_event_ablation.py" --out-root "$OUT/formal" --protocols cross_day,within_subject_day --seeds 240800,240801,240802 --epochs 80 --patience 15 --device cuda
touch "$OUT/FORMAL_COMPLETE"
