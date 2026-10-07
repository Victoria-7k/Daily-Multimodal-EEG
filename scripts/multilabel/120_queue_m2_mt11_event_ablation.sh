#!/usr/bin/env bash
set -euo pipefail
ROOT="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
RUNNER="${M2_RUNNER:-$ROOT/scripts/multilabel/118_run_mae_mt11_event_ablation.py}"
OUT="${M2_OUT:-$ROOT/outputs/mae_mt11_m2_20260928}"
LOG="$OUT/logs/120_queue_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$OUT/logs" "$OUT/smoke" "$OUT/formal"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
run() { "$@" 2>&1 | tee -a "$LOG"; }
# M2 replaces EEG and Wear simultaneously; Video remains the A1 DINO token in the archived reference bag.
run "$PY" "$RUNNER" --root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" --mae-root "$ROOT/outputs/mae_20260924" --out-root "$OUT/smoke" --protocols cross_day --conditions M2 --seeds 240800 --epochs 3 --patience 3 --device cuda
touch "$OUT/SMOKE_COMPLETE"
run "$PY" "$RUNNER" --root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" --mae-root "$ROOT/outputs/mae_20260924" --out-root "$OUT/formal" --protocols cross_day,within_subject_day --conditions M2 --seeds 240800,240801,240802 --epochs 80 --patience 15 --device cuda
touch "$OUT/FORMAL_COMPLETE"
