#!/usr/bin/env bash
# Run VideoMAE exclusively on ncc_serve_4090, where /mnt/dataset1 is mounted.
set -euo pipefail

ROOT=/home/lzs/DailyVideoMAE_20260927
PY=/home/lzs/miniconda3/envs/eeg3dim/bin/python
export CUDA_VISIBLE_DEVICES=1
export TMPDIR="$ROOT/tmp"
mkdir -p "$ROOT/outputs" "$ROOT/cache" "$ROOT/tmp"

run() {
  "$PY" "$ROOT/scripts/114_run_video_mae.py" \
    --video-metadata "$ROOT/input/video_metadata.npz" \
    --splits-root "$ROOT/input/splits" --out-root "$ROOT/outputs/mae_20260927" \
    --protocol "$1" --device cuda "${@:2}"
}

# Gate: validates raw-file access, clip decoding, tubelet masking and one
# validation-selected checkpoint without materializing an all-video cache.
run cross_day --epochs 2 --batch-size 2 --smoke-per-split 16 --skip-full-export

# Full caching remains on ncc; H20 receives only final token/checkpoint files.
CACHE="$ROOT/cache/video_8x112.uint8.mmap"
run cross_day --clip-cache "$CACHE" --build-cache-only
run cross_day --clip-cache "$CACHE"
run within_subject_day --clip-cache "$CACHE"
date --iso-8601=seconds > "$ROOT/outputs/mae_20260927/VIDEO_STAGE_A_COMPLETE"
