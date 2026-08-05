#!/usr/bin/env bash
set -euo pipefail

ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
PY=$ROOT/runtime/envs/b0/bin/python
SCRIPT=$ROOT/scripts/generate_eegpt_eeg23win_embeddings.py

mkdir -p "$ROOT/logs" "$ROOT/reports" "$ROOT/checksums" "$ROOT/tmp" "$ROOT/outputs/checkpoints/eegpt-pretrained"

export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

"$PY" "$SCRIPT" \
  --max-rows 10 \
  --out "$ROOT/tmp/eeg_eegpt_smoke10_embeddings.npz" \
  --report-md "$ROOT/reports/eeg_eegpt_eeg23win_smoke10_report.md" \
  --report-json "$ROOT/reports/eeg_eegpt_eeg23win_smoke10_report.json" \
  --checksum-out "$ROOT/checksums/eeg_eegpt_eeg23win_smoke10_sha256.txt" \
  --checkpoint "$ROOT/outputs/checkpoints/eegpt-pretrained" \
  --download-checkpoint \
  --force \
  --batch-size 2 \
  --torch-threads 8 \
  --progress-every 2
