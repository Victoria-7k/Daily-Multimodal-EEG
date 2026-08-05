#!/usr/bin/env bash
set -euo pipefail

ROOT="${B0_ROOT:-/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned}"
PYTHON="${B0_PYTHON:-$ROOT/runtime/envs/b0/bin/python}"

"$PYTHON" "$ROOT/scripts/run_b0_fusion_matrix.py" \
  --protocol cross_subject \
  --experiments B0_Wphysio_full \
  --epochs 3 \
  --hidden-dim 32 \
  --batch-size 128 \
  --patience 3 \
  --max-train 512 \
  --max-val 256 \
  --max-test 256 \
  --torch-threads 2 \
  --out-dir "$ROOT/reports/b0_smoke" \
  --intermediate-dir "$ROOT/intermediate/b0_smoke" \
  --summary-json "$ROOT/reports/b0_smoke_report.json" \
  --summary-md "$ROOT/reports/b0_smoke_report.md"
