#!/usr/bin/env bash
# Wait for the old-configuration scalar queue, then strictly summarize all runs.
set -euo pipefail

QUEUE_PID="${1:?queue PID required}"
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
LOG=outputs/multiemotion_legacy_replay_20260917/logs/legacy_single_task_matrix.log

while ps -p "$QUEUE_PID" -o args= 2>/dev/null | grep -Fq '107_queue_legacy_single_task_matrix.sh'; do
  sleep 30
done

if ! tail -n 2 "$LOG" | grep -Fq 'legacy scalar matrix complete '; then
  echo "legacy scalar queue did not finish successfully; inspect $LOG" >&2
  exit 1
fi

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python \
  scripts/multilabel/102_summarize_single_task_structure_matrix.py \
  --root outputs/multiemotion_legacy_replay_20260917/single_task_matrix_A1 \
  --seeds 240729,240730,240731 \
  --pipeline single_task_legacy_eegpt_scalar
echo "legacy scalar summary complete $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
