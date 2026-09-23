#!/usr/bin/env bash
# Run the old-configuration scalar matrix with one protocol/label per process.
set -euo pipefail

cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
export PYTHONPATH=src
export TMPDIR="$PWD/outputs/tmp"
export XDG_CACHE_HOME="$PWD/outputs/tmp/xdg-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
ulimit -c 0

echo "legacy scalar matrix start $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
for protocol in cross_day date_in_order; do
  for label in inspired alert determined attentive active hostile nervous upset afraid ashamed fatigue; do
    for attempt in 1 2 3; do
      echo "run protocol=$protocol label=$label attempt=$attempt $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
      if runtime/envs/eegpt-gpu-min/bin/python \
          scripts/multilabel/104_run_legacy_single_task_matrix.py \
          --protocols "$protocol" --labels "$label"; then
        break
      fi
      if [ "$attempt" -eq 3 ]; then
        echo "failed protocol=$protocol label=$label after three attempts" >&2
        exit 1
      fi
      sleep 60
    done
  done
done
echo "legacy scalar matrix complete $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
