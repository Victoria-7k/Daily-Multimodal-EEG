#!/usr/bin/env bash
# Continue the old-configuration EEGPT bank, one label per process.
set -euo pipefail

cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
FIRST_PID=3521701
while ps -p "$FIRST_PID" -o args= 2>/dev/null | grep -Fq '103_run_legacy_single_label_eegpt.py --protocols cross_day --labels alert --limit-runs 1'; do
  sleep 30
done

echo "legacy EEGPT bank start $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
ulimit -c 0
for protocol in cross_day date_in_order; do
  for label in inspired alert determined attentive active hostile nervous upset afraid ashamed fatigue; do
    for attempt in 1 2 3; do
      echo "run protocol=$protocol label=$label attempt=$attempt $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
      if PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python \
          scripts/multilabel/103_run_legacy_single_label_eegpt.py \
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
echo "legacy EEGPT bank complete $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
