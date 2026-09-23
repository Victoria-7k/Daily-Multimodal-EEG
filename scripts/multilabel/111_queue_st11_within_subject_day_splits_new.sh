#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <mt11_upstream_pid>" >&2
  exit 2
fi

mt11_pid="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$root"
while kill -0 "$mt11_pid" 2>/dev/null; do
  state="$(ps -o stat= -p "$mt11_pid" 2>/dev/null || true)"
  [[ "$state" == Z* ]] && break
  sleep 30
done

export PYTHONPATH=src TMPDIR="$root/outputs/tmp" XDG_CACHE_HOME="$root/outputs/tmp/xdg"
log="$root/outputs/multiemotion_20260913/logs/st11_within_subject_day_splits_new_20260923.log"
"$root/runtime/envs/eegpt-gpu-min/bin/python" scripts/multilabel/93_run_eegpt_single_label.py \
  --protocols within_subject_day \
  --splits-root /vePFS-0x0d/DailyEEG/splits_new \
  --seeds 240800 --out-root outputs/multiemotion_20260913/phase3_single_task_eeg \
  --device cuda --force > "$log" 2>&1 &
st11_pid="$!"
echo "st11 upstream pid=$st11_pid log=$log"
exec bash scripts/multilabel/110_rerun_within_subject_day_splits_new.sh st11 "$st11_pid"
