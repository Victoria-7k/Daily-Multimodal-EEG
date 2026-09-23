#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <phase4_multitask_eeg_pid>" >&2
  exit 2
fi

upstream_pid="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

python_bin="runtime/envs/eegpt-gpu-min/bin/python"
output_root="$root/outputs/multiemotion_20260913/structure_matrix_A1"
embedding_root="/vePFS-0x0d/DailyEEG_multimodal/embeddings"

mkdir -p "$root/outputs/tmp" "$root/outputs/tmp/xdg" "$root/outputs/multiemotion_20260913/logs"
export PYTHONPATH=src
export TMPDIR="$root/outputs/tmp"
export XDG_CACHE_HOME="$root/outputs/tmp/xdg"

while kill -0 "$upstream_pid" 2>/dev/null; do
  sleep 30
done

"$python_bin" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/multiemotion_20260913/phase4_multitask_eeg")
embeddings = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg_encoder_256d_tokens/multitask_11label")
for protocol in ("cross_day", "date_in_order"):
    metrics = root / protocol / "all_11_labels" / "seed_240800" / "metrics.json"
    token = embeddings / protocol / "seed_240800.npz"
    if not metrics.is_file() or not token.is_file():
        raise SystemExit(f"missing MT11 upstream artifact: {metrics} or {token}")
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if payload.get("status") != "ok" or payload.get("label_names") != [
        "inspired", "alert", "determined", "attentive", "active",
        "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue",
    ]:
        raise SystemExit(f"invalid MT11 upstream metrics: {metrics}")
PY

"$python_bin" scripts/multilabel/98_run_structure_emotion_matrix.py \
  --stage preflight --embeddings-root "$embedding_root" --out-root "$output_root"

safe_remove() {
  local target="$1"
  case "$target" in
    "$output_root/runs"|"$output_root/summary"|\
    "$output_root/bags/cross_day/A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1"|\
    "$output_root/bags/date_in_order/A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1") ;;
    *) echo "refusing unexpected removal target: $target" >&2; exit 3 ;;
  esac
  if [[ -e "$target" ]]; then
    rm -rf -- "$target"
  fi
}

safe_remove "$output_root/runs"
safe_remove "$output_root/summary"
safe_remove "$output_root/bags/cross_day/A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1"
safe_remove "$output_root/bags/date_in_order/A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1"

"$python_bin" scripts/multilabel/98_run_structure_emotion_matrix.py \
  --embeddings-root "$embedding_root" --out-root "$output_root"
"$python_bin" scripts/multilabel/99_summarize_structure_emotion_matrix.py \
  --root "$output_root" --out-dir "$output_root/summary"

date --iso-8601=seconds > "$output_root/MT11_REPLACEMENT_COMPLETE"
echo "MT11 structure matrix replacement complete: $output_root"
