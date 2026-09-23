#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! "$2" =~ ^[0-9]+$ || ( "$1" != mt11 && "$1" != st11 ) ]]; then
  echo "usage: $0 <mt11|st11> <upstream_pid>" >&2
  exit 2
fi

mode="$1"
upstream_pid="$2"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$root"
python_bin="$root/runtime/envs/eegpt-gpu-min/bin/python"
split_root="/vePFS-0x0d/DailyEEG/splits_new"
embedding_root="/vePFS-0x0d/DailyEEG_multimodal/embeddings"
matrix_root="$root/outputs/multiemotion_20260913"
export PYTHONPATH=src TMPDIR="$root/outputs/tmp" XDG_CACHE_HOME="$root/outputs/tmp/xdg"

while kill -0 "$upstream_pid" 2>/dev/null; do
  state="$(ps -o stat= -p "$upstream_pid" 2>/dev/null || true)"
  [[ "$state" == Z* ]] && break
  sleep 30
done

"$python_bin" - "$mode" "$split_root" "$embedding_root" "$matrix_root" <<'PY'
import json
import sys
from pathlib import Path

mode, split_root, embedding_root, matrix_root = sys.argv[1:]
labels = ("inspired", "alert", "determined", "attentive", "active",
          "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue")
tasks = ("all_11_labels",) if mode == "mt11" else labels
stage = "phase4_multitask_eeg" if mode == "mt11" else "phase3_single_task_eeg"
for task in tasks:
    metrics = Path(matrix_root) / stage / "within_subject_day" / task / "seed_240800/metrics.json"
    token = (Path(embedding_root) / "eeg_encoder_256d_tokens" /
             ("multitask_11label/within_subject_day/seed_240800.npz" if mode == "mt11"
              else f"single_task/within_subject_day/{task}/seed_240800.npz"))
    if not metrics.is_file() or not token.is_file():
        raise SystemExit(f"missing upstream result: {metrics} or {token}")
    result = json.loads(metrics.read_text(encoding="utf-8"))
    if (result.get("status") != "ok" or result.get("protocol") != "within_subject_day"
            or result.get("split_root") != f"{split_root}/within_subject_day"
            or result.get("seed") != 240800 or result.get("task_name") != task):
        raise SystemExit(f"upstream result does not use the requested split: {metrics}")
print(f"upstream verified: {mode} {len(tasks)}/{len(tasks)}", flush=True)
PY

safe_remove() {
  local target="$1"
  local resolved
  resolved="$(realpath -m -- "$target")"
  case "$resolved" in
    "$matrix_root/structure_matrix_A1/bags/within_subject_day"|\
    "$matrix_root/structure_matrix_A1/runs/within_subject_day"|\
    "$matrix_root/single_task_structure_matrix_A1/bags/within_subject_day"|\
    "$matrix_root/single_task_structure_matrix_A1/runs/within_subject_day"|\
    "$matrix_root/single_task_structure_matrix_A1/preflight/within_subject_day"|\
    "$matrix_root/within_subject_day_splits_new_preflight_mt11"|\
    "$matrix_root/within_subject_day_splits_new_preflight_st11") ;;
    *) echo "refusing unexpected removal target: $target" >&2; exit 3 ;;
  esac
  if [[ -e "$target" ]]; then
    rm -rf -- "$target"
  fi
}

if [[ "$mode" == mt11 ]]; then
  output_root="$matrix_root/structure_matrix_A1"
  staging="$matrix_root/within_subject_day_splits_new_preflight_mt11"
  safe_remove "$staging"
  "$python_bin" scripts/multilabel/98_run_structure_emotion_matrix.py \
    --stage preflight --protocols within_subject_day --splits-root "$split_root" \
    --embeddings-root "$embedding_root" --out-root "$staging"
  "$python_bin" scripts/multilabel/98_run_structure_emotion_matrix.py \
    --stage preflight --protocols cross_day --embeddings-root "$embedding_root" \
    --out-root "$output_root"
  safe_remove "$output_root/bags/within_subject_day"
  safe_remove "$output_root/runs/within_subject_day"
  mkdir -p "$output_root/bags"
  mv "$staging/bags/within_subject_day" "$output_root/bags/within_subject_day"
  "$python_bin" - "$output_root" "$staging" <<'PY'
import json
import sys
from pathlib import Path

root, staging = map(Path, sys.argv[1:])
cross = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
within = json.loads((staging / "preflight.json").read_text(encoding="utf-8"))
if len(cross) != 1 or cross[0]["protocol"] != "cross_day" or len(within) != 1 or within[0]["protocol"] != "within_subject_day":
    raise SystemExit("unexpected preflight inventory")
(root / "preflight.json").write_text(json.dumps(cross + within, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  "$python_bin" scripts/multilabel/98_run_structure_emotion_matrix.py \
    --protocols within_subject_day --splits-root "$split_root" \
    --embeddings-root "$embedding_root" --out-root "$output_root"
  "$python_bin" scripts/multilabel/99_summarize_structure_emotion_matrix.py \
    --protocols cross_day,within_subject_day --root "$output_root" --out-dir "$output_root/summary"
  safe_remove "$staging"
  date --iso-8601=seconds > "$output_root/WITHIN_SUBJECT_DAY_SPLITS_NEW_COMPLETE"
else
  output_root="$matrix_root/single_task_structure_matrix_A1"
  staging="$matrix_root/within_subject_day_splits_new_preflight_st11"
  safe_remove "$staging"
  "$python_bin" scripts/multilabel/100_audit_eegpt_single_label_bank.py \
    --protocols cross_day,within_subject_day --splits-root "$split_root" \
    --embeddings-root "$embedding_root" \
    --out "$matrix_root/phase3_single_task_eeg/bank_audit.json"
  "$python_bin" scripts/multilabel/101_run_single_task_structure_matrix.py \
    --stage preflight --protocols within_subject_day --splits-root "$split_root" \
    --embeddings-root "$embedding_root" --out-root "$staging"
  safe_remove "$output_root/bags/within_subject_day"
  safe_remove "$output_root/runs/within_subject_day"
  safe_remove "$output_root/preflight/within_subject_day"
  mkdir -p "$output_root/bags"
  mv "$staging/bags/within_subject_day" "$output_root/bags/within_subject_day"
  "$python_bin" scripts/multilabel/101_run_single_task_structure_matrix.py \
    --stage preflight --protocols within_subject_day --splits-root "$split_root" \
    --embeddings-root "$embedding_root" --out-root "$output_root"
  "$python_bin" scripts/multilabel/101_run_single_task_structure_matrix.py \
    --stage preflight --protocols cross_day \
    --embeddings-root "$embedding_root" --out-root "$output_root"
  "$python_bin" scripts/multilabel/101_run_single_task_structure_matrix.py \
    --protocols within_subject_day --splits-root "$split_root" \
    --embeddings-root "$embedding_root" --out-root "$output_root"
  "$python_bin" scripts/multilabel/102_summarize_single_task_structure_matrix.py \
    --protocols cross_day,within_subject_day --root "$output_root" --out-dir "$output_root/summary"
  safe_remove "$staging"
  date --iso-8601=seconds > "$output_root/WITHIN_SUBJECT_DAY_SPLITS_NEW_COMPLETE"
fi

echo "within_subject_day split replay complete: mode=$mode output=$output_root"
