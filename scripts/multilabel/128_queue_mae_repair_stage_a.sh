#!/usr/bin/env bash
# Isolated repair: fixed preprocessing, label-free train/val gates, old artifacts preserved.
set -Eeuo pipefail
ROLE="${1:?usage: 128_queue_mae_repair_stage_a.sh h20|ncc [smoke|formal|all]}"
PHASE="${2:-all}"
[[ "$PHASE" == smoke || "$PHASE" == formal || "$PHASE" == all ]]
if [[ "$ROLE" == h20 ]]; then
  ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
  STAGING="${MAE_REPAIR_STAGING:-/home/wangzw/mae_repair_20261008}"
  PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
  modalities=(eeg wear)
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
  [[ "$ROLE" == ncc ]]
  ROOT=/home/lzs/DailyVideoMAE_20260927
  STAGING="${MAE_REPAIR_STAGING:-/tmp/wangzw_mae_repair_20261008}"
  PY=/home/lzs/miniconda3/envs/eeg3dim/bin/python
  modalities=(video)
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
fi
OUT="$STAGING/outputs/stage_a_v2"
mkdir -p "$OUT/logs" "$STAGING/tmp"
export PYTHONPATH="$STAGING/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$STAGING/tmp"
trap 'touch "$OUT/STAGE_A_QUEUE_FAILED"' ERR
cd "$STAGING"

check_stage() {
  "$PY" - "$OUT/$1" "$ROLE" "$1" <<'PY'
import json, math, sys
from pathlib import Path
import numpy as np
root, role, phase = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for protocol in ('cross_day', 'within_subject_day'):
    for modality in (('eeg', 'wear') if role == 'h20' else ('video',)):
        directory = root / protocol / f'{modality}_seed_240800'
        config = json.loads((directory / 'config.json').read_text())
        expected_version = 'seed_before_init_fixed_validation_health_v2' if modality == 'video' else 'position_preserved_fixed_validation_v2'
        assert config['training_version'] == expected_version
        assert config['test_labels_read'] is False and config['embedding_seed'] == 240800
        audit = config['reconstruction_audit']
        assert not audit['selected_embedding_health']['collapsed']
        assert audit['selected_embedding_health']['relative_variation'] >= 1e-3
        assert all(math.isfinite(row['train_masked_nmse']) and math.isfinite(row['val_masked_nmse']) and not row['embedding_health']['collapsed'] for row in audit['history'])
        if phase == 'smoke':
            assert len(audit['history']) == 15 and config['token_export'] == 'skipped_for_smoke'
            assert config['smoke_per_split'] == 1024
        else:
            assert config['smoke_per_split'] == 0
            with np.load(directory / 'window_embeddings.npz', allow_pickle=False) as token:
                assert token['embedding'].shape == (28819, 256) and np.isfinite(token['embedding']).all()
                assert len(np.unique(token['sample_id'])) == 28819
                valid = token['valid_mask'].astype(bool)
                assert int(valid.sum()) == {'eeg': 28819, 'wear': 24127, 'video': 18012}[modality]
                assert (token['embedding'][~valid] == 0).all()
        print(f'GATE_PASS phase={phase} protocol={protocol} modality={modality} best_epoch={audit["best_epoch"]} relative_variation={audit["selected_embedding_health"]["relative_variation"]}', flush=True)
PY
}

run_one() {
  local phase="$1" modality="$2" protocol="$3"
  local common=(--protocol "$protocol" --out-root "$OUT/$phase" --seed 240800 --device cuda --health-probe-count 256 --min-relative-variation 0.001)
  local smoke=()
  [[ "$phase" != smoke ]] || smoke=(--epochs 15 --patience 15 --smoke-per-split 1024 --skip-full-export)
  printf '%s starting phase=%s modality=%s protocol=%s\n' "$(date -Iseconds)" "$phase" "$modality" "$protocol"
  if [[ "$modality" == video ]]; then
    "$PY" "$STAGING/scripts/multilabel/114_run_video_mae.py" \
      --video-metadata "$ROOT/input/video_metadata.npz" --splits-root "$ROOT/input/splits" \
      --clip-cache "$ROOT/cache/video_8x112.uint8.mmap" --reuse-complete-cache \
      "${common[@]}" "${smoke[@]}" 2>&1 | tee "$OUT/logs/${phase}_${modality}_${protocol}.log"
  else
    "$PY" "$STAGING/scripts/multilabel/112_run_modality_mae.py" \
      --aligned-root "$ROOT" --modality "$modality" \
      "${common[@]}" "${smoke[@]}" 2>&1 | tee "$OUT/logs/${phase}_${modality}_${protocol}.log"
  fi
}

if [[ "$PHASE" == smoke || "$PHASE" == all ]]; then
  for modality in "${modalities[@]}"; do
    for protocol in cross_day within_subject_day; do run_one smoke "$modality" "$protocol"; done
  done
  check_stage smoke
  touch "$OUT/STAGE_A_SMOKE_COMPLETE"
fi
if [[ "$PHASE" == formal || "$PHASE" == all ]]; then
  [[ -f "$OUT/STAGE_A_SMOKE_COMPLETE" ]]
  check_stage smoke
  for modality in "${modalities[@]}"; do
    for protocol in cross_day within_subject_day; do run_one formal "$modality" "$protocol"; done
  done
  check_stage formal
  touch "$OUT/STAGE_A_FORMAL_COMPLETE"
fi
