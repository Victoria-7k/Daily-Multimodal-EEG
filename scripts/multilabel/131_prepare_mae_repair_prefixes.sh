#!/usr/bin/env bash
set -Eeuo pipefail
ROLE="${1:?usage: 131_prepare_mae_repair_prefixes.sh h20|ncc}"
if [[ "$ROLE" == h20 ]]; then
  STAGING=/home/wangzw/mae_repair_20261008
  ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
  PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  modalities=(eeg wear)
else
  [[ "$ROLE" == ncc ]]
  STAGING=/tmp/wangzw_mae_repair_20261008
  ROOT=/home/lzs/DailyVideoMAE_20260927
  PY=/home/lzs/miniconda3/envs/eeg3dim/bin/python
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
  modalities=(video)
fi
BASE="$STAGING/outputs"
OUT="$BASE/prefixes_v2"
mkdir -p "$OUT" "$BASE/logs"
exec 9>"$OUT/.prepare.lock"
if ! flock -n 9; then echo 'prefix_preparation_already_running'; exit 0; fi
trap 'touch "$OUT/PREFIX_PREPARATION_FAILED"' ERR
[[ -f "$BASE/stage_a_v2/STAGE_A_FORMAL_COMPLETE" ]]
export PYTHONPATH="$STAGING/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for protocol in cross_day within_subject_day; do
  for modality in "${modalities[@]}"; do
    "$PY" "$STAGING/scripts/multilabel/123_export_mae_frozen_prefix.py" \
      --modality "$modality" --initialization pretrained \
      --checkpoint "$BASE/stage_a_v2/formal/$protocol/${modality}_seed_240800/checkpoint.pt" \
      --token-file "$BASE/stage_a_v2/formal/$protocol/${modality}_seed_240800/window_embeddings.npz" \
      --out-dir "$OUT/pretrained/$protocol/$modality" --batch-size 64 --device cuda \
      --eeg-source /vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy \
      --wear-root /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/wear_fm/staged_inputs \
      --video-cache "$ROOT/cache/video_8x112.uint8.mmap" \
      2>&1 | tee "$BASE/logs/prefix_${modality}_${protocol}.log"
  done
done
touch "$OUT/PREFIX_${ROLE^^}_COMPLETE"
