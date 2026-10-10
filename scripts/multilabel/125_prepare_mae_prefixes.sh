#!/usr/bin/env bash
set -euo pipefail
ROLE="${1:?usage: 125_prepare_mae_prefixes.sh h20|ncc}"
if [[ "$ROLE" == h20 ]]; then
  ROOT="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"
  PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
  STAGING="${MAE_REMAINING_STAGING:-/home/wangzw/mae_remaining_staging}"
  OUT="${MAE_PREFIX_ROOT:-/home/wangzw/outputs/mae_remaining_20261007/prefixes}"
  MAE_ROOT="$ROOT/outputs/mae_20260924"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
  modalities=(eeg wear)
else
  [[ "$ROLE" == ncc ]]
  ROOT="/home/lzs/DailyVideoMAE_20260927"
  PY="/home/lzs/miniconda3/envs/eeg3dim/bin/python"
  STAGING="${MAE_REMAINING_STAGING:-/tmp/wangzw_mae_remaining_20261007}"
  OUT="${MAE_PREFIX_ROOT:-/tmp/wangzw_mae_remaining_20261007/prefixes}"
  MAE_ROOT="$ROOT/outputs/mae_20260927"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
  modalities=(video)
fi
mkdir -p "$OUT"
trap 'touch "$OUT/PREFIX_PREPARATION_FAILED"' ERR
for initialization in pretrained random; do
  if [[ "$initialization" == pretrained ]]; then
    protocols=(cross_day within_subject_day)
  else
    protocols=(shared)
  fi
  for protocol in "${protocols[@]}"; do
    source_protocol="$protocol"
    [[ "$protocol" != shared ]] || source_protocol=cross_day
    for modality in "${modalities[@]}"; do
      common=(--modality "$modality" --initialization "$initialization" --checkpoint "$MAE_ROOT/$source_protocol/${modality}_seed_240800/checkpoint.pt" --token-file "$MAE_ROOT/$source_protocol/${modality}_seed_240800/window_embeddings.npz" --out-dir "$OUT/$initialization/$protocol/$modality" --batch-size 64 --device cuda)
      if [[ "$ROLE" == h20 ]]; then
        "$PY" "$STAGING/123_export_mae_frozen_prefix.py" "${common[@]}" --eeg-source /vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy --wear-root "$ROOT/outputs/wear_fm/staged_inputs"
      else
        "$PY" "$STAGING/123_export_mae_frozen_prefix.py" "${common[@]}" --video-cache "$ROOT/cache/video_8x112.uint8.mmap"
      fi
    done
  done
done
touch "$OUT/PREFIX_${ROLE^^}_COMPLETE"
