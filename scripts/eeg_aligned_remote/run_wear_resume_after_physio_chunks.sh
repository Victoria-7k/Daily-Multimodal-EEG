#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
EXPORT=/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export
PY=/usr/bin/python
IDX=$EXPORT/index/eeg23win_video_audio_window_index.jsonl
CHUNKS=$EXPORT/index/wear_chunks

mkdir -p "$EXPORT/reports/wear_chunks" "$EXPORT/embeddings/wear/chunks" "$EXPORT/tmp/wear_chunks" "$EXPORT/checksums"

merge_profile() {
  local profile=$1
  "$PY" "$EXPORT/scripts/09_merge_wear_chunks.py" \
    --index "$IDX" \
    --sources "$EXPORT"/embeddings/wear/chunks/${profile}_chunk_*.npz \
    --out "$EXPORT/embeddings/wear/${profile}_eeg23win_embeddings.npz" \
    --report "$EXPORT/reports/${profile}_aligned_pack_report.json" \
    --encoder-version "$profile"
  sha256sum "$EXPORT/embeddings/wear/${profile}_eeg23win_embeddings.npz" \
    > "$EXPORT/checksums/${profile}_sha256.txt"
}

run_deep_chunks() {
  local profile=wear_deep_sequence_preprocessed_v1
  cd "$REPO"
  for c in 0 1 2 3; do
    (
      set -euo pipefail
      CI=$(printf '%02d' "$c")
      WIN="$CHUNKS/wear_chunk_${CI}.jsonl"
      RAW="$EXPORT/embeddings/wear/chunks/${profile}_chunk_${CI}.npz"
      python scripts/15_extract_wear_embeddings.py \
        --window-index "$WIN" \
        --cache-root "$EXPORT/tmp/wear_chunks/${profile}_chunk_${CI}" \
        --encoder-profile "$profile" \
        --out "$RAW" \
        --failures-out "$EXPORT/reports/wear_chunks/${profile}_failures_${CI}.json" \
        --summary-out "$EXPORT/reports/wear_chunks/${profile}_summary_${CI}.json"
    ) > "$EXPORT/reports/wear_chunks/${profile}_chunk_${c}.log" 2>&1 &
  done
  wait
}

merge_profile wear_physio_features_preprocessed_v1
run_deep_chunks
merge_profile wear_deep_sequence_preprocessed_v1
