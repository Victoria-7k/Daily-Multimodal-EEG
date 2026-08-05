#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
EXPORT=/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export
PY=/home/lzs/miniconda3/envs/lzs/bin/python
MODEL=/home/lzs/.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415
IDX=$EXPORT/index/eeg23win_video_audio_window_index.jsonl
CHUNKS=$EXPORT/index/video_chunks
CACHE=$EXPORT/tmp/video_2xroi_openface_cache_full
DEVICE=${VIDEO_DEVICE:-cpu}
BATCH_SIZE=${VIDEO_BATCH_SIZE:-4}
NUM_FRAMES=${VIDEO_NUM_FRAMES:-4}
FPS=${VIDEO_FPS:-1}
AUG_VIEWS=${VIDEO_AUG_VIEWS:-2}

mkdir -p \
  "$EXPORT/embeddings/video/chunks" \
  "$EXPORT/reports/video_embedding_chunks" \
  "$EXPORT/checksums"

run_variant() {
  local route=$1
  local aug=$2
  local encoder=$3
  echo "variant=$route aug=$aug views=$AUG_VIEWS device=$DEVICE batch=$BATCH_SIZE frames=$NUM_FRAMES"
  cd "$REPO"
  local pids=()
  for c in 0 1 2 3 4 5 6 7; do
    (
      set -euo pipefail
      CI=$(printf '%02d' "$c")
      WIN="$CHUNKS/video_chunk_${CI}.jsonl"
      RAW="$EXPORT/embeddings/video/chunks/video_${route}_chunk_${CI}.npz"
      TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "$PY" scripts/27_extract_dinov2_roi_embeddings.py \
        --window-index "$WIN" \
        --openface-cache-root "$CACHE" \
        --openface-encoder-profile openface_temporal_v1 \
        --video-region 2x_face_roi \
        --fps "$FPS" \
        --num-frames "$NUM_FRAMES" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --model-name "$MODEL" \
        --augmentation-profile "$aug" \
        --augmentation-views "$AUG_VIEWS" \
        --out "$RAW" \
        --failures-out "$EXPORT/reports/video_embedding_chunks/video_${route}_failures_${CI}.json" \
        --progress-out "$EXPORT/reports/video_embedding_chunks/video_${route}_progress_${CI}.jsonl" \
        --progress-every 200
    ) > "$EXPORT/reports/video_embedding_chunks/video_${route}_chunk_${c}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "variant $route failed during chunk extraction" >&2
    exit 1
  fi
  "$PY" "$EXPORT/scripts/12_merge_video_chunks.py" \
    --index "$IDX" \
    --sources "$EXPORT"/embeddings/video/chunks/video_${route}_chunk_*.npz \
    --out "$EXPORT/embeddings/video/video_${route}_2xroi_eeg23win_embeddings.npz" \
    --report "$EXPORT/reports/video_${route}_2xroi_aligned_pack_report.json" \
    --encoder-version "$encoder"
  sha256sum "$EXPORT/embeddings/video/video_${route}_2xroi_eeg23win_embeddings.npz" \
    > "$EXPORT/checksums/video_${route}_2xroi_sha256.txt"
}

run_variant A1 v4d_a1_color_brightness video_v4d_a1_dinov2_2x_face_roi_mean_std_max
run_variant A2 v4d_a2_color_brightness_grayscale video_v4d_a2_dinov2_2x_face_roi_mean_std_max
