from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.dinov2_roi import build_dinov2_roi_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract V4 DINOv2-Base frozen embeddings from ROI or video region window clips.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--openface-cache-root")
    parser.add_argument("--openface-encoder-profile", default="openface_temporal_v1")
    parser.add_argument("--region-cache-root", help="Optional outputs/cache/video_regions root.")
    parser.add_argument("--video-region", default="2x_face_roi", choices=["2x_face_roi", "upper_body", "full_frame"])
    parser.add_argument(
        "--fallback-video-region",
        choices=["full_frame"],
        help="Optional explicit V4d ROI policy fallback. Currently supported as upper_body -> full_frame for region caches.",
    )
    parser.add_argument(
        "--direct-video-region-from-window",
        action="store_true",
        help="For upper_body/full_frame, sample region frames directly from window-index source videos instead of a region cache.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-sequences-out", help="Optional V4b input bundle with DINOv2 frame_embeddings.")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--max-frames-per-window",
        type=int,
        help="Deprecated alias for --num-frames; kept for older run scripts.",
    )
    parser.add_argument("--temporal-pooling", default="mean_std_max", choices=["mean_std_max"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--device")
    parser.add_argument("--progress-out")
    parser.add_argument("--failures-out")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based row offset into --window-index for chunked/resume-safe runs.")
    parser.add_argument("--max-windows", type=int, help="Optional cap for smoke/debug runs; default is full input.")
    parser.add_argument(
        "--augmentation-profile",
        default="none",
        choices=[
            "none",
            "v4d_mild_color_crop_scale",
            "v4d_appearance_mild",
            "v4d_a1_color_brightness",
            "v4d_a2_color_brightness_grayscale",
            "v4d_a3_color_brightness_grayscale_crop_scale",
            "v4d_weak_color_brightness_contrast",
        ],
        help="Optional V4d deterministic frame augmentation profile. Default keeps V4a unchanged.",
    )
    parser.add_argument(
        "--augmentation-views",
        type=int,
        default=1,
        help="Number of deterministic views to average when augmentation is enabled; includes the original view.",
    )
    args = parser.parse_args()

    summary = build_dinov2_roi_embeddings(
        window_index_path=args.window_index,
        openface_cache_root=args.openface_cache_root,
        openface_encoder_profile=args.openface_encoder_profile,
        region_cache_root=args.region_cache_root,
        video_region=args.video_region,
        fallback_video_region=args.fallback_video_region,
        direct_video_region_from_window=args.direct_video_region_from_window,
        out_path=args.out,
        frame_sequences_out=args.frame_sequences_out,
        fps=args.fps,
        num_frames=args.max_frames_per_window if args.max_frames_per_window is not None else args.num_frames,
        temporal_pooling=args.temporal_pooling,
        batch_size=args.batch_size,
        model_name=args.model_name,
        device=args.device,
        progress_out=args.progress_out,
        failures_out=args.failures_out,
        progress_every=args.progress_every,
        start_index=args.start_index,
        max_windows=args.max_windows,
        augmentation_profile=args.augmentation_profile,
        augmentation_views=args.augmentation_views,
    )
    print(f"variant={summary['variant']}")
    print(f"row_count={summary['row_count']}")
    print(f"face_mask_sum={summary['mask_sum']}")
    print(f"failure_count={summary['failure_count']}")
    print(f"start_index={summary['start_index']}")
    print(f"source_row_count={summary['source_row_count']}")
    print(f"out_path={summary['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
