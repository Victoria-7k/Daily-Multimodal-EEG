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
    parser = argparse.ArgumentParser(description="Extract V4 DINOv2-Base frozen embeddings from 2x ROI window videos.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--openface-cache-root", required=True)
    parser.add_argument("--openface-encoder-profile", default="openface_temporal_v1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames-per-window", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--device")
    parser.add_argument("--progress-out")
    parser.add_argument("--failures-out")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    summary = build_dinov2_roi_embeddings(
        window_index_path=args.window_index,
        openface_cache_root=args.openface_cache_root,
        openface_encoder_profile=args.openface_encoder_profile,
        out_path=args.out,
        fps=args.fps,
        max_frames_per_window=args.max_frames_per_window,
        batch_size=args.batch_size,
        model_name=args.model_name,
        device=args.device,
        progress_out=args.progress_out,
        failures_out=args.failures_out,
        progress_every=args.progress_every,
    )
    print(f"variant={summary['variant']}")
    print(f"row_count={summary['row_count']}")
    print(f"face_mask_sum={summary['mask_sum']}")
    print(f"failure_count={summary['failure_count']}")
    print(f"out_path={summary['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
