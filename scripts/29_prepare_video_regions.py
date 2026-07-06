from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.video_regions import VIDEO_REGIONS, build_video_region_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare 2x face ROI, upper-body, and full-frame video region caches.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--out-root", default="outputs/cache/video_regions")
    parser.add_argument("--roi-cache-root", help="Cache root containing openface/<sample_id>/<profile>/window.mp4.")
    parser.add_argument("--roi-encoder-profile", default="openface_temporal_v1")
    parser.add_argument("--regions", nargs="+", default=list(VIDEO_REGIONS), choices=list(VIDEO_REGIONS))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--manifest-out")
    parser.add_argument("--failures-out")
    parser.add_argument("--no-skip-existing", action="store_true", help="Rewrite region clips even when window.mp4 and region.json already exist.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N windows to stderr.")
    args = parser.parse_args()

    summary = build_video_region_cache(
        window_index_path=args.window_index,
        out_root=args.out_root,
        roi_cache_root=args.roi_cache_root,
        roi_encoder_profile=args.roi_encoder_profile,
        regions=args.regions,
        start_index=args.start_index,
        max_windows=args.max_windows,
        manifest_out=args.manifest_out,
        failures_out=args.failures_out,
        skip_existing=not args.no_skip_existing,
        progress_every=args.progress_every,
    )
    print(f"selected_count={summary['selected_count']}")
    print(f"written_count={summary['written_count']}")
    print(f"skipped_existing_count={summary['skipped_existing_count']}")
    print(f"failure_count={summary['failure_count']}")
    print(f"manifest_path={args.manifest_out or Path(args.out_root) / 'video_regions_manifest.jsonl'}")
    print(f"failures_path={args.failures_out or Path(args.out_root) / 'video_regions_failures.json'}")
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
