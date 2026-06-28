from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.video_audio_alignment import (
    align_video_audio_rows,
    cached_probe_func,
    probe_many_mp4_paths,
    save_aligned_manifest,
    save_alignment_report,
)
from daily_multimodal.manifest.validate_manifest import load_jsonl_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Align video/audio candidates with event windows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/manifests/events_manifest_with_video_audio.jsonl")
    parser.add_argument("--report-out", default="outputs/reports/video_audio_alignment_report.json")
    parser.add_argument("--start-seconds", type=float, default=-60)
    parser.add_argument("--end-seconds", type=float, default=0)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--ffprobe-timeout", type=float, default=10)
    parser.add_argument("--ffprobe-workers", type=int, default=8)
    parser.add_argument("--ffprobe-cache", default="outputs/cache/video_audio/ffprobe_cache.jsonl")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--retry-failed-ffprobe", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl_manifest(args.manifest)
    ffprobe_timeout = None if args.ffprobe_timeout <= 0 else args.ffprobe_timeout
    candidate_paths = [
        path
        for row in rows
        for path in (row.get("candidate_mp4_paths") or [])
    ]
    probe_cache, probe_report = probe_many_mp4_paths(
        candidate_paths,
        timeout_seconds=ffprobe_timeout,
        max_workers=args.ffprobe_workers,
        cache_path=args.ffprobe_cache,
        progress_every=args.progress_every,
        retry_failed=args.retry_failed_ffprobe,
    )
    enriched, report = align_video_audio_rows(
        rows,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        timezone_name=args.timezone,
        ffprobe_timeout_seconds=ffprobe_timeout,
        ffprobe_func=cached_probe_func(probe_cache),
    )
    report = {**report, "ffprobe_probe_report": probe_report}
    manifest_path = save_aligned_manifest(enriched, args.out)
    report_path = save_alignment_report(report, args.report_out)
    print(f"aligned_manifest_path={manifest_path}")
    print(f"report_path={report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
