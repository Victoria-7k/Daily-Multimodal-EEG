from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.event_windows import load_window_index
from daily_multimodal.embeddings.video_behavior_flags import extract_behavior_flags


FACE_FILTER_WARNING = (
    "Use an unfiltered window index such as outputs/window_index/window_index.jsonl. "
    "Face-filtered inputs such as real_cache_face_detected...jsonl are not valid for "
    "the main behavior audit because offscreen, occlusion, and low-face windows may "
    "already be removed."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Extract window-level video behavior flags. WARNING: {FACE_FILTER_WARNING}"
    )
    parser.add_argument(
        "--window-index",
        default="outputs/window_index/window_index.jsonl",
        help=f"Unfiltered window index JSONL. {FACE_FILTER_WARNING}",
    )
    parser.add_argument(
        "--out",
        default="outputs/cache/video_behavior_flags/video_behavior_flags.jsonl",
        help="Output JSONL for successful window-level behavior rows.",
    )
    parser.add_argument(
        "--failures-out",
        default="outputs/reports/video_behavior_flags_failures.json",
        help="Output JSON for dependency_missing/not_implemented/source failures.",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate for real extraction.")
    parser.add_argument("--max-windows", type=int, help="Optional cap for smoke runs.")
    parser.add_argument(
        "--openface-cache-root",
        help=(
            "Optional OpenFace/ROI cache root. When set, reads "
            "openface/<sample_id>/<profile>/openface_target.json and its CSV instead "
            "of requiring live video detector extraction."
        ),
    )
    parser.add_argument(
        "--openface-encoder-profile",
        default="openface_temporal_v1",
        help="OpenFace cache profile directory name under each sample_id.",
    )
    parser.add_argument(
        "--behavior-backend",
        choices=["openface_cache", "mediapipe_holistic_v1"],
        default="openface_cache",
        help="Behavior backend. mediapipe_holistic_v1 samples raw video and uses OpenFace cache only for head/face priority fields.",
    )
    parser.add_argument(
        "--max-frames-per-window",
        type=int,
        help="Optional cap for sampled frames per window. Default is ceil(window_duration * fps).",
    )
    parser.add_argument(
        "--mediapipe-max-image-size",
        type=int,
        default=640,
        help="Resize sampled frames so the longest side is at most this many pixels before MediaPipe. Use 0 to disable.",
    )
    parser.add_argument(
        "--progress-out",
        help="Optional progress log. Each window writes start/done lines with a progress bar and sample_id.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Write done progress lines at least every N windows. Start lines are always written.",
    )
    args = parser.parse_args()

    if _looks_face_filtered(args.window_index):
        print(f"warning={FACE_FILTER_WARNING}", file=sys.stderr)

    out = args.out
    if args.behavior_backend == "mediapipe_holistic_v1" and out == parser.get_default("out"):
        out = "outputs/cache/video_behavior_flags/mediapipe_holistic_behavior_flags.jsonl"
    progress_out = args.progress_out
    if args.behavior_backend == "mediapipe_holistic_v1" and progress_out is None:
        progress_out = "outputs/reports/mediapipe_holistic_progress.log"

    windows = load_window_index(args.window_index)
    summary = extract_behavior_flags(
        windows,
        out=out,
        failures_out=args.failures_out,
        fps=args.fps,
        max_windows=args.max_windows,
        openface_cache_root=args.openface_cache_root,
        openface_encoder_profile=args.openface_encoder_profile,
        behavior_backend=args.behavior_backend if args.behavior_backend != "openface_cache" else None,
        max_frames_per_window=args.max_frames_per_window,
        mediapipe_max_image_size=args.mediapipe_max_image_size,
        progress_out=progress_out,
        progress_every=args.progress_every,
    )
    print(f"behavior_flags_path={out}")
    print(f"failures_path={args.failures_out}")
    if progress_out:
        print(f"progress_path={progress_out}")
    print(f"selected_count={summary['selected_count']}")
    print(f"written_count={summary['written_count']}")
    print(f"failure_count={summary['failure_count']}")
    return 0 if summary["failure_count"] == 0 else 1


def _looks_face_filtered(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    return "real_cache_face_detected" in lowered or "face_detected" in lowered


if __name__ == "__main__":
    raise SystemExit(main())
