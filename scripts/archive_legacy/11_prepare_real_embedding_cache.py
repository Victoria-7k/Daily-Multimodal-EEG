from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.event_windows import load_window_index
from daily_multimodal.embeddings.cache import RealCacheProfiles, prepare_real_embedding_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare stage-12 real embedding cache records and audio clips.")
    parser.add_argument("--window-index", required=True, help="Window index JSONL produced by stage 3.")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--out-report", default="outputs/reports/real_embedding_readiness_report.md")
    parser.add_argument("--failures-out", default="outputs/reports/real_embedding_failures.json")
    parser.add_argument(
        "--filtered-window-index-out",
        default="outputs/window_index/real_cache_face_detected.jsonl",
        help="JSONL window index kept after optional face-presence filtering.",
    )
    parser.add_argument(
        "--skip-face-presence-filter",
        action="store_true",
        help="Disable the initial face detector filter and keep every input window.",
    )
    parser.add_argument("--eeg-encoder-profile", default="eeg_real_frozen_v1")
    parser.add_argument("--wear-encoder-profile", default="wear_sequence_v1")
    parser.add_argument("--face-encoder-profile", default="openface_temporal_v1")
    parser.add_argument("--audio-encoder-profile", default="wavlm_frozen_v1")
    args = parser.parse_args()

    windows = load_window_index(args.window_index)
    selected = windows[: args.max_windows] if args.max_windows is not None else windows
    summary = prepare_real_embedding_cache(
        selected,
        cache_root=args.cache_root,
        report_out=args.out_report,
        failures_out=args.failures_out,
        profiles=RealCacheProfiles(
            eeg=args.eeg_encoder_profile,
            wear=args.wear_encoder_profile,
            face=args.face_encoder_profile,
            audio=args.audio_encoder_profile,
        ),
        filter_no_face=not args.skip_face_presence_filter,
        filtered_window_index_out=args.filtered_window_index_out,
    )

    print(f"report_path={args.out_report}")
    print(f"failures_path={args.failures_out}")
    print(f"filtered_window_index_path={args.filtered_window_index_out}")
    print(f"selected_window_count={summary['selected_window_count']}")
    print(f"face_filter_dropped_count={summary['face_filter']['dropped_count']}")
    print(f"face_filter_dropped_no_face_count={summary['face_filter']['dropped_no_face_count']}")
    for modality in ("eeg", "wear", "face", "audio"):
        values = summary["modalities"][modality]
        print(f"{modality}_ready_count={values['ready_count']}")
        print(f"{modality}_missing_count={values['missing_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
