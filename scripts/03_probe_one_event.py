from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.event_windows import (
    build_window_index,
    load_window_index,
    save_window_index,
)
from daily_multimodal.alignment.probe import (
    build_probe_report,
    save_probe_report,
    save_shapes_report,
)
from daily_multimodal.manifest.validate_manifest import load_jsonl_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe one multimodal event window.")
    parser.add_argument("--manifest", help="Event manifest JSONL.")
    parser.add_argument("--window-index", help="Existing window index JSONL.")
    parser.add_argument("--window-index-out", default="outputs/window_index/window_index.jsonl")
    parser.add_argument("--report-out", default="outputs/reports/probe_one_event.json")
    parser.add_argument("--shapes-out", default="outputs/reports/probe_one_event_shapes.txt")
    parser.add_argument("--sample-id")
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--start-seconds", type=float, default=-10)
    parser.add_argument("--end-seconds", type=float, default=0)
    parser.add_argument("--window-size-seconds", type=float, default=10)
    parser.add_argument("--stride-seconds", type=float, default=5)
    parser.add_argument("--eeg-resample-hz", type=int, default=250)
    args = parser.parse_args()

    if args.window_index:
        windows = load_window_index(args.window_index)
    elif args.manifest:
        rows = load_jsonl_manifest(args.manifest)
        windows = build_window_index(
            rows,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            window_size_seconds=args.window_size_seconds,
            stride_seconds=args.stride_seconds,
        )
        save_window_index(windows, args.window_index_out)
    else:
        parser.error("Provide either --manifest or --window-index.")

    if not windows:
        raise SystemExit("No windows available for probing.")

    window = _select_window(windows, args.sample_id, args.require_all_modalities)
    report = build_probe_report(window, eeg_resample_hz=args.eeg_resample_hz)
    report_path = save_probe_report(report, args.report_out)
    shapes_path = save_shapes_report(report, args.shapes_out)
    print(f"sample_id={report['sample_id']}")
    print(f"probe_report_path={report_path}")
    print(f"probe_shapes_path={shapes_path}")
    print(f"window_index_path={args.window_index or args.window_index_out}")
    return 0


def _select_window(windows: list[dict], sample_id: str | None, require_all_modalities: bool) -> dict:
    if sample_id:
        for window in windows:
            if window.get("sample_id") == sample_id:
                if require_all_modalities and not _has_all_modalities(window):
                    raise SystemExit(f"sample_id is not complete multimodal: {sample_id}")
                return window
        raise SystemExit(f"sample_id not found: {sample_id}")
    for window in windows:
        if not require_all_modalities or _has_all_modalities(window):
            return window
    raise SystemExit("No complete multimodal window available for probing.")


def _has_all_modalities(window: dict) -> bool:
    return all(
        bool(window.get(key))
        for key in ["has_eeg", "has_ppg", "has_gsr", "has_acc", "has_face", "has_audio"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
