from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.event_windows import (
    DEFAULT_END_SECONDS,
    DEFAULT_START_SECONDS,
    DEFAULT_STRIDE_SECONDS,
    DEFAULT_WINDOW_SIZE_SECONDS,
    build_window_index_with_summary,
    save_window_index,
    save_window_index_summary,
)
from daily_multimodal.manifest.validate_manifest import load_jsonl_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build window index from event manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/window_index/window_index.jsonl")
    parser.add_argument("--summary-out", default="outputs/reports/window_index_summary.json")
    parser.add_argument("--start-seconds", type=float, default=DEFAULT_START_SECONDS)
    parser.add_argument("--end-seconds", type=float, default=DEFAULT_END_SECONDS)
    parser.add_argument("--window-size-seconds", type=float, default=DEFAULT_WINDOW_SIZE_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--require-all-modalities", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl_manifest(args.manifest)
    windows, summary = build_window_index_with_summary(
        rows,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        window_size_seconds=args.window_size_seconds,
        stride_seconds=args.stride_seconds,
        require_all_modalities=args.require_all_modalities,
    )
    out = save_window_index(windows, args.out)
    summary_out = save_window_index_summary(summary, args.summary_out)
    print(f"window_index_path={out}")
    print(f"window_index_summary_path={summary_out}")
    print(f"windows_total={len(windows)}")
    print(f"events_selected={summary['events_selected']}")
    print(f"events_skipped={summary['events_skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
