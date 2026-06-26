from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.event_windows import build_window_index, load_window_index
from daily_multimodal.embeddings.pipeline import (
    extract_many_basic_embeddings,
    save_embedding_batch,
)
from daily_multimodal.manifest.validate_manifest import load_jsonl_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract stage-6 basic smoke embeddings.")
    parser.add_argument("--manifest", help="Event manifest JSONL.")
    parser.add_argument("--window-index", help="Existing window index JSONL.")
    parser.add_argument("--max-events", type=int, default=10)
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--encoder-profile", default="basic", choices=["basic"])
    parser.add_argument("--out", default="outputs/embeddings/smoke_10_events_basic_embeddings.npz")
    parser.add_argument("--report-out", default="outputs/reports/smoke_10_events_basic_report.json")
    args = parser.parse_args()

    windows = _load_or_build_windows(args.manifest, args.window_index)
    if args.require_all_modalities:
        windows = [window for window in windows if _has_all_modalities(window)]
    batch = extract_many_basic_embeddings(windows, max_windows=args.max_events)
    npz_path, report_path = save_embedding_batch(batch, args.out, args.report_out)
    print(f"embedding_path={npz_path}")
    print(f"report_path={report_path}")
    print(f"requested_windows={batch.summary['requested_windows']}")
    print(f"success_count={batch.summary['success_count']}")
    print(f"failure_count={batch.summary['failure_count']}")
    return 0 if not batch.failures else 1


def _load_or_build_windows(manifest: str | None, window_index: str | None) -> list[dict]:
    if window_index:
        return load_window_index(window_index)
    if manifest:
        rows = load_jsonl_manifest(manifest)
        return build_window_index(
            rows,
            start_seconds=-10,
            end_seconds=0,
            window_size_seconds=10,
            stride_seconds=5,
        )
    raise SystemExit("Provide either --manifest or --window-index.")


def _has_all_modalities(window: dict) -> bool:
    return all(
        bool(window.get(key))
        for key in ["has_eeg", "has_ppg", "has_gsr", "has_acc", "has_face", "has_audio"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
