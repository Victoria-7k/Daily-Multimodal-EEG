from __future__ import annotations

import argparse
import json
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
from daily_multimodal.embeddings.subject import select_subject_windows, summarize_subject_selection
from daily_multimodal.manifest.validate_manifest import load_jsonl_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract stage-7 basic embeddings for one subject.")
    parser.add_argument("--manifest", help="Event manifest JSONL.")
    parser.add_argument("--window-index", help="Existing window index JSONL.")
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--encoder-profile", default="basic", choices=["basic"])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--out")
    parser.add_argument("--report-out")
    args = parser.parse_args()

    windows = _load_or_build_windows(args.manifest, args.window_index)
    selected = select_subject_windows(
        windows,
        subject_id=args.subject_id,
        require_all_modalities=args.require_all_modalities,
    )
    batch = extract_many_basic_embeddings(selected, max_windows=args.max_windows)
    selection_summary = summarize_subject_selection(windows, selected, subject_id=args.subject_id)
    batch.summary = {**selection_summary, **batch.summary}
    out = args.out or f"outputs/embeddings/{args.subject_id}_basic_embeddings.npz"
    report_out = args.report_out or f"outputs/reports/{args.subject_id}_basic_report.json"
    npz_path, report_path = save_embedding_batch(batch, out, report_out)
    _append_stage7_report_fields(report_path, require_all_modalities=args.require_all_modalities)
    print(f"embedding_path={npz_path}")
    print(f"report_path={report_path}")
    print(f"subject_id={args.subject_id}")
    print(f"selected_windows={len(selected)}")
    print(f"requested_windows={batch.summary['requested_windows']}")
    print(f"success_count={batch.summary['success_count']}")
    print(f"failure_count={batch.summary['failure_count']}")
    return 0 if not batch.failures and selected else 1


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


def _append_stage7_report_fields(report_path: Path, *, require_all_modalities: bool) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["stage"] = 7
    report["require_all_modalities"] = require_all_modalities
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
