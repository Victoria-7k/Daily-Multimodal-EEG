from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.eeg_coverage import summarize_eeg_coverage
from daily_multimodal.alignment.event_windows import load_window_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit EEG window offsets against BDF recording duration.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--out-json", default="outputs/reports/eeg_coverage_audit_full.json")
    parser.add_argument("--out-table", default="outputs/reports/eeg_coverage_audit_full.md")
    args = parser.parse_args()

    windows = load_window_index(args.window_index)
    summary = summarize_eeg_coverage(windows)
    _write_json(summary, args.out_json)
    _write_table(summary, args.out_table)
    print(f"total_windows={summary['total_windows']}")
    print(f"audited_windows={summary['audited_windows']}")
    print(f"in_range_count={summary['in_range_count']}")
    print(f"negative_offset_count={summary['negative_offset_count']}")
    print(f"after_recording_end_count={summary['after_recording_end_count']}")
    print(f"whole_day_shift_candidate_count={summary['whole_day_shift_candidate_count']}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


def _write_json(summary: dict, output: Path | str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_table(summary: dict, output: Path | str) -> None:
    rows = [
        "| classification | count |",
        "| --- | ---: |",
        f"| in_range | {summary['in_range_count']} |",
        f"| negative_offset | {summary['negative_offset_count']} |",
        f"| after_recording_end | {summary['after_recording_end_count']} |",
        f"| partial_overlap | {summary['partial_overlap_count']} |",
        f"| whole_day_shift_candidate | {summary['whole_day_shift_candidate_count']} |",
        f"| out_of_range | {summary['out_of_range_count']} |",
        f"| missing_duration | {summary['missing_duration_count']} |",
        "",
        "## Affected Subject Sessions",
        "",
    ]
    sessions = summary.get("affected_subject_sessions", [])
    rows.extend([f"- {session}" for session in sessions] or ["- None"])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
