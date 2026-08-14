import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from daily_multimodal.alignment.eeg_coverage import (
    classify_eeg_window_coverage,
    summarize_eeg_coverage,
)


class EEGCoverageTests(unittest.TestCase):
    def test_classifies_in_range_window(self):
        result = classify_eeg_window_coverage(
            start_offset_seconds=5.0,
            end_offset_seconds=15.0,
            bdf_duration_seconds=30.0,
        )

        self.assertEqual(result["classification"], "in_range")
        self.assertFalse(result["whole_day_shift_candidate"])

    def test_classifies_negative_offset(self):
        result = classify_eeg_window_coverage(
            start_offset_seconds=-20.0,
            end_offset_seconds=-10.0,
            bdf_duration_seconds=30.0,
        )

        self.assertEqual(result["classification"], "negative_offset")

    def test_classifies_after_recording_end(self):
        result = classify_eeg_window_coverage(
            start_offset_seconds=40.0,
            end_offset_seconds=50.0,
            bdf_duration_seconds=30.0,
        )

        self.assertEqual(result["classification"], "after_recording_end")

    def test_classifies_partial_overlap(self):
        result = classify_eeg_window_coverage(
            start_offset_seconds=25.0,
            end_offset_seconds=35.0,
            bdf_duration_seconds=30.0,
        )

        self.assertEqual(result["classification"], "partial_overlap")

    def test_classifies_whole_day_shift_candidate(self):
        result = classify_eeg_window_coverage(
            start_offset_seconds=86405.0,
            end_offset_seconds=86415.0,
            bdf_duration_seconds=30.0,
        )

        self.assertEqual(result["classification"], "whole_day_shift_candidate")
        self.assertEqual(result["suggested_shift_seconds"], -86400)

    def test_summarizes_affected_subject_sessions(self):
        rows = [
            _row("sample-1", "sub-02", "ses-01", 5, 15, 30),
            _row("sample-2", "sub-02", "ses-02", 40, 50, 30),
            _row("sample-3", "sub-02", "ses-02", -20, -10, 30),
        ]

        summary = summarize_eeg_coverage(rows)

        self.assertEqual(summary["total_windows"], 3)
        self.assertEqual(summary["in_range_count"], 1)
        self.assertEqual(summary["after_recording_end_count"], 1)
        self.assertEqual(summary["negative_offset_count"], 1)
        self.assertEqual(summary["affected_subject_sessions"], ["sub-02/ses-02"])

    def test_cli_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "window_index.jsonl"
            window_index.write_text(
                "\n".join(
                    [
                        json.dumps(_row("sample-1", "sub-02", "ses-01", 5, 15, 30)),
                        json.dumps(_row("sample-2", "sub-03", "ses-01", 86405, 86415, 30)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out_json = root / "audit.json"
            out_table = root / "audit.md"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/archive_legacy/19_audit_eeg_coverage.py",
                    "--window-index",
                    str(window_index),
                    "--out-json",
                    str(out_json),
                    "--out-table",
                    str(out_table),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            summary = json.loads(out_json.read_text(encoding="utf-8"))
            table = out_table.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("total_windows=2", completed.stdout)
        self.assertEqual(summary["whole_day_shift_candidate_count"], 1)
        self.assertIn("| classification | count |", table)


def _row(sample_id, subject_id, session_id, start_offset, end_offset, duration):
    return {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": subject_id,
        "session_id": session_id,
        "eeg_bdf_path": f"/tmp/{subject_id}_{session_id}.bdf",
        "window_start_offset_seconds": start_offset,
        "window_end_offset_seconds": end_offset,
        "eeg_recording_duration_seconds": duration,
    }


if __name__ == "__main__":
    unittest.main()
