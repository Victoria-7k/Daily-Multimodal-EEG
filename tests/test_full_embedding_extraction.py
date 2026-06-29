import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.embeddings.full_extract import run_full_basic_extraction


class FullEmbeddingExtractionTests(unittest.TestCase):
    def test_run_full_basic_extraction_keeps_complete_windows_and_writes_outputs(self):
        windows = [
            _window("sample-complete", "sub-02", has_all=True),
            _window("sample-missing-face", "sub-02", has_all=True, has_face=False),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "window_index.jsonl"
            embedding_out = root / "all_complete_basic_embeddings.npz"
            manifest_out = root / "all_complete_multimodal_manifest.jsonl"
            report_out = root / "all_complete_basic_embedding_report.json"
            failures_out = root / "all_complete_basic_embedding_failures.json"
            window_index.write_text(
                "\n".join(json.dumps(window) for window in windows) + "\n",
                encoding="utf-8",
            )

            summary = run_full_basic_extraction(
                window_index=window_index,
                output_npz=embedding_out,
                manifest_out=manifest_out,
                report_out=report_out,
                failures_out=failures_out,
                require_all_modalities=True,
            )

            with np.load(embedding_out, allow_pickle=True) as loaded:
                sample_ids = loaded["sample_id"].tolist()
                mask = loaded["modality_mask"]
            selected_rows = [
                json.loads(line)
                for line in manifest_out.read_text(encoding="utf-8").splitlines()
                if line
            ]
            report = json.loads(report_out.read_text(encoding="utf-8"))
            failures = json.loads(failures_out.read_text(encoding="utf-8"))

        self.assertEqual(summary["selected_windows"], 1)
        self.assertEqual(sample_ids, ["sample-complete"])
        self.assertEqual(mask.shape, (1, 4))
        self.assertEqual(selected_rows[0]["sample_id"], "sample-complete")
        self.assertEqual(report["stage"], 8)
        self.assertEqual(report["summary"]["success_count"], 1)
        self.assertEqual(failures, [])


def _window(sample_id, subject_id, *, has_all, **overrides):
    window = {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": subject_id,
        "session_id": "ses-01",
        "window_start_time": "2025-02-28 14:13:00",
        "window_end_time": "2025-02-28 14:13:10",
        "label_columns": {"alert": "3"},
        "has_eeg": has_all,
        "has_ppg": has_all,
        "has_gsr": has_all,
        "has_acc": has_all,
        "has_face": has_all,
        "has_audio": has_all,
    }
    window.update(overrides)
    return window


if __name__ == "__main__":
    unittest.main()
