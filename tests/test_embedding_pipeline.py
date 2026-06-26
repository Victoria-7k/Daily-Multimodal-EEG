import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.embeddings.basic import extract_basic_embedding
from daily_multimodal.embeddings.pipeline import (
    extract_many_basic_embeddings,
    save_embedding_batch,
)


class BasicEmbeddingPipelineTests(unittest.TestCase):
    def test_extract_basic_embedding_returns_unified_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ppg = root / "ppg.csv"
            gsr = root / "gsr.csv"
            acc = root / "acc.csv"
            ppg.write_text(
                "PPG,csv_time_PPG\n"
                "1,2025-02-28 14:13:00\n"
                "3,2025-02-28 14:13:05\n",
                encoding="utf-8",
            )
            gsr.write_text(
                "GSR,csv_time_GSR\n"
                "2,2025-02-28 14:13:01\n",
                encoding="utf-8",
            )
            acc.write_text(
                "Motion_dataX,Motion_dataY,Motion_dataZ,csv_time_motion\n"
                "0,0,1,2025-02-28 14:13:09\n",
                encoding="utf-8",
            )
            window = {
                "sample_id": "sub-02_ses-03_00_row-0012_win-0000",
                "event_id": "sub-02_ses-03_00_row-0012",
                "subject_id": "sub-02",
                "session_id": "ses-03",
                "window_start_time": "2025-02-28 14:13:00",
                "window_end_time": "2025-02-28 14:13:10",
                "eeg_bdf_path": "",
                "wear_ppg_path": str(ppg),
                "wear_gsr_path": str(gsr),
                "wear_acc_path": str(acc),
                "candidate_mp4_paths": [],
                "candidate_audio_paths": [],
                "label_columns": {"alert": "3"},
                "has_eeg": False,
                "has_ppg": True,
                "has_gsr": True,
                "has_acc": True,
                "has_face": False,
                "has_audio": False,
            }

            sample = extract_basic_embedding(window)

        self.assertEqual(sample.sample_id, "sub-02_ses-03_00_row-0012_win-0000")
        self.assertEqual(sample.eeg_emb.shape, (256,))
        self.assertEqual(sample.wear_emb.shape, (256,))
        self.assertEqual(sample.face_emb.shape, (256,))
        self.assertEqual(sample.audio_emb.shape, (256,))
        np.testing.assert_array_equal(sample.modality_mask, np.array([0, 1, 0, 0], dtype=np.int8))
        self.assertEqual(sample.labels, {"alert": "3"})
        self.assertEqual(sample.quality_flags["wear"]["ppg_rows"], 2)
        self.assertFalse(np.all(sample.wear_emb == 0.0))
        self.assertTrue(np.all(sample.eeg_emb == 0.0))

    def test_extract_many_saves_npz_and_report(self):
        windows = [
            {
                "sample_id": f"sample-{idx:04d}",
                "event_id": f"event-{idx:04d}",
                "subject_id": "sub-02",
                "session_id": "ses-01",
                "window_start_time": "2025-02-28 14:13:00",
                "window_end_time": "2025-02-28 14:13:10",
                "label_columns": {"alert": str(idx)},
                "has_eeg": False,
                "has_ppg": False,
                "has_gsr": False,
                "has_acc": False,
                "has_face": False,
                "has_audio": False,
            }
            for idx in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_npz = Path(tmp) / "embeddings.npz"
            report_out = Path(tmp) / "report.json"
            batch = extract_many_basic_embeddings(windows, max_windows=2)
            save_embedding_batch(batch, out_npz, report_out)
            with np.load(out_npz, allow_pickle=True) as loaded:
                eeg_shape = loaded["eeg_emb"].shape
                wear_shape = loaded["wear_emb"].shape
                mask_shape = loaded["modality_mask"].shape
            report = json.loads(report_out.read_text(encoding="utf-8"))

        self.assertEqual(batch.summary["requested_windows"], 2)
        self.assertEqual(batch.summary["success_count"], 2)
        self.assertEqual(eeg_shape, (2, 256))
        self.assertEqual(wear_shape, (2, 256))
        self.assertEqual(mask_shape, (2, 4))
        self.assertEqual(report["summary"]["success_count"], 2)


if __name__ == "__main__":
    unittest.main()
