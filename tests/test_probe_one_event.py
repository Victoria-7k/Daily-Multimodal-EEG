import tempfile
import unittest
from pathlib import Path

from daily_multimodal.alignment.probe import build_probe_report


class ProbeOneEventTests(unittest.TestCase):
    def test_probe_report_counts_wear_rows_and_describes_expected_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ppg = root / "ppg.csv"
            gsr = root / "gsr.csv"
            acc = root / "acc.csv"
            bdf = root / "sample.bdf"
            mp4 = root / "sample.MP4"
            bdf.write_bytes(b"fake")
            mp4.write_bytes(b"fake")
            ppg.write_text(
                "PPG,csv_time_PPG\n"
                "1,2025-02-28 14:12:59\n"
                "2,2025-02-28 14:13:00\n"
                "3,2025-02-28 14:13:05\n",
                encoding="utf-8",
            )
            gsr.write_text(
                "GSR,csv_time_GSR\n"
                "1,2025-02-28 14:13:01\n",
                encoding="utf-8",
            )
            acc.write_text(
                "Motion_dataX,Motion_dataY,Motion_dataZ,csv_time_motion\n"
                "0,0,1,2025-02-28 14:13:09.500\n",
                encoding="utf-8",
            )
            window = {
                "sample_id": "sub-02_ses-03_00_row-0012_win-0000",
                "event_id": "sub-02_ses-03_00_row-0012",
                "subject_id": "sub-02",
                "session_id": "ses-03",
                "window_start_time": "2025-02-28 14:13:00",
                "window_end_time": "2025-02-28 14:13:10",
                "window_size_seconds": 10,
                "eeg_recording_start_time": "2025-02-28T13:55:00",
                "eeg_bdf_path": str(bdf),
                "wear_ppg_path": str(ppg),
                "wear_gsr_path": str(gsr),
                "wear_acc_path": str(acc),
                "candidate_mp4_paths": [str(mp4)],
                "has_eeg": True,
                "has_ppg": True,
                "has_gsr": True,
                "has_acc": True,
                "has_face": True,
                "has_audio": True,
            }

            report = build_probe_report(window, eeg_resample_hz=250)

        self.assertEqual(report["sample_id"], "sub-02_ses-03_00_row-0012_win-0000")
        self.assertEqual(report["eeg"]["expected_resampled_shape"], ["channels_unknown", 2500])
        self.assertEqual(report["eeg"]["window_start_offset_seconds"], 1080.0)
        self.assertEqual(report["wear"]["ppg"]["rows_in_window"], 2)
        self.assertEqual(report["wear"]["gsr"]["rows_in_window"], 1)
        self.assertEqual(report["wear"]["acc"]["rows_in_window"], 1)
        self.assertTrue(report["video"]["candidate_count"] == 1)
        self.assertTrue(report["audio"]["candidate_count"] == 1)


if __name__ == "__main__":
    unittest.main()
