from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "archive_legacy" / "21_draw_data_overview.py"
    spec = importlib.util.spec_from_file_location("draw_data_overview", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataOverviewFigureTests(unittest.TestCase):
    def test_build_timeline_uses_precise_video_candidates_for_one_day(self):
        module = _load_script_module()
        manifest = [
            {
                "subject_id": "sub-10",
                "session_id": "ses-03",
                "segment_id": "00",
                "rating_row_index": 1,
                "absolute_onset_time": "2025-07-08 18:39:01",
                "eeg_recording_start_time": "2025-07-08T18:24:23",
                "eeg_recording_duration": 13410.0,
                "wear_ppg_path": "/data/Study_UID1_ID1_20250708175811_20250708221732_PPG.csv",
                "wear_gsr_path": "/data/Study_UID1_ID1_20250708175811_20250708221732_GSR.csv",
                "wear_acc_path": "/data/Study_UID1_ID1_20250708175811_20250708221732_ACC.csv",
                "is_complete_multimodal_candidate": True,
                "labels": {"fatigue": "1"},
            },
            {
                "subject_id": "sub-10",
                "session_id": "ses-04",
                "segment_id": "00",
                "rating_row_index": 1,
                "absolute_onset_time": "2025-07-09 18:39:01",
                "eeg_recording_start_time": "2025-07-09T18:24:23",
                "eeg_recording_duration": 100.0,
                "wear_ppg_path": "",
                "wear_gsr_path": "",
                "wear_acc_path": "",
                "is_complete_multimodal_candidate": True,
                "labels": {"fatigue": "5"},
            },
        ]
        windows = [
            {
                "sample_id": "sub-10_ses-03_00_row-0001_win-0000",
                "subject_id": "sub-10",
                "session_id": "ses-03",
                "absolute_onset_time": "2025-07-08 18:39:01",
                "window_start_time": "2025-07-08 18:38:01",
                "window_end_time": "2025-07-08 18:39:01",
                "label_columns": {"fatigue": "1"},
                "video_candidates": [
                    {
                        "mp4_path": "/video/DJI_0129.MP4",
                        "mp4_start_time": "2025-07-08 18:34:47",
                        "mp4_end_time": "2025-07-08 18:55:48.024",
                        "covers_window": True,
                        "has_audio_stream": True,
                    }
                ],
            },
            {
                "sample_id": "sub-10_ses-04_00_row-0001_win-0000",
                "subject_id": "sub-10",
                "session_id": "ses-04",
                "absolute_onset_time": "2025-07-09 18:39:01",
                "window_start_time": "2025-07-09 18:38:01",
                "window_end_time": "2025-07-09 18:39:01",
                "label_columns": {"fatigue": "5"},
                "video_candidates": [],
            },
        ]

        summary, intervals, event_rows, _, _ = module.build_timeline(
            manifest=manifest,
            windows=windows,
            subject_id="sub-10",
            date="2025-07-08",
        )

        self.assertEqual(summary["date"], "2025-07-08")
        self.assertEqual(summary["events_total"], 1)
        self.assertEqual(summary["precise_video_full_coverage"], 1)
        self.assertEqual(summary["precise_audio_full_coverage"], 1)
        self.assertEqual(summary["interval_counts"]["Precise MP4"], 1)
        self.assertEqual(summary["interval_counts"]["Audio stream"], 1)
        self.assertEqual(event_rows[0]["timeline_label"], "18:39 f=1 DJI_0129.MP4")
        self.assertEqual(intervals["Precise MP4"][0][2], "DJI_0129.MP4")


if __name__ == "__main__":
    unittest.main()
