import unittest

from daily_multimodal.alignment.event_windows import build_window_index


class EventWindowTests(unittest.TestCase):
    def test_builds_single_base_window_before_event_onset(self):
        rows = [
            {
                "event_id": "sub-02_ses-03_00_row-0012",
                "subject_id": "sub-02",
                "session_id": "ses-03",
                "segment_id": "00",
                "absolute_onset_time": "2025-02-28 14:13:10",
                "eeg_recording_start_time": "2025-02-28T13:55:00",
                "eeg_onset_seconds": 1090.0,
                "eeg_bdf_path": "/data/eeg.bdf",
                "wear_ppg_path": "/data/ppg.csv",
                "wear_gsr_path": "/data/gsr.csv",
                "wear_acc_path": "/data/acc.csv",
                "candidate_mp4_paths": ["/data/video.MP4"],
                "has_eeg": True,
                "has_ppg": True,
                "has_gsr": True,
                "has_acc": True,
                "has_video": True,
                "has_audio": True,
                "labels": {"alert": "3"},
                "activity_category": "In-work",
                "social_presence": "Y",
            }
        ]

        windows = build_window_index(
            rows,
            start_seconds=-10,
            end_seconds=0,
            window_size_seconds=10,
            stride_seconds=5,
        )

        self.assertEqual(len(windows), 1)
        window = windows[0]
        self.assertEqual(window["sample_id"], "sub-02_ses-03_00_row-0012_win-0000")
        self.assertEqual(window["window_start_time"], "2025-02-28 14:13:00")
        self.assertEqual(window["window_end_time"], "2025-02-28 14:13:10")
        self.assertEqual(window["window_start_offset_seconds"], -10)
        self.assertEqual(window["window_end_offset_seconds"], 0)
        self.assertTrue(window["has_wear"])
        self.assertTrue(window["has_face"])
        self.assertTrue(window["has_audio"])
        self.assertEqual(window["label_columns"], {"alert": "3"})

    def test_builds_sliding_windows_when_range_is_larger_than_window(self):
        rows = [
            {
                "event_id": "sub-02_ses-03_00_row-0012",
                "subject_id": "sub-02",
                "session_id": "ses-03",
                "segment_id": "00",
                "absolute_onset_time": "2025-02-28 14:13:10",
                "candidate_mp4_paths": [],
                "has_eeg": True,
                "has_ppg": False,
                "has_gsr": False,
                "has_acc": False,
                "has_video": False,
                "has_audio": False,
                "labels": {},
            }
        ]

        windows = build_window_index(
            rows,
            start_seconds=-20,
            end_seconds=0,
            window_size_seconds=10,
            stride_seconds=5,
        )

        self.assertEqual(
            [window["window_start_offset_seconds"] for window in windows],
            [-20, -15, -10],
        )
        self.assertEqual(windows[-1]["window_end_time"], "2025-02-28 14:13:10")


if __name__ == "__main__":
    unittest.main()
