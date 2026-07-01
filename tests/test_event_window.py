import json
import tempfile
import unittest
from pathlib import Path

from daily_multimodal.alignment.event_windows import (
    build_window_index,
    build_window_index_with_summary,
    load_window_index,
)


class EventWindowTests(unittest.TestCase):
    def test_default_builds_twelve_ten_second_windows_over_two_minute_history(self):
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

        windows = build_window_index(rows)

        self.assertEqual(len(windows), 12)
        window = windows[0]
        self.assertEqual(window["sample_id"], "sub-02_ses-03_00_row-0012_win-0000")
        self.assertEqual(window["window_start_time"], "2025-02-28 14:11:10")
        self.assertEqual(window["window_end_time"], "2025-02-28 14:11:20")
        self.assertEqual(window["window_start_offset_seconds"], -120)
        self.assertEqual(window["window_end_offset_seconds"], -110)
        self.assertEqual(windows[-1]["sample_id"], "sub-02_ses-03_00_row-0012_win-0011")
        self.assertEqual(windows[-1]["window_start_offset_seconds"], -10)
        self.assertEqual(windows[-1]["window_end_offset_seconds"], 0)
        self.assertEqual(windows[0]["event_window_start_seconds"], -120)
        self.assertEqual(windows[0]["event_window_end_seconds"], 0)
        self.assertTrue(window["has_wear"])
        self.assertTrue(window["has_face"])
        self.assertTrue(window["has_audio"])
        self.assertEqual(window["label_columns"], {"alert": "3"})

    def test_summary_skips_events_with_less_than_two_minutes_of_pre_event_history(self):
        rows = [
            {
                "event_id": "too-early",
                "absolute_onset_time": "2025-02-28 14:01:59",
                "eeg_recording_start_time": "2025-02-28 14:00:00",
                "eeg_onset_seconds": 119.0,
                "labels": {},
            },
            {
                "event_id": "enough-history",
                "absolute_onset_time": "2025-02-28 14:02:00",
                "eeg_recording_start_time": "2025-02-28 14:00:00",
                "eeg_onset_seconds": 120.0,
                "labels": {},
            },
        ]

        windows, summary = build_window_index_with_summary(rows)

        self.assertEqual(len(windows), 12)
        self.assertEqual(summary["events_total"], 2)
        self.assertEqual(summary["events_selected"], 1)
        self.assertEqual(summary["events_skipped"], 1)
        self.assertEqual(summary["skip_reasons"], {"insufficient_pre_event_history": 1})
        self.assertEqual(summary["skipped_events"][0]["event_id"], "too-early")
        self.assertEqual(summary["skipped_events"][0]["available_history_seconds"], 119.0)

    def test_summary_skips_precise_video_events_without_full_two_minute_coverage(self):
        rows = [
            {
                "event_id": "partial-video",
                "absolute_onset_time": "2025-02-28 14:13:10",
                "eeg_onset_seconds": 180.0,
                "video_candidates": [
                    {
                        "mp4_path": "/data/video.mp4",
                        "overlap_seconds": 60.0,
                        "covers_window": False,
                    }
                ],
                "labels": {},
            }
        ]

        windows, summary = build_window_index_with_summary(rows)

        self.assertEqual(windows, [])
        self.assertEqual(summary["events_skipped"], 1)
        self.assertEqual(summary["skip_reasons"], {"insufficient_video_coverage": 1})
        self.assertEqual(summary["skipped_events"][0]["event_id"], "partial-video")

    def test_window_video_candidates_are_rebased_to_each_ten_second_sample(self):
        rows = [
            {
                "event_id": "event-with-video",
                "absolute_onset_time": "2025-02-28 14:13:10",
                "eeg_onset_seconds": 240.0,
                "video_candidates": [
                    {
                        "mp4_path": "/data/video.mp4",
                        "mp4_start_time": "2025-02-28 14:10:00",
                        "mp4_end_time": "2025-02-28 14:20:00",
                        "duration_seconds": 600.0,
                        "clip_start_seconds": 130.0,
                        "clip_end_seconds": 190.0,
                        "overlap_seconds": 60.0,
                        "covers_window": True,
                        "has_audio_stream": True,
                    }
                ],
                "labels": {},
            }
        ]

        windows = build_window_index(rows)

        first_candidate = windows[0]["video_candidates"][0]
        second_candidate = windows[1]["video_candidates"][0]
        self.assertEqual(first_candidate["clip_start_seconds"], 70.0)
        self.assertEqual(first_candidate["clip_end_seconds"], 80.0)
        self.assertEqual(first_candidate["overlap_seconds"], 10.0)
        self.assertEqual(second_candidate["clip_start_seconds"], 80.0)
        self.assertEqual(second_candidate["clip_end_seconds"], 90.0)

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

    def test_load_window_index_accepts_utf8_bom_jsonl(self):
        row = {
            "sample_id": "sample-cli-1",
            "event_id": "event-cli-1",
            "subject_id": "sub-02",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window_index.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8-sig")

            loaded = load_window_index(path)

        self.assertEqual(loaded, [row])


if __name__ == "__main__":
    unittest.main()
