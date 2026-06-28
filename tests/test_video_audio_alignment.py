import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from daily_multimodal.alignment.video_audio_alignment import (
    align_video_audio_rows,
    probe_many_mp4_paths,
    run_ffprobe,
)


class VideoAudioAlignmentTests(unittest.TestCase):
    def test_aligns_mp4_creation_time_to_event_window_clip(self):
        rows = [
            {
                "event_id": "sub-02_ses-03_00_row-0012",
                "absolute_onset_time": "2025-02-28 14:13:10",
                "candidate_mp4_paths": ["/data/sub2/0228/clip.MP4"],
                "has_video": True,
            }
        ]

        def fake_probe(path):
            self.assertEqual(path, "/data/sub2/0228/clip.MP4")
            return {
                "format": {
                    "duration": "90.0",
                    "tags": {"creation_time": "2025-02-28T06:12:00.000000Z"},
                },
                "streams": [
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                    }
                ],
            }

        enriched, report = align_video_audio_rows(
            rows,
            start_seconds=-60,
            end_seconds=0,
            ffprobe_func=fake_probe,
        )

        candidate = enriched[0]["video_candidates"][0]
        self.assertEqual(candidate["mp4_start_time"], "2025-02-28 14:12:00")
        self.assertEqual(candidate["mp4_end_time"], "2025-02-28 14:13:30")
        self.assertEqual(candidate["clip_start_seconds"], 10.0)
        self.assertEqual(candidate["clip_end_seconds"], 70.0)
        self.assertEqual(candidate["overlap_seconds"], 60.0)
        self.assertTrue(candidate["covers_window"])
        self.assertTrue(candidate["has_audio_stream"])
        self.assertEqual(candidate["audio_sample_rate"], 48000)
        self.assertTrue(enriched[0]["has_precise_video"])
        self.assertTrue(enriched[0]["has_precise_audio"])
        self.assertEqual(report["events_total"], 1)
        self.assertEqual(report["events_with_precise_video_overlap"], 1)
        self.assertEqual(report["events_with_full_window_video_coverage"], 1)

    def test_run_ffprobe_uses_timeout_and_narrow_metadata_query(self):
        with patch("daily_multimodal.alignment.video_audio_alignment.subprocess.run") as run:
            run.return_value.stdout = '{"format": {}, "streams": []}'

            run_ffprobe("/data/clip.MP4", timeout_seconds=7)

        command = run.call_args.args[0]
        self.assertIn("-show_entries", command)
        self.assertIn("format=duration:format_tags=creation_time", " ".join(command))
        self.assertEqual(run.call_args.kwargs["timeout"], 7)

    def test_probe_many_mp4_paths_writes_and_reuses_cache(self):
        calls = []

        def fake_probe(path, *, timeout_seconds):
            calls.append((path, timeout_seconds))
            if path.endswith("bad.MP4"):
                raise TimeoutError("slow file")
            return {"format": {"duration": "1.0"}, "streams": []}

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "ffprobe_cache.jsonl"
            first_cache, first_report = probe_many_mp4_paths(
                ["/data/good.MP4", "/data/bad.MP4", "/data/good.MP4"],
                timeout_seconds=3,
                max_workers=1,
                cache_path=cache_path,
                ffprobe_func=fake_probe,
            )
            second_cache, second_report = probe_many_mp4_paths(
                ["/data/good.MP4", "/data/bad.MP4"],
                timeout_seconds=3,
                max_workers=1,
                cache_path=cache_path,
                ffprobe_func=fake_probe,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(first_report["unique_mp4_total"], 2)
        self.assertEqual(first_report["ffprobe_success_files"], 1)
        self.assertEqual(first_report["ffprobe_failed_files"], 1)
        self.assertEqual(second_report["cache_hits"], 2)
        self.assertEqual(second_report["newly_probed"], 0)
        self.assertIn("/data/good.MP4", first_cache)
        self.assertIn("/data/bad.MP4", second_cache)

    def test_probe_many_mp4_paths_can_retry_failed_cache_records(self):
        attempts = []

        def fake_probe(path, *, timeout_seconds):
            attempts.append((path, timeout_seconds))
            return {"format": {"duration": "2.0"}, "streams": []}

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "ffprobe_cache.jsonl"
            cache_path.write_text(
                '{"mp4_path": "/data/slow.MP4", "ok": false, '
                '"error_type": "TimeoutExpired", "error": "timed out"}\n',
                encoding="utf-8",
            )

            cache, report = probe_many_mp4_paths(
                ["/data/slow.MP4"],
                timeout_seconds=None,
                retry_failed=True,
                max_workers=1,
                cache_path=cache_path,
                ffprobe_func=fake_probe,
            )

        self.assertEqual(attempts, [("/data/slow.MP4", None)])
        self.assertTrue(cache["/data/slow.MP4"]["ok"])
        self.assertEqual(report["cache_hits"], 0)
        self.assertEqual(report["newly_probed"], 1)
        self.assertEqual(report["ffprobe_success_files"], 1)


if __name__ == "__main__":
    unittest.main()
