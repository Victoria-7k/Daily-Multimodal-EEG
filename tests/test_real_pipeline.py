import json
import tempfile
import unittest
from pathlib import Path

from daily_multimodal.embeddings.cache import (
    RealCacheProfiles,
    build_cache_key,
    prepare_real_embedding_cache,
)


class RealPipelineCacheTests(unittest.TestCase):
    def test_build_cache_key_is_stable_and_rejects_path_traversal(self):
        self.assertEqual(
            build_cache_key("sub-02_row-1_win-0000", "audio", "wavlm_frozen_v1"),
            "sub-02_row-1_win-0000/audio/wavlm_frozen_v1",
        )

        with self.assertRaisesRegex(ValueError, "cache key"):
            build_cache_key("../escape", "audio", "wavlm_frozen_v1")

    def test_prepare_real_embedding_cache_writes_modality_cache_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            source_root.mkdir()
            video = _touch(source_root / "clip.mp4")
            eeg = _touch(source_root / "sample.bdf")
            ppg = _touch(source_root / "ppg.csv")
            gsr = _touch(source_root / "gsr.csv")
            acc = _touch(source_root / "acc.csv")
            cache_root = root / "cache"
            report_out = root / "reports" / "real_embedding_readiness_report.md"
            failures_out = root / "reports" / "real_embedding_failures.json"
            extracted_audio: list[Path] = []

            def fake_audio_extractor(source, start_seconds, end_seconds, output):
                self.assertEqual(source, video)
                self.assertEqual(start_seconds, 1.25)
                self.assertEqual(end_seconds, 4.75)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"wav")
                extracted_audio.append(output)

            summary = prepare_real_embedding_cache(
                [_window(video, eeg, ppg, gsr, acc)],
                cache_root=cache_root,
                report_out=report_out,
                failures_out=failures_out,
                profiles=RealCacheProfiles(),
                audio_extractor=fake_audio_extractor,
            )

            failures = json.loads(failures_out.read_text(encoding="utf-8"))
            report = report_out.read_text(encoding="utf-8")

        self.assertEqual(summary["modalities"]["audio"]["ready_count"], 1)
        self.assertEqual(summary["modalities"]["face"]["ready_count"], 1)
        self.assertEqual(summary["modalities"]["eeg"]["ready_count"], 1)
        self.assertEqual(summary["modalities"]["wear"]["ready_count"], 1)
        self.assertEqual(failures, [])
        self.assertEqual(len(extracted_audio), 1)
        self.assertIn("Audio ready: 1", report)
        self.assertIn("Wear ready: 1", report)

    def test_prepare_real_embedding_cache_records_explicit_failures_for_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = prepare_real_embedding_cache(
                [_window(root / "missing.mp4", root / "missing.bdf", root / "ppg.csv", root / "gsr.csv", root / "acc.csv")],
                cache_root=root / "cache",
                report_out=root / "readiness.md",
                failures_out=root / "failures.json",
                profiles=RealCacheProfiles(),
                audio_extractor=lambda source, start_seconds, end_seconds, output: None,
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["modalities"]["audio"]["ready_count"], 0)
        self.assertEqual(summary["modalities"]["audio"]["missing_count"], 1)
        self.assertGreaterEqual(len(failures), 4)
        self.assertTrue(all(failure["sample_id"] == "sample-1" for failure in failures))
        self.assertIn("source_missing", {failure["error_type"] for failure in failures})


def _touch(path: Path) -> Path:
    path.write_text("x", encoding="utf-8")
    return path


def _window(video: Path, eeg: Path, ppg: Path, gsr: Path, acc: Path) -> dict:
    return {
        "sample_id": "sample-1",
        "event_id": "event-1",
        "subject_id": "sub-02",
        "window_start_time": "2025-02-28 14:13:00",
        "window_end_time": "2025-02-28 14:13:10",
        "eeg_bdf_path": str(eeg),
        "wear_ppg_path": str(ppg),
        "wear_gsr_path": str(gsr),
        "wear_acc_path": str(acc),
        "video_candidates": [
            {
                "mp4_path": str(video),
                "clip_start_seconds": 1.25,
                "clip_end_seconds": 4.75,
                "covers_window": True,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
