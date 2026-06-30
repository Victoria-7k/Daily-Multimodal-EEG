import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.embeddings.face_real import extract_face_real_embeddings


class FaceRealEmbeddingTests(unittest.TestCase):
    def test_extract_openface_stats_writes_npz_and_empty_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            csv_path = _write_face_cache(cache_root, sample_id="sample-1")
            _write_openface_csv(csv_path)

            summary = extract_face_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "face_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="face_raw_openface_stats_v1",
            )

            with np.load(root / "face_real_embeddings.npz", allow_pickle=True) as loaded:
                sample_ids = loaded["sample_id"].astype(str).tolist()
                face_emb = loaded["face_emb"]
                modality_mask = loaded["modality_mask"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(sample_ids, ["sample-1"])
        self.assertEqual(face_emb.shape, (1, 256))
        self.assertEqual(face_emb.dtype, np.float32)
        self.assertFalse(np.isnan(face_emb).any())
        self.assertEqual(modality_mask.tolist(), [[0, 0, 1, 0]])
        self.assertEqual(quality_flags[0]["csv_path"], str(csv_path))
        self.assertAlmostEqual(quality_flags[0]["face_detection_success_rate"], 0.75)
        self.assertEqual(quality_flags[0]["frame_count"], 4)
        self.assertEqual(failures, [])

    def test_missing_face_cache_records_source_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            summary = extract_face_real_embeddings(
                [_window("sample-1")],
                cache_root=root / "cache",
                output_npz=root / "face_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="face_raw_openface_stats_v1",
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "source_missing")
        self.assertEqual(failures[0]["stage"], "read_face_cache")

    def test_missing_openface_csv_without_executable_records_dependency_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_face_cache(cache_root, sample_id="sample-1")

            summary = extract_face_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "face_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="face_raw_openface_stats_v1",
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "dependency_missing")
        self.assertEqual(failures[0]["stage"], "run_openface")

    def test_missing_csv_can_use_injected_raw_video_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            csv_path = _write_face_cache(cache_root, sample_id="sample-1")
            generated = []

            def fake_generator(source_path, output_csv):
                generated.append((source_path, output_csv))
                _write_openface_csv(output_csv)

            summary = extract_face_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "face_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="face_raw_openface_stats_v1",
                csv_generator=fake_generator,
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(generated[0][1], csv_path)
        self.assertEqual(failures, [])

    def test_low_quality_window_is_masked_and_records_quality_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            csv_path = _write_face_cache(cache_root, sample_id="sample-1")
            _write_openface_csv(csv_path, successes=[0, 0, 1, 0])

            summary = extract_face_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "face_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="face_raw_openface_stats_v1",
                min_success_rate=0.5,
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))
            with np.load(root / "face_real_embeddings.npz", allow_pickle=True) as loaded:
                face_emb = loaded["face_emb"]
                modality_mask = loaded["modality_mask"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "quality_threshold_failed")
        self.assertEqual(face_emb.shape, (1, 256))
        self.assertEqual(modality_mask.tolist(), [[0, 0, 0, 0]])
        self.assertTrue(quality_flags[0]["masked"])

    def test_face_extraction_cli_writes_dependency_missing_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_face_cache(cache_root, sample_id="sample-1")
            window_index = root / "window_index.jsonl"
            window_index.write_text(json.dumps(_window("sample-1")) + "\n", encoding="utf-8")
            failures_out = root / "failures.json"
            summary_out = root / "summary.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/13_extract_face_embeddings.py",
                    "--window-index",
                    str(window_index),
                    "--cache-root",
                    str(cache_root),
                    "--encoder-profile",
                    "face_raw_openface_stats_v1",
                    "--out",
                    str(root / "face_real_embeddings.npz"),
                    "--failures-out",
                    str(failures_out),
                    "--summary-out",
                    str(summary_out),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            failures = json.loads(failures_out.read_text(encoding="utf-8"))
            summary = json.loads(summary_out.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(failures[0]["error_type"], "dependency_missing")
        self.assertEqual(summary["failure_types"], {"dependency_missing": 1})


def _window(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": "sub-02",
        "window_start_time": "2025-02-28 14:13:00",
        "window_end_time": "2025-02-28 14:13:10",
    }


def _write_face_cache(
    cache_root: Path,
    *,
    sample_id: str,
    encoder_profile: str = "face_raw_openface_stats_v1",
) -> Path:
    cache_dir = cache_root / "openface" / sample_id / encoder_profile
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = cache_dir / "source.mp4"
    mp4_path.write_bytes(b"fake-mp4")
    csv_path = cache_dir / "openface.csv"
    (cache_dir / "openface_target.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": sample_id.replace("sample", "event"),
                "subject_id": "sub-02",
                "modality": "face",
                "encoder_profile": encoder_profile,
                "cache_key": f"{sample_id}/face/{encoder_profile}",
                "source_path": str(mp4_path),
                "target_csv_path": str(csv_path),
                "openface_required": True,
            }
        ),
        encoding="utf-8",
    )
    return csv_path


def _write_openface_csv(csv_path: Path, *, successes=None) -> None:
    successes = successes or [1, 1, 1, 0]
    rows = ["frame,timestamp,confidence,success,pose_Rx,pose_Ry,pose_Rz,gaze_0_x,gaze_0_y,AU01_r,AU02_r"]
    for idx, success in enumerate(successes):
        rows.append(
            ",".join(
                [
                    str(idx + 1),
                    f"{idx * 0.1:.3f}",
                    "0.90" if success else "0.30",
                    str(success),
                    "0.10",
                    "0.20",
                    "0.05",
                    "0.01",
                    "0.02",
                    f"{0.4 + idx * 0.1:.3f}",
                    f"{0.2 + idx * 0.05:.3f}",
                ]
            )
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
