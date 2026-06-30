import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from daily_multimodal.embeddings import wear_real
from daily_multimodal.embeddings.wear_real import extract_wear_real_embeddings


class WearRealEmbeddingTests(unittest.TestCase):
    def test_extract_wear_sequence_writes_npz_sequences_and_empty_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            cache_dir = _write_wear_cache(cache_root, sample_id="sample-1")
            _write_wear_csvs(cache_dir)

            summary = extract_wear_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_sequence_v1",
            )

            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                sample_ids = loaded["sample_id"].astype(str).tolist()
                wear_emb = loaded["wear_emb"]
                modality_mask = loaded["modality_mask"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]
            with np.load(cache_dir / "sequence.npz") as sequences:
                ppg_sequence = sequences["ppg"]
                gsr_sequence = sequences["gsr"]
                acc_sequence = sequences["acc"]
            stats = json.loads((cache_dir / "stats.json").read_text(encoding="utf-8"))
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(sample_ids, ["sample-1"])
        self.assertEqual(wear_emb.shape, (1, 256))
        self.assertEqual(wear_emb.dtype, np.float32)
        self.assertFalse(np.isnan(wear_emb).any())
        self.assertEqual(modality_mask.tolist(), [[0, 1, 0, 0]])
        self.assertEqual(ppg_sequence.shape, (640, 1))
        self.assertEqual(gsr_sequence.shape, (320, 1))
        self.assertEqual(acc_sequence.shape, (320, 3))
        self.assertGreater(quality_flags[0]["motion_intensity"], 0.0)
        self.assertGreaterEqual(quality_flags[0]["stationary_ratio"], 0.0)
        self.assertIn("ppg_effective_sampling_rate_hz", quality_flags[0])
        self.assertEqual(stats["target_sample_rates_hz"], {"ppg": 64, "gsr": 32, "acc": 32})
        self.assertEqual(failures, [])

    def test_missing_wear_cache_records_source_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            summary = extract_wear_real_embeddings(
                [_window("sample-1")],
                cache_root=root / "cache",
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_sequence_v1",
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "source_missing")
        self.assertEqual(failures[0]["stage"], "read_wear_cache")

    def test_duplicate_and_nonmonotonic_timestamps_are_quality_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            cache_dir = _write_wear_cache(cache_root, sample_id="sample-1")
            _write_wear_csvs(cache_dir, duplicate=True, nonmonotonic=True)

            summary = extract_wear_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_sequence_v1",
            )
            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

        self.assertEqual(summary["success_count"], 1)
        self.assertTrue(quality_flags[0]["ppg_duplicate_timestamps"])
        self.assertTrue(quality_flags[0]["ppg_nonmonotonic_timestamps"])

    def test_empty_wear_window_is_masked_and_records_quality_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            cache_dir = _write_wear_cache(cache_root, sample_id="sample-1")
            _write_wear_csvs(cache_dir, outside_window=True)

            summary = extract_wear_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_sequence_v1",
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))
            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                wear_emb = loaded["wear_emb"]
                modality_mask = loaded["modality_mask"]

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "quality_threshold_failed")
        self.assertEqual(wear_emb.shape, (1, 256))
        self.assertEqual(modality_mask.tolist(), [[0, 0, 0, 0]])

    def test_wear_extraction_cli_writes_source_missing_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "window_index.jsonl"
            window_index.write_text(json.dumps(_window("sample-1")) + "\n", encoding="utf-8")
            failures_out = root / "failures.json"
            summary_out = root / "summary.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/15_extract_wear_embeddings.py",
                    "--window-index",
                    str(window_index),
                    "--cache-root",
                    str(root / "missing-cache"),
                    "--encoder-profile",
                    "wear_sequence_v1",
                    "--out",
                    str(root / "wear_real_embeddings.npz"),
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
        self.assertEqual(failures[0]["error_type"], "source_missing")
        self.assertEqual(summary["failure_types"], {"source_missing": 1})

    def test_wear_csv_sources_are_reused_across_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            first_cache = _write_wear_cache(cache_root, sample_id="sample-1")
            _write_wear_csvs(first_cache)
            second_cache = _write_wear_cache(cache_root, sample_id="sample-2")
            metadata = json.loads((second_cache / "window.json").read_text(encoding="utf-8"))
            metadata["source_paths"] = {
                "ppg": str(first_cache / "ppg.csv"),
                "gsr": str(first_cache / "gsr.csv"),
                "acc": str(first_cache / "acc.csv"),
            }
            (second_cache / "window.json").write_text(json.dumps(metadata), encoding="utf-8")
            original_reader = wear_real.csv.DictReader
            reader_calls = 0

            def counting_reader(*args, **kwargs):
                nonlocal reader_calls
                reader_calls += 1
                return original_reader(*args, **kwargs)

            with patch.object(wear_real.csv, "DictReader", side_effect=counting_reader):
                summary = extract_wear_real_embeddings(
                    [_window("sample-1"), _window("sample-2")],
                    cache_root=cache_root,
                    output_npz=root / "wear_real_embeddings.npz",
                    failures_out=root / "failures.json",
                    encoder_profile="wear_sequence_v1",
                )

        self.assertEqual(summary["success_count"], 2)
        self.assertLessEqual(reader_calls, 3)


def _window(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": "sub-02",
        "window_start_time": "2025-02-28 14:13:00",
        "window_end_time": "2025-02-28 14:13:10",
    }


def _write_wear_cache(
    cache_root: Path,
    *,
    sample_id: str,
    encoder_profile: str = "wear_sequence_v1",
) -> Path:
    cache_dir = cache_root / "wear_windows" / sample_id / encoder_profile
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ppg": cache_dir / "ppg.csv",
        "gsr": cache_dir / "gsr.csv",
        "acc": cache_dir / "acc.csv",
    }
    (cache_dir / "window.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": sample_id.replace("sample", "event"),
                "subject_id": "sub-02",
                "modality": "wear",
                "encoder_profile": encoder_profile,
                "cache_key": f"{sample_id}/wear/{encoder_profile}",
                "source_paths": {name: str(path) for name, path in paths.items()},
                "window_start_time": "2025-02-28 14:13:00",
                "window_end_time": "2025-02-28 14:13:10",
                "target_sample_rates_hz": {"ppg": 64, "gsr": 32, "acc": 32},
            }
        ),
        encoding="utf-8",
    )
    return cache_dir


def _write_wear_csvs(
    cache_dir: Path,
    *,
    duplicate: bool = False,
    nonmonotonic: bool = False,
    outside_window: bool = False,
) -> None:
    start = datetime.fromisoformat("2025-02-28 14:13:00")
    if outside_window:
        start = start - timedelta(minutes=5)
    ppg_times = [start + timedelta(seconds=index * 0.5) for index in range(20)]
    if duplicate:
        ppg_times[5] = ppg_times[4]
    if nonmonotonic:
        ppg_times[10], ppg_times[11] = ppg_times[11], ppg_times[10]

    _write_csv(cache_dir / "ppg.csv", ["PPG", "csv_time_PPG"], [[0.5 + i * 0.01, t] for i, t in enumerate(ppg_times)])
    gsr_times = [start + timedelta(seconds=index * 0.5) for index in range(20)]
    _write_csv(cache_dir / "gsr.csv", ["GSR", "csv_time_GSR"], [[0.2 + i * 0.005, t] for i, t in enumerate(gsr_times)])
    acc_times = [start + timedelta(seconds=index * 0.5) for index in range(20)]
    _write_csv(
        cache_dir / "acc.csv",
        ["Motion_dataX", "Motion_dataY", "Motion_dataZ", "csv_time_motion"],
        [[0.1 * i, 0.2 * i, 0.3 * i, t] for i, t in enumerate(acc_times)],
    )


def _write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, datetime):
                values.append(value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
