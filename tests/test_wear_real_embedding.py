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
        self.assertEqual(quality_flags[0]["ppg_rows_in_window"], 20)
        self.assertEqual(quality_flags[0]["ppg_duplicate_timestamp_rows"], 1)
        self.assertEqual(quality_flags[0]["ppg_effective_sampling_rate_hz"], 2.0)

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

    def test_wear_physio_features_v2_reports_hrv_gsr_and_static_acc_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            cache_dir = _write_wear_cache(
                cache_root,
                sample_id="sample-1",
                encoder_profile="wear_physio_features_v2",
            )
            _write_physio_v2_csvs(cache_dir, flat_ppg=False, moving_acc=False)

            summary = extract_wear_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_physio_features_v2",
            )
            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                wear_emb = loaded["wear_emb"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

        flags = quality_flags[0]
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(wear_emb.shape, (1, 256))
        self.assertGreater(flags["heart_rate"], 50.0)
        self.assertLess(flags["rmssd"], 0.05)
        self.assertGreater(flags["peak_count"], 5)
        self.assertGreater(flags["gsr_slope"], 0.0)
        self.assertGreater(flags["stationary_ratio"], 0.90)
        self.assertIn("heart_rate", flags["physio_feature_names"])

    def test_wear_physio_features_v2_flags_flat_ppg_and_moving_acc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            static_cache = _write_wear_cache(
                cache_root,
                sample_id="sample-static",
                encoder_profile="wear_physio_features_v2",
            )
            moving_cache = _write_wear_cache(
                cache_root,
                sample_id="sample-moving",
                encoder_profile="wear_physio_features_v2",
            )
            _write_physio_v2_csvs(static_cache, flat_ppg=True, moving_acc=False)
            _write_physio_v2_csvs(moving_cache, flat_ppg=True, moving_acc=True)

            summary = extract_wear_real_embeddings(
                [_window("sample-static"), _window("sample-moving")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_physio_features_v2",
            )
            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

        static_flags, moving_flags = quality_flags
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(static_flags["peak_count"], 0)
        self.assertTrue(static_flags["ppg_peak_insufficient"])
        self.assertGreater(moving_flags["motion_intensity"], static_flags["motion_intensity"])
        self.assertLess(moving_flags["stationary_ratio"], static_flags["stationary_ratio"])

    def test_wear_quality_audit_summary_reports_requested_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            plausible_cache = _write_wear_cache(
                cache_root,
                sample_id="sample-plausible",
                encoder_profile="wear_physio_features_v2",
            )
            flat_cache = _write_wear_cache(
                cache_root,
                sample_id="sample-flat",
                encoder_profile="wear_physio_features_v2",
            )
            _write_physio_v2_csvs(plausible_cache, flat_ppg=False, moving_acc=False)
            _write_physio_v2_csvs(flat_cache, flat_ppg=True, moving_acc=True)

            summary = extract_wear_real_embeddings(
                [_window("sample-plausible"), _window("sample-flat")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_physio_features_v2",
            )
            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

        audit = summary["quality_audit"]
        self.assertEqual(audit["window_count"], 2)
        self.assertEqual(audit["failure_count"], 0)
        self.assertEqual(audit["modalities"]["ppg"]["rows_in_window"]["min"], 640.0)
        self.assertEqual(audit["modalities"]["gsr"]["rows_in_window"]["mean"], 320.0)
        self.assertEqual(audit["modalities"]["acc"]["invalid_rows"]["sum"], 0.0)
        self.assertEqual(audit["modalities"]["ppg"]["source_rows"]["min"], 640.0)
        self.assertEqual(audit["modalities"]["ppg"]["duplicate_timestamps_count"], 0)
        self.assertEqual(audit["modalities"]["ppg"]["nonmonotonic_timestamps_count"], 0)
        self.assertEqual(audit["modalities"]["ppg"]["flatline_window_count"], 1)
        self.assertEqual(audit["ppg"]["heart_rate_plausible_range_bpm"], [40.0, 180.0])
        self.assertEqual(audit["ppg"]["heart_rate_plausible_count"], 1)
        self.assertEqual(audit["ppg"]["heart_rate_implausible_count"], 1)
        self.assertEqual(audit["ppg"]["peak_count"]["min"], 0.0)
        self.assertGreater(audit["ppg"]["peak_count"]["max"], 5.0)
        self.assertIn("slope_abnormal_count", audit["gsr"])
        self.assertIn("scr_count_abnormal_count", audit["gsr"])
        self.assertIn("motion_intensity", audit["acc"])
        self.assertIn("stationary_ratio", audit["acc"])
        self.assertFalse(quality_flags[0]["ppg_flatline"])
        self.assertTrue(quality_flags[0]["heart_rate_plausible"])
        self.assertTrue(quality_flags[1]["ppg_flatline"])
        self.assertFalse(quality_flags[1]["heart_rate_plausible"])
        self.assertEqual(quality_flags[0]["wear_quality_grade"], "A")
        self.assertEqual(quality_flags[0]["wear_quality_label"], "high")
        self.assertIn("wear_physio_features_v2", quality_flags[0]["wear_quality_recommended_use"])
        self.assertEqual(quality_flags[1]["wear_quality_grade"], "C")
        self.assertEqual(quality_flags[1]["wear_quality_label"], "low")
        self.assertTrue(quality_flags[1]["motion_artifact_risk"])
        self.assertEqual(audit["wear_quality_grade_counts"], {"A": 1, "B": 0, "C": 1})

    def test_wear_quality_grade_b_keeps_motion_risky_but_usable_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            cache_dir = _write_wear_cache(
                cache_root,
                sample_id="sample-moving",
                encoder_profile="wear_physio_features_v2",
            )
            _write_physio_v2_csvs(cache_dir, flat_ppg=False, moving_acc=True, gsr_step=0.0)

            summary = extract_wear_real_embeddings(
                [_window("sample-moving")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_physio_features_v2",
            )
            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                flags = json.loads(loaded["quality_flags"].tolist()[0])
                mask = loaded["modality_mask"].tolist()

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(flags["wear_quality_grade"], "B")
        self.assertEqual(flags["wear_quality_label"], "medium")
        self.assertTrue(flags["motion_artifact_risk"])
        self.assertEqual(mask, [[0, 1, 0, 0]])

    def test_mask_low_quality_wear_sets_mask_zero_for_grade_c(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            cache_dir = _write_wear_cache(
                cache_root,
                sample_id="sample-flat",
                encoder_profile="wear_physio_features_v2",
            )
            _write_physio_v2_csvs(cache_dir, flat_ppg=True, moving_acc=True)

            summary = extract_wear_real_embeddings(
                [_window("sample-flat")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_physio_features_v2",
                mask_low_quality_wear=True,
            )
            with np.load(root / "wear_real_embeddings.npz", allow_pickle=True) as loaded:
                flags = json.loads(loaded["quality_flags"].tolist()[0])
                mask = loaded["modality_mask"].tolist()

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["masked_count"], 1)
        self.assertEqual(flags["wear_quality_grade"], "C")
        self.assertTrue(flags["wear_low_quality_masked"])
        self.assertEqual(mask, [[0, 0, 0, 0]])

    def test_wear_physio_v2_can_reuse_sequence_cache_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            cache_dir = _write_wear_cache(
                cache_root,
                sample_id="sample-1",
                encoder_profile="wear_sequence_v1",
            )
            _write_physio_v2_csvs(cache_dir, flat_ppg=False, moving_acc=False)

            summary = extract_wear_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_physio_features_v2",
            )

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)

    def test_wear_physio_v2_can_read_window_index_sources_without_cache_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sources"
            source_dir.mkdir()
            _write_physio_v2_csvs(source_dir, flat_ppg=False, moving_acc=False)
            window = {
                **_window("sample-1"),
                "wear_ppg_path": str(source_dir / "ppg.csv"),
                "wear_gsr_path": str(source_dir / "gsr.csv"),
                "wear_acc_path": str(source_dir / "acc.csv"),
            }

            summary = extract_wear_real_embeddings(
                [window],
                cache_root=root / "missing-cache",
                output_npz=root / "wear_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wear_physio_features_v2",
            )

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["quality_audit"]["modalities"]["ppg"]["rows_in_window"]["mean"], 640.0)


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


def _write_physio_v2_csvs(cache_dir: Path, *, flat_ppg: bool, moving_acc: bool, gsr_step: float = 0.001) -> None:
    start = datetime.fromisoformat("2025-02-28 14:13:00")
    ppg_rows = []
    gsr_rows = []
    acc_rows = []
    for index in range(640):
        t = start + timedelta(seconds=index / 64.0)
        phase = index % 64
        pulse = 1.0 if phase == 8 else 0.0
        ppg_value = 0.5 if flat_ppg else 0.2 + pulse
        ppg_rows.append([ppg_value, t])
    for index in range(320):
        t = start + timedelta(seconds=index / 32.0)
        gsr_rows.append([0.2 + index * gsr_step, t])
        if moving_acc:
            acc_rows.append([
                np.sin(index / 5.0),
                np.cos(index / 7.0),
                np.sin(index / 3.0),
                t,
            ])
        else:
            acc_rows.append([0.0, 0.0, 1.0, t])
    _write_csv(cache_dir / "ppg.csv", ["PPG", "csv_time_PPG"], ppg_rows)
    _write_csv(cache_dir / "gsr.csv", ["GSR", "csv_time_GSR"], gsr_rows)
    _write_csv(
        cache_dir / "acc.csv",
        ["Motion_dataX", "Motion_dataY", "Motion_dataZ", "csv_time_motion"],
        acc_rows,
    )


if __name__ == "__main__":
    unittest.main()
