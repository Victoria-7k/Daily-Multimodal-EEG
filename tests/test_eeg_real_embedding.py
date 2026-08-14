import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.embeddings.eeg_real import (
    ArrayEEGReader,
    EEGWindowData,
    FakeDeepEEGBackend,
    extract_eeg_real_embeddings,
)


class EEGRealEmbeddingTests(unittest.TestCase):
    def test_extract_eeg_bandpower_writes_npz_and_empty_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            bdf_path = _write_eeg_cache(cache_root, sample_id="sample-1")
            reader = ArrayEEGReader(
                EEGWindowData(
                    data=np.ones((3, 2500), dtype=np.float32),
                    sfreq=250.0,
                    channel_names=["Fz", "Cz", "Pz"],
                    source_window_samples=5000,
                    original_sfreq=500.0,
                    start_offset_seconds=90.0,
                    end_offset_seconds=100.0,
                )
            )

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_bandpower_v1",
                reader=reader,
            )

            with np.load(root / "eeg_real_embeddings.npz", allow_pickle=True) as loaded:
                sample_ids = loaded["sample_id"].astype(str).tolist()
                eeg_emb = loaded["eeg_emb"]
                modality_mask = loaded["modality_mask"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(sample_ids, ["sample-1"])
        self.assertEqual(eeg_emb.shape, (1, 256))
        self.assertEqual(eeg_emb.dtype, np.float32)
        self.assertFalse(np.isnan(eeg_emb).any())
        self.assertEqual(modality_mask.tolist(), [[1, 0, 0, 0]])
        self.assertEqual(quality_flags[0]["source_path"], str(bdf_path))
        self.assertEqual(quality_flags[0]["channel_count"], 3)
        self.assertEqual(quality_flags[0]["sample_count"], 2500)
        self.assertEqual(quality_flags[0]["source_window_samples"], 5000)
        self.assertEqual(failures, [])

    def test_missing_eeg_cache_records_source_missing_without_importing_mne(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=root / "cache",
                output_npz=root / "eeg_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_bandpower_v1",
                reader=ArrayEEGReader(_valid_eeg_window()),
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "source_missing")
        self.assertEqual(failures[0]["modality"], "eeg")
        self.assertEqual(failures[0]["stage"], "read_eeg_cache")

    def test_wrong_eeg_window_shape_records_eeg_window_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_eeg_cache(cache_root, sample_id="sample-1")
            reader = ArrayEEGReader(
                EEGWindowData(
                    data=np.ones((2, 2400), dtype=np.float32),
                    sfreq=250.0,
                    channel_names=["Fz", "Cz"],
                    source_window_samples=4800,
                    original_sfreq=500.0,
                    start_offset_seconds=90.0,
                    end_offset_seconds=100.0,
                )
            )

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_bandpower_v1",
                reader=reader,
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))
            with np.load(root / "eeg_real_embeddings.npz", allow_pickle=True) as loaded:
                eeg_emb = loaded["eeg_emb"]

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "eeg_window_shape_mismatch")
        self.assertEqual(failures[0]["stage"], "encode_eeg")
        self.assertEqual(eeg_emb.shape, (0, 256))

    def test_partial_overlap_shape_failure_records_specific_coverage_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_eeg_cache(
                cache_root,
                sample_id="sample-1",
                window_start_offset_seconds=25.0,
                window_end_offset_seconds=35.0,
                eeg_recording_duration_seconds=30.0,
            )
            reader = ArrayEEGReader(
                EEGWindowData(
                    data=np.ones((2, 1250), dtype=np.float32),
                    sfreq=250.0,
                    channel_names=["Fz", "Cz"],
                    source_window_samples=2500,
                    original_sfreq=500.0,
                    start_offset_seconds=25.0,
                    end_offset_seconds=30.0,
                )
            )

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_bandpower_v1",
                reader=reader,
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(failures[0]["error_type"], "eeg_window_partial_overlap")

    def test_reader_out_of_range_value_error_records_after_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_eeg_cache(
                cache_root,
                sample_id="sample-1",
                window_start_offset_seconds=88700.0,
                window_end_offset_seconds=88710.0,
                eeg_recording_duration_seconds=13410.0,
            )

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_bandpower_v1",
                reader=_RaisingEEGReader("tmax (88710.0) must be less than or equal to the max time (13409.9980 s)"),
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(failures[0]["error_type"], "eeg_window_after_recording")

    def test_extract_eeg_deep_checkpoint_uses_deep_backend_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            checkpoint = _touch(root / "eegpt-checkpoint")
            _write_eeg_cache(cache_root, sample_id="sample-1", encoder_profile="eeg_deep_frozen_v1")
            reader = ArrayEEGReader(_valid_eeg_window())
            backend = FakeDeepEEGBackend(hidden_dim=12)

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_deep_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_deep_frozen_v1",
                checkpoint_path=checkpoint,
                reader=reader,
                deep_backend=backend,
            )

            with np.load(root / "eeg_deep_embeddings.npz", allow_pickle=True) as loaded:
                eeg_emb = loaded["eeg_emb"]
                encoder_versions = loaded["encoder_version"].astype(str).tolist()
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["encoder_profile"], "eeg_deep_frozen_v1")
        self.assertEqual(eeg_emb.shape, (1, 256))
        self.assertFalse(np.isnan(eeg_emb).any())
        self.assertEqual(encoder_versions, ["eeg_deep_frozen_v1"])
        self.assertEqual(quality_flags[0]["deep_backend"], "fake_deep_eeg")
        self.assertEqual(quality_flags[0]["deep_feature_dim"], 12)
        self.assertEqual(failures, [])

    def test_missing_deep_checkpoint_records_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_eeg_cache(cache_root, sample_id="sample-1", encoder_profile="eeg_deep_frozen_v1")

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_deep_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_deep_frozen_v1",
                checkpoint_path=root / "missing-checkpoint",
                reader=ArrayEEGReader(_valid_eeg_window()),
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "checkpoint_missing")
        self.assertEqual(failures[0]["stage"], "load_eeg_encoder")
        self.assertEqual(failures[0]["modality"], "eeg")

    def test_deep_backend_bad_feature_shape_records_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            checkpoint = _touch(root / "eegpt-checkpoint")
            _write_eeg_cache(cache_root, sample_id="sample-1", encoder_profile="eeg_deep_frozen_v1")

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_deep_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_deep_frozen_v1",
                checkpoint_path=checkpoint,
                reader=ArrayEEGReader(_valid_eeg_window()),
                deep_backend=FakeDeepEEGBackend(hidden_dim=0),
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "shape_mismatch")
        self.assertEqual(failures[0]["stage"], "encode_eeg")

    def test_deep_backend_cuda_oom_records_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            checkpoint = _touch(root / "eegpt-checkpoint")
            _write_eeg_cache(cache_root, sample_id="sample-1", encoder_profile="eeg_deep_frozen_v1")

            summary = extract_eeg_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "eeg_deep_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="eeg_deep_frozen_v1",
                checkpoint_path=checkpoint,
                reader=ArrayEEGReader(_valid_eeg_window()),
                deep_backend=_RaisingDeepEEGBackend("CUDA error: out of memory"),
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "oom")
        self.assertEqual(failures[0]["stage"], "encode_eeg")

    def test_eeg_extraction_cli_writes_source_missing_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "window_index.jsonl"
            window_index.write_text(json.dumps(_window("sample-1")) + "\n", encoding="utf-8")
            failures_out = root / "failures.json"
            summary_out = root / "summary.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/archive_legacy/14_extract_eeg_embeddings.py",
                    "--window-index",
                    str(window_index),
                    "--cache-root",
                    str(root / "missing-cache"),
                    "--encoder-profile",
                    "eeg_bandpower_v1",
                    "--out",
                    str(root / "eeg_real_embeddings.npz"),
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

    def test_eeg_deep_cli_writes_checkpoint_missing_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_eeg_cache(cache_root, sample_id="sample-1", encoder_profile="eeg_deep_frozen_v1")
            window_index = root / "window_index.jsonl"
            window_index.write_text(json.dumps(_window("sample-1")) + "\n", encoding="utf-8")
            failures_out = root / "failures.json"
            summary_out = root / "summary.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/archive_legacy/14_extract_eeg_embeddings.py",
                    "--window-index",
                    str(window_index),
                    "--cache-root",
                    str(cache_root),
                    "--encoder-profile",
                    "eeg_deep_frozen_v1",
                    "--checkpoint",
                    str(root / "missing-checkpoint"),
                    "--out",
                    str(root / "eeg_deep_embeddings.npz"),
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
        self.assertEqual(failures[0]["error_type"], "checkpoint_missing")
        self.assertEqual(summary["failure_types"], {"checkpoint_missing": 1})


def _valid_eeg_window() -> EEGWindowData:
    return EEGWindowData(
        data=np.ones((2, 2500), dtype=np.float32),
        sfreq=250.0,
        channel_names=["Fz", "Cz"],
        source_window_samples=5000,
        original_sfreq=500.0,
        start_offset_seconds=90.0,
        end_offset_seconds=100.0,
    )


def _window(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": "sub-02",
        "window_start_time": "2025-02-28 14:14:50",
        "window_end_time": "2025-02-28 14:15:00",
        "eeg_recording_start_time": "2025-02-28T14:13:20",
        "eeg_onset_seconds": 100.0,
        "window_start_offset_seconds": -10,
        "window_end_offset_seconds": 0,
    }


def _write_eeg_cache(
    cache_root: Path,
    *,
    sample_id: str,
    encoder_profile: str = "eeg_bandpower_v1",
    window_start_offset_seconds: float | None = None,
    window_end_offset_seconds: float | None = None,
    eeg_recording_duration_seconds: float | None = None,
) -> Path:
    cache_dir = cache_root / "eeg_windows" / sample_id / encoder_profile
    cache_dir.mkdir(parents=True, exist_ok=True)
    bdf_path = cache_dir / "sample.bdf"
    bdf_path.write_text("fake-bdf", encoding="utf-8")
    payload = {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": "sub-02",
        "modality": "eeg",
        "encoder_profile": encoder_profile,
        "cache_key": f"{sample_id}/eeg/{encoder_profile}",
        "source_path": str(bdf_path),
        "window_start_time": "2025-02-28 14:14:50",
        "window_end_time": "2025-02-28 14:15:00",
        "source_sampling_frequency_hz": 500.0,
        "target_resample_hz": 250,
        "target_window_samples": 2500,
    }
    if window_start_offset_seconds is not None:
        payload["window_start_offset_seconds"] = window_start_offset_seconds
    if window_end_offset_seconds is not None:
        payload["window_end_offset_seconds"] = window_end_offset_seconds
    if eeg_recording_duration_seconds is not None:
        payload["eeg_recording_duration_seconds"] = eeg_recording_duration_seconds
    (cache_dir / "window.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return bdf_path


def _touch(path: Path) -> Path:
    path.write_text("checkpoint", encoding="utf-8")
    return path


class _RaisingDeepEEGBackend:
    name = "raising_deep_eeg"

    def __init__(self, message: str) -> None:
        self._message = message

    def embed_features(self, data, *, channel_names):
        raise RuntimeError(self._message)


class _RaisingEEGReader:
    def __init__(self, message: str) -> None:
        self._message = message

    def read_window(self, source_path, *, start_offset_seconds, end_offset_seconds, target_sfreq):
        raise ValueError(self._message)


if __name__ == "__main__":
    unittest.main()
