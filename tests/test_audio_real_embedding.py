import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.embeddings.audio_real import (
    FakeAudioBackend,
    extract_audio_real_embeddings,
)


class AudioRealEmbeddingTests(unittest.TestCase):
    def test_extract_audio_real_embeddings_writes_npz_and_empty_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _touch(root / "checkpoint")
            cache_root = root / "cache"
            wav_path = _write_audio_cache(cache_root, sample_id="sample-1")
            output_npz = root / "audio_real_embeddings.npz"
            failures_out = root / "failures.json"

            summary = extract_audio_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=output_npz,
                failures_out=failures_out,
                encoder_profile="wavlm_frozen_v1",
                checkpoint_path=checkpoint,
                backend=FakeAudioBackend(hidden_dim=4),
            )

            with np.load(output_npz, allow_pickle=True) as loaded:
                sample_ids = loaded["sample_id"].astype(str).tolist()
                audio_emb = loaded["audio_emb"]
                modality_mask = loaded["modality_mask"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]
            failures = json.loads(failures_out.read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(sample_ids, ["sample-1"])
        self.assertEqual(audio_emb.shape, (1, 256))
        self.assertEqual(audio_emb.dtype, np.float32)
        self.assertFalse(np.isnan(audio_emb).any())
        self.assertEqual(modality_mask.tolist(), [[0, 0, 0, 1]])
        self.assertEqual(quality_flags[0]["wav_path"], str(wav_path))
        self.assertEqual(failures, [])

    def test_missing_checkpoint_records_checkpoint_missing_without_backend_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_audio_cache(cache_root, sample_id="sample-1")
            failures_out = root / "failures.json"

            summary = extract_audio_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "audio_real_embeddings.npz",
                failures_out=failures_out,
                encoder_profile="wavlm_frozen_v1",
                checkpoint_path=root / "missing-checkpoint",
            )

            failures = json.loads(failures_out.read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "checkpoint_missing")
        self.assertEqual(failures[0]["modality"], "audio")
        self.assertEqual(failures[0]["stage"], "load_audio_encoder")

    def test_opensmile_profile_with_backend_returns_deterministic_256_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_audio_cache(cache_root, sample_id="sample-1", encoder_profile="audio_opensmile_egemaps_v1")

            summary = extract_audio_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "audio_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="audio_opensmile_egemaps_v1",
                backend=FakeAudioBackend(hidden_dim=6, frames=1),
            )
            with np.load(root / "audio_real_embeddings.npz", allow_pickle=True) as loaded:
                audio_emb = loaded["audio_emb"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(audio_emb.shape, (1, 256))
        self.assertEqual(quality_flags[0]["pooling"], "functionals")

    def test_emotion2vec_missing_checkpoint_records_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_audio_cache(cache_root, sample_id="sample-1", encoder_profile="audio_emotion2vec_plus_v1")

            summary = extract_audio_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "audio_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="audio_emotion2vec_plus_v1",
                checkpoint_path=root / "missing-checkpoint",
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "checkpoint_missing")

    def test_emotion2vec_backend_dependency_missing_is_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _touch(root / "emotion2vec-checkpoint")
            cache_root = root / "cache"
            _write_audio_cache(cache_root, sample_id="sample-1", encoder_profile="audio_emotion2vec_plus_v1")

            summary = extract_audio_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "audio_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="audio_emotion2vec_plus_v1",
                checkpoint_path=checkpoint,
                backend_factory=lambda checkpoint_path, device: (_ for _ in ()).throw(
                    RuntimeError("missing audio emotion dependency: modelscope")
                ),
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "dependency_missing")

    def test_mean_std_max_pooling_expands_features_before_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            checkpoint = _touch(root / "emotion2vec-checkpoint")
            _write_audio_cache(cache_root, sample_id="sample-1", encoder_profile="audio_emotion2vec_plus_v1")

            summary = extract_audio_real_embeddings(
                [_window("sample-1")],
                cache_root=cache_root,
                output_npz=root / "audio_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="audio_emotion2vec_plus_v1",
                checkpoint_path=checkpoint,
                backend=FakeAudioBackend(hidden_dim=4, frames=5),
            )
            with np.load(root / "audio_real_embeddings.npz", allow_pickle=True) as loaded:
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(quality_flags[0]["pooling"], "mean_std_max")
        self.assertEqual(quality_flags[0]["pooled_feature_dim"], 12)

    def test_missing_audio_cache_records_source_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _touch(root / "checkpoint")

            summary = extract_audio_real_embeddings(
                [_window("sample-1")],
                cache_root=root / "cache",
                output_npz=root / "audio_real_embeddings.npz",
                failures_out=root / "failures.json",
                encoder_profile="wavlm_frozen_v1",
                checkpoint_path=checkpoint,
                backend=FakeAudioBackend(hidden_dim=4),
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(failures[0]["error_type"], "source_missing")
        self.assertEqual(failures[0]["stage"], "read_audio_cache")

    def test_audio_extraction_cli_writes_checkpoint_missing_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "cache"
            _write_audio_cache(cache_root, sample_id="sample-1")
            window_index = root / "window_index.jsonl"
            window_index.write_text(json.dumps(_window("sample-1")) + "\n", encoding="utf-8")
            failures_out = root / "failures.json"
            summary_out = root / "summary.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/12_extract_audio_embeddings.py",
                    "--window-index",
                    str(window_index),
                    "--cache-root",
                    str(cache_root),
                    "--encoder-profile",
                    "wavlm_frozen_v1",
                    "--checkpoint",
                    str(root / "missing-checkpoint"),
                    "--out",
                    str(root / "audio_real_embeddings.npz"),
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


def _window(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": "sub-02",
    }


def _write_audio_cache(cache_root: Path, *, sample_id: str, encoder_profile: str = "wavlm_frozen_v1") -> Path:
    cache_dir = cache_root / "audio_clips" / sample_id / encoder_profile
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / "audio.wav"
    wav_path.write_bytes(b"fake-wav")
    (cache_dir / "audio.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": sample_id.replace("sample", "event"),
                "subject_id": "sub-02",
                "modality": "audio",
                "encoder_profile": encoder_profile,
                "cache_key": f"{sample_id}/audio/{encoder_profile}",
                "wav_path": str(wav_path),
                "clip_start_seconds": 0.0,
                "clip_end_seconds": 10.0,
                "target_sample_rate_hz": 16000,
                "target_channels": 1,
            }
        ),
        encoding="utf-8",
    )
    return wav_path


def _touch(path: Path) -> Path:
    path.write_text("checkpoint", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
