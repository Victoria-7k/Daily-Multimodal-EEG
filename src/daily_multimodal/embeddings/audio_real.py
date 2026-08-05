from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape
from daily_multimodal.embeddings.failures import EmbeddingFailure, write_failure_list


class AudioBackend(Protocol):
    def embed_frames(self, wav_path: Path) -> np.ndarray:
        """Return frame-level frozen model embeddings shaped `[frames, hidden_dim]`."""


AudioBackendFactory = Callable[[Path | None, str], AudioBackend]


@dataclass
class FakeAudioBackend:
    hidden_dim: int = 8
    frames: int = 5

    def embed_frames(self, wav_path: Path) -> np.ndarray:
        digest = hashlib.sha256(wav_path.read_bytes() + str(wav_path).encode("utf-8")).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        tiled = np.resize(values, self.frames * self.hidden_dim)
        return tiled.reshape(self.frames, self.hidden_dim).astype(np.float32)


class TransformersAudioBackend:
    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        try:
            import torch
            import torchaudio
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            raise RuntimeError(f"missing audio model dependency: {exc.name}") from exc

        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(checkpoint_path),
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(str(checkpoint_path), local_files_only=True)
        self._model.to(device)
        self._model.eval()

    def embed_frames(self, wav_path: Path) -> np.ndarray:  # pragma: no cover - requires model deps
        waveform, sample_rate = self._torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = self._torchaudio.functional.resample(waveform, sample_rate, 16000)
        audio = waveform.squeeze(0).detach().cpu().numpy()
        inputs = self._feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)


class OpenSmileEgemapsBackend:
    def __init__(self) -> None:
        try:
            import opensmile
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            raise RuntimeError(f"missing audio openSMILE dependency: {exc.name}") from exc
        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def embed_frames(self, wav_path: Path) -> np.ndarray:  # pragma: no cover - requires opensmile
        values = self._smile.process_file(str(wav_path)).to_numpy().reshape(-1)
        return values.astype(np.float32).reshape(1, -1)


class Emotion2VecBackend:
    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            raise RuntimeError(f"missing audio emotion dependency: {exc.name}") from exc
        kwargs: dict[str, Any] = {"model": str(checkpoint_path)}
        if device:
            kwargs["device"] = device
        try:
            self._pipeline = pipeline(Tasks.emotion_recognition, **kwargs)
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            missing = exc.name or str(exc).strip() or "modelscope[audio]"
            raise RuntimeError(f"missing audio emotion dependency: {missing}") from exc

    def embed_frames(self, wav_path: Path) -> np.ndarray:  # pragma: no cover - requires emotion2vec runtime
        result = self._pipeline(str(wav_path), granularity="utterance", extract_embedding=True)
        embedding = None
        if isinstance(result, dict):
            embedding = _first_present(result, ("embedding", "feats", "hidden_states"))
        elif isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                embedding = _first_present(first, ("embedding", "feats", "hidden_states"))
        if embedding is None:
            raise RuntimeError("emotion2vec did not return an embedding")
        values = np.asarray(embedding, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values.astype(np.float32, copy=False)


def extract_audio_real_embeddings(
    windows: list[dict[str, Any]],
    *,
    cache_root: Path | str,
    output_npz: Path | str,
    failures_out: Path | str,
    encoder_profile: str,
    cache_profile: str | None = None,
    checkpoint_path: Path | str | None = None,
    backend: AudioBackend | None = None,
    backend_factory: AudioBackendFactory | None = None,
    device: str = "cpu",
    projection_seed: int = 13013,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    resolved_cache_profile = cache_profile or encoder_profile
    failures: list[EmbeddingFailure] = []
    samples: list[dict[str, Any]] = []

    if backend is None:
        profile = _profile_settings(encoder_profile)
        if profile["checkpoint_required"] and (checkpoint is None or not checkpoint.exists()):
            for window in windows:
                failures.append(
                    _failure(
                        window,
                        encoder_profile,
                        stage="load_audio_encoder",
                        error_type="checkpoint_missing",
                        error="audio checkpoint path is required and must exist",
                        source_path=str(checkpoint or "<missing-checkpoint-path>"),
                    )
                )
            _write_audio_npz([], output_npz)
            write_failure_list(failures, failures_out)
            return _summary(samples, failures, encoder_profile)
        try:
            backend = (
                backend_factory(checkpoint, device)
                if backend_factory is not None
                else _build_backend_for_profile(encoder_profile, checkpoint, device=device)
            )
        except RuntimeError as exc:
            for window in windows:
                failures.append(
                    _failure(
                        window,
                        encoder_profile,
                        stage="load_audio_encoder",
                        error_type="dependency_missing",
                        error=str(exc),
                        source_path=str(checkpoint or "<no-checkpoint-required>"),
                    )
                )
            _write_audio_npz([], output_npz)
            write_failure_list(failures, failures_out)
            return _summary(samples, failures, encoder_profile)

    for window in windows:
        cache = _read_audio_cache(
            window,
            cache_root=cache_root,
            cache_profile=resolved_cache_profile,
        )
        if cache is None:
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="read_audio_cache",
                    error_type="source_missing",
                    error="audio cache metadata or wav file is missing",
                    source_path=str(_audio_cache_dir(window, cache_root, resolved_cache_profile)),
                )
            )
            continue
        wav_path = Path(cache["wav_path"])
        try:
            frames = np.asarray(backend.embed_frames(wav_path), dtype=np.float32)
            if frames.ndim != 2 or frames.shape[0] == 0:
                raise ValueError(f"expected frame embedding shape [frames, hidden_dim], got {frames.shape}")
            pooling = _profile_settings(encoder_profile)["pooling"]
            pooled = _pool_frames(frames, pooling=pooling)
            embedding = _project_to_256(pooled, seed=projection_seed, salt=encoder_profile)
            embedding = validate_embedding_shape("audio_emb", embedding)
        except ValueError as exc:
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_audio",
                    error_type="shape_mismatch",
                    error=str(exc),
                    source_path=str(wav_path),
                )
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_audio",
                    error_type="extraction_failed",
                    error=str(exc),
                    source_path=str(wav_path),
                )
            )
            continue
        samples.append(
            {
                "sample_id": window.get("sample_id", cache.get("sample_id", "")),
                "event_id": window.get("event_id", cache.get("event_id", "")),
                "subject_id": window.get("subject_id", cache.get("subject_id", "")),
                "audio_emb": embedding,
                "modality_mask": np.array([0, 0, 0, 1], dtype=np.int8),
                "quality_flags": {
                    "wav_path": str(wav_path),
                    "clip_start_seconds": cache.get("clip_start_seconds"),
                    "clip_end_seconds": cache.get("clip_end_seconds"),
                    "duration_seconds": _duration_seconds(cache),
                    "frame_count": int(frames.shape[0]),
                    "hidden_dim": int(frames.shape[1]),
                    "pooling": pooling,
                    "pooled_feature_dim": int(np.asarray(pooled).reshape(-1).shape[0]),
                    "target_sample_rate_hz": cache.get("target_sample_rate_hz", 16000),
                    "cache_profile": resolved_cache_profile,
                },
                "encoder_version": encoder_profile,
            }
        )

    _write_audio_npz(samples, output_npz)
    write_failure_list(failures, failures_out)
    return _summary(samples, failures, encoder_profile)


def _profile_settings(encoder_profile: str) -> dict[str, Any]:
    if encoder_profile == "audio_opensmile_egemaps_v1":
        return {
            "backend": "opensmile_egemaps",
            "checkpoint_required": False,
            "pooling": "functionals",
        }
    if encoder_profile == "audio_emotion2vec_plus_v1":
        return {
            "backend": "emotion2vec",
            "checkpoint_required": True,
            "pooling": "mean_std_max",
        }
    return {
        "backend": "transformers",
        "checkpoint_required": True,
        "pooling": "mean",
    }


def _build_backend_for_profile(
    encoder_profile: str,
    checkpoint: Path | None,
    *,
    device: str,
) -> AudioBackend:
    settings = _profile_settings(encoder_profile)
    if settings["backend"] == "opensmile_egemaps":
        return OpenSmileEgemapsBackend()
    if settings["backend"] == "emotion2vec":
        if checkpoint is None:
            raise RuntimeError("emotion2vec checkpoint path is required")
        return Emotion2VecBackend(checkpoint, device=device)
    if checkpoint is None:
        raise RuntimeError("audio checkpoint path is required")
    return TransformersAudioBackend(checkpoint, device=device)


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _pool_frames(frames: np.ndarray, *, pooling: str) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if pooling == "functionals":
        return values.reshape(-1)
    if pooling == "mean_std_max":
        return np.concatenate(
            [
                values.mean(axis=0),
                values.std(axis=0),
                values.max(axis=0),
            ]
        ).astype(np.float32)
    return values.mean(axis=0)


def write_audio_quality_summary(summary: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _read_audio_cache(
    window: dict[str, Any],
    *,
    cache_root: Path | str,
    cache_profile: str,
) -> dict[str, Any] | None:
    cache_dir = _audio_cache_dir(window, cache_root, cache_profile)
    metadata_path = cache_dir / "audio.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    wav_path = Path(str(metadata.get("wav_path") or cache_dir / "audio.wav"))
    if not wav_path.is_file():
        return None
    metadata["wav_path"] = str(wav_path)
    return metadata


def _audio_cache_dir(window: dict[str, Any], cache_root: Path | str, cache_profile: str) -> Path:
    return Path(cache_root) / "audio_clips" / str(window.get("sample_id", "")) / cache_profile


def _project_to_256(vector: np.ndarray, *, seed: int, salt: str) -> np.ndarray:
    pooled = np.asarray(vector, dtype=np.float32).reshape(-1)
    if pooled.size == 0:
        raise ValueError("pooled audio embedding is empty")
    if not np.isfinite(pooled).all():
        raise ValueError("pooled audio embedding contains NaN or infinite values")
    rng_seed = seed + int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    scale = 1.0 / max(1.0, float(np.sqrt(pooled.size)))
    weights = rng.normal(0.0, scale, size=(pooled.size, EMBEDDING_DIM)).astype(np.float32)
    projected = pooled @ weights
    return np.tanh(projected).astype(np.float32)


def _write_audio_npz(samples: list[dict[str, Any]], output_npz: Path | str) -> Path:
    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.array([sample["sample_id"] for sample in samples], dtype=object),
        event_id=np.array([sample["event_id"] for sample in samples], dtype=object),
        subject_id=np.array([sample["subject_id"] for sample in samples], dtype=object),
        audio_emb=np.stack([sample["audio_emb"] for sample in samples]).astype(np.float32)
        if samples
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        modality_mask=np.stack([sample["modality_mask"] for sample in samples]).astype(np.int8)
        if samples
        else np.zeros((0, 4), dtype=np.int8),
        quality_flags=np.array(
            [json.dumps(sample["quality_flags"], ensure_ascii=False) for sample in samples],
            dtype=object,
        ),
        encoder_version=np.array([sample["encoder_version"] for sample in samples], dtype=object),
    )
    return out


def _summary(
    samples: list[dict[str, Any]],
    failures: list[EmbeddingFailure],
    encoder_profile: str,
) -> dict[str, Any]:
    durations = [
        float(sample["quality_flags"]["duration_seconds"])
        for sample in samples
        if sample["quality_flags"].get("duration_seconds") is not None
    ]
    return {
        "stage": 13,
        "modality": "audio",
        "encoder_profile": encoder_profile,
        "success_count": len(samples),
        "failure_count": len(failures),
        "failure_types": _count_by_error_type(failures),
        "mean_duration_seconds": None if not durations else float(np.mean(durations)),
        "nan_count": int(sum(np.isnan(sample["audio_emb"]).sum() for sample in samples)),
    }


def _count_by_error_type(failures: list[EmbeddingFailure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.error_type] = counts.get(failure.error_type, 0) + 1
    return counts


def _failure(
    window: dict[str, Any],
    encoder_profile: str,
    *,
    stage: str,
    error_type: str,
    error: str,
    source_path: str,
) -> EmbeddingFailure:
    return EmbeddingFailure(
        sample_id=str(window.get("sample_id") or "<missing-sample-id>"),
        event_id=str(window.get("event_id") or "<missing-event-id>"),
        subject_id=str(window.get("subject_id") or "<missing-subject-id>"),
        modality="audio",
        encoder_profile=encoder_profile,
        stage=stage,
        error_type=error_type,
        error=error,
        source_path=source_path,
        recoverable=True,
    )


def _duration_seconds(cache: dict[str, Any]) -> float | None:
    start = cache.get("clip_start_seconds")
    end = cache.get("clip_end_seconds")
    try:
        return float(end) - float(start)
    except (TypeError, ValueError):
        return None
