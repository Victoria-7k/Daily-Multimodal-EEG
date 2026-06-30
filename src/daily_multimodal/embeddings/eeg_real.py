from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from daily_multimodal.alignment.eeg_coverage import classify_eeg_window_coverage, eeg_duration_seconds
from daily_multimodal.alignment.time_utils import parse_absolute_time
from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape
from daily_multimodal.embeddings.failures import EmbeddingFailure, write_failure_list


TARGET_SAMPLE_RATE_HZ = 250.0
TARGET_WINDOW_SAMPLES = 2500


@dataclass(frozen=True)
class EEGWindowData:
    data: np.ndarray
    sfreq: float
    channel_names: list[str]
    source_window_samples: int | None = None
    original_sfreq: float | None = None
    start_offset_seconds: float | None = None
    end_offset_seconds: float | None = None


class EEGReader(Protocol):
    def read_window(
        self,
        source_path: Path,
        *,
        start_offset_seconds: float,
        end_offset_seconds: float,
        target_sfreq: float,
    ) -> EEGWindowData:
        """Return a preprocessed EEG window shaped `[channels, samples]`."""


class DeepEEGBackend(Protocol):
    name: str

    def embed_features(self, data: np.ndarray, *, channel_names: list[str]) -> np.ndarray:
        """Return one frozen deep EEG feature vector, or token/frame features."""


@dataclass
class ArrayEEGReader:
    window: EEGWindowData

    def read_window(
        self,
        source_path: Path,
        *,
        start_offset_seconds: float,
        end_offset_seconds: float,
        target_sfreq: float,
    ) -> EEGWindowData:
        del source_path, start_offset_seconds, end_offset_seconds, target_sfreq
        return self.window


@dataclass
class FakeDeepEEGBackend:
    hidden_dim: int = 16
    name: str = "fake_deep_eeg"

    def embed_features(self, data: np.ndarray, *, channel_names: list[str]) -> np.ndarray:
        if self.hidden_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        digest = hashlib.sha256(
            data[: min(4, data.shape[0]), : min(32, data.shape[1])].tobytes()
            + "|".join(channel_names).encode("utf-8")
        ).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        return np.resize(values, self.hidden_dim).astype(np.float32)


class MNEBDFEEGReader:
    def __init__(
        self,
        *,
        notch_hz: float = 50.0,
        bandpass_low_hz: float = 1.0,
        bandpass_high_hz: float = 45.0,
    ) -> None:
        try:
            import mne
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            raise RuntimeError(f"missing EEG dependency: {exc.name}") from exc
        self._mne = mne
        self._notch_hz = notch_hz
        self._bandpass_low_hz = bandpass_low_hz
        self._bandpass_high_hz = bandpass_high_hz

    def read_window(
        self,
        source_path: Path,
        *,
        start_offset_seconds: float,
        end_offset_seconds: float,
        target_sfreq: float,
    ) -> EEGWindowData:  # pragma: no cover - requires real MNE/BDF runtime
        raw = self._mne.io.read_raw_bdf(str(source_path), preload=False, verbose="ERROR")
        original_sfreq = float(raw.info["sfreq"])
        tmin = max(0.0, float(start_offset_seconds))
        tmax = max(tmin, float(end_offset_seconds))
        raw.crop(tmin=tmin, tmax=tmax, include_tmax=False).load_data()
        try:
            raw.pick("eeg", exclude=[])
        except Exception:
            pass
        source_window_samples = int(raw.n_times)
        raw.notch_filter(freqs=[self._notch_hz], verbose="ERROR")
        raw.filter(
            l_freq=self._bandpass_low_hz,
            h_freq=self._bandpass_high_hz,
            verbose="ERROR",
        )
        raw.resample(float(target_sfreq), verbose="ERROR")
        data = raw.get_data().astype(np.float32, copy=False)
        return EEGWindowData(
            data=data,
            sfreq=float(raw.info["sfreq"]),
            channel_names=[str(name) for name in raw.ch_names],
            source_window_samples=source_window_samples,
            original_sfreq=original_sfreq,
            start_offset_seconds=tmin,
            end_offset_seconds=tmax,
        )


class BraindecodeEEGPTBackend:
    name = "braindecode_eegpt"

    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        try:
            import torch
            from braindecode.models import EEGPT
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            raise RuntimeError(f"missing EEG deep dependency: {exc.name}") from exc

        self._torch = torch
        self._EEGPT = EEGPT
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._config = _read_checkpoint_config(checkpoint_path)
        self._model = None
        self.load_report: dict[str, Any] = {}

    def embed_features(
        self,
        data: np.ndarray,
        *,
        channel_names: list[str],
    ) -> np.ndarray:  # pragma: no cover - requires real braindecode runtime
        del channel_names
        self._ensure_model(n_chans=int(data.shape[0]), n_times=int(data.shape[1]))
        tensor = self._torch.as_tensor(data[None, :, :], dtype=self._torch.float32, device=self._device)
        with self._torch.no_grad():
            try:
                output = self._model(tensor, return_features=True)
            except TypeError:
                output = self._model(tensor)
        if isinstance(output, dict):
            output = output.get("features", output.get("cls_token"))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if hasattr(output, "detach"):
            values = output.detach().cpu().numpy()
        else:
            values = np.asarray(output)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim >= 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim > 1:
            values = values.reshape(-1, values.shape[-1]).mean(axis=0)
        return values.astype(np.float32, copy=False).reshape(-1)

    def _ensure_model(self, *, n_chans: int, n_times: int) -> None:
        if self._model is not None:
            return
        kwargs = {
            "n_outputs": int(self._config.get("n_outputs") or 1),
            "n_chans": n_chans,
            "n_times": n_times,
            "sfreq": float(self._config.get("sfreq") or TARGET_SAMPLE_RATE_HZ),
        }
        self._model = self._EEGPT(**kwargs)
        self.load_report = _load_matching_torch_weights(
            self._model,
            self._checkpoint_path,
            torch_module=self._torch,
        )
        self._model.to(self._device)
        self._model.eval()


def extract_eeg_real_embeddings(
    windows: list[dict[str, Any]],
    *,
    cache_root: Path | str,
    output_npz: Path | str,
    failures_out: Path | str,
    encoder_profile: str,
    checkpoint_path: Path | str | None = None,
    reader: EEGReader | None = None,
    deep_backend: DeepEEGBackend | None = None,
    device: str = "cpu",
    projection_seed: int = 15015,
) -> dict[str, Any]:
    failures: list[EmbeddingFailure] = []
    samples: list[dict[str, Any]] = []
    cached: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for window in windows:
        cache = _read_eeg_cache(window, cache_root=cache_root, encoder_profile=encoder_profile)
        if cache is None:
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="read_eeg_cache",
                    error_type="source_missing",
                    error="EEG cache metadata or BDF file is missing",
                    source_path=str(_eeg_cache_dir(window, cache_root, encoder_profile)),
                )
            )
            continue
        cached.append((window, cache))

    deep_mode = encoder_profile == "eeg_deep_frozen_v1" or deep_backend is not None
    if deep_mode and cached and deep_backend is None:
        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint is None or not checkpoint.exists():
            for window, cache in cached:
                failures.append(
                    _failure(
                        window,
                        encoder_profile,
                        stage="load_eeg_encoder",
                        error_type="checkpoint_missing",
                        error="EEG deep checkpoint path is required and must exist",
                        source_path=str(checkpoint or "<missing-checkpoint-path>"),
                    )
                )
            _write_eeg_npz([], output_npz)
            write_failure_list(failures, failures_out)
            return _summary(samples, failures, encoder_profile)
        try:
            deep_backend = BraindecodeEEGPTBackend(checkpoint, device=device)
        except RuntimeError as exc:
            for window, cache in cached:
                failures.append(
                    _failure(
                        window,
                        encoder_profile,
                        stage="load_eeg_encoder",
                        error_type="dependency_missing",
                        error=str(exc),
                        source_path=str(cache.get("source_path") or checkpoint),
                    )
                )
            _write_eeg_npz([], output_npz)
            write_failure_list(failures, failures_out)
            return _summary(samples, failures, encoder_profile)

    if reader is None and cached:
        try:
            reader = MNEBDFEEGReader()
        except RuntimeError as exc:
            for window, cache in cached:
                failures.append(
                    _failure(
                        window,
                        encoder_profile,
                        stage="load_eeg_reader",
                        error_type="dependency_missing",
                        error=str(exc),
                        source_path=str(cache.get("source_path") or ""),
                    )
                )
            _write_eeg_npz([], output_npz)
            write_failure_list(failures, failures_out)
            return _summary(samples, failures, encoder_profile)

    for window, cache in cached:
        source_path = Path(str(cache["source_path"]))
        start_offset: float | None = None
        end_offset: float | None = None
        try:
            start_offset, end_offset = _window_offsets(window, cache)
            eeg_window = reader.read_window(  # type: ignore[union-attr]
                source_path,
                start_offset_seconds=start_offset,
                end_offset_seconds=end_offset,
                target_sfreq=TARGET_SAMPLE_RATE_HZ,
            )
            data = _validate_window_data(eeg_window)
            if deep_mode:
                features = _validate_deep_features(
                    deep_backend.embed_features(data, channel_names=eeg_window.channel_names)  # type: ignore[union-attr]
                )
            else:
                features = _bandpower_statistics(data, sfreq=float(eeg_window.sfreq))
            embedding = _project_to_256(features, seed=projection_seed, salt=encoder_profile)
            embedding = validate_embedding_shape("eeg_emb", embedding)
        except ValueError as exc:
            error_type = _value_error_type(exc, window=window, cache=cache, start_offset=start_offset, end_offset=end_offset)
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_eeg",
                    error_type=error_type,
                    error=str(exc),
                    source_path=str(source_path),
                )
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_eeg" if deep_mode else "read_eeg_window",
                    error_type=_runtime_error_type(exc),
                    error=str(exc),
                    source_path=str(source_path),
                )
            )
            continue
        samples.append(
            {
                "sample_id": window.get("sample_id", cache.get("sample_id", "")),
                "event_id": window.get("event_id", cache.get("event_id", "")),
                "subject_id": window.get("subject_id", cache.get("subject_id", "")),
                "eeg_emb": embedding,
                "modality_mask": np.array([1, 0, 0, 0], dtype=np.int8),
                "quality_flags": {
                    "source_path": str(source_path),
                    "channel_count": int(data.shape[0]),
                    "sample_count": int(data.shape[1]),
                    "source_window_samples": eeg_window.source_window_samples,
                    "source_sampling_frequency_hz": eeg_window.original_sfreq
                    or cache.get("source_sampling_frequency_hz"),
                    "target_sample_rate_hz": float(eeg_window.sfreq),
                    "start_offset_seconds": eeg_window.start_offset_seconds or start_offset,
                    "end_offset_seconds": eeg_window.end_offset_seconds or end_offset,
                    "channel_names": eeg_window.channel_names,
                    "deep_backend": getattr(deep_backend, "name", None) if deep_mode else None,
                    "deep_feature_dim": int(np.asarray(features).reshape(-1).shape[0])
                    if deep_mode
                    else None,
                    "deep_load_report": getattr(deep_backend, "load_report", None)
                    if deep_mode
                    else None,
                },
                "encoder_version": encoder_profile,
            }
        )

    _write_eeg_npz(samples, output_npz)
    write_failure_list(failures, failures_out)
    return _summary(samples, failures, encoder_profile)


def write_eeg_quality_summary(summary: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _read_eeg_cache(
    window: dict[str, Any],
    *,
    cache_root: Path | str,
    encoder_profile: str,
) -> dict[str, Any] | None:
    cache_dir = _eeg_cache_dir(window, cache_root, encoder_profile)
    metadata_path = cache_dir / "window.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_path = Path(str(metadata.get("source_path") or ""))
    if not source_path.is_file():
        return None
    metadata["source_path"] = str(source_path)
    return metadata


def _read_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    config_path = checkpoint_path / "config.json" if checkpoint_path.is_dir() else checkpoint_path.with_name("config.json")
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_matching_torch_weights(
    model: Any,
    checkpoint_path: Path,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    state_path = _select_checkpoint_state_file(checkpoint_path)
    model_state = model.state_dict()
    if state_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            raise RuntimeError(f"missing EEG deep dependency: {exc.name}") from exc
        checkpoint_state = load_file(str(state_path), device="cpu")
    else:
        checkpoint_state = torch_module.load(str(state_path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
            checkpoint_state = checkpoint_state["state_dict"]
    matched = {}
    skipped: list[str] = []
    for key, value in checkpoint_state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key not in model_state:
            skipped.append(normalized_key)
            continue
        if tuple(model_state[normalized_key].shape) != tuple(value.shape):
            skipped.append(normalized_key)
            continue
        matched[normalized_key] = value
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {
        "state_path": str(state_path),
        "loaded_key_count": len(matched),
        "skipped_key_count": len(skipped),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "skipped_keys_preview": sorted(skipped)[:10],
    }


def _select_checkpoint_state_file(checkpoint_path: Path) -> Path:
    candidates = [checkpoint_path]
    if checkpoint_path.is_dir():
        candidates = [
            checkpoint_path / "model.safetensors",
            checkpoint_path / "pytorch_model.bin",
            checkpoint_path / "model.pt",
            checkpoint_path / "checkpoint.pt",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"EEG deep checkpoint state file was not found under {checkpoint_path}")


def _eeg_cache_dir(window: dict[str, Any], cache_root: Path | str, encoder_profile: str) -> Path:
    return Path(cache_root) / "eeg_windows" / str(window.get("sample_id", "")) / encoder_profile


def _window_offsets(window: dict[str, Any], cache: dict[str, Any]) -> tuple[float, float]:
    if cache.get("window_start_offset_seconds") is not None and cache.get("window_end_offset_seconds") is not None:
        return float(cache["window_start_offset_seconds"]), float(cache["window_end_offset_seconds"])

    recording_start_text = window.get("eeg_recording_start_time") or cache.get("eeg_recording_start_time")
    start_text = cache.get("window_start_time") or window.get("window_start_time")
    end_text = cache.get("window_end_time") or window.get("window_end_time")
    if recording_start_text and start_text and end_text:
        recording_start = parse_absolute_time(str(recording_start_text))
        window_start = parse_absolute_time(str(start_text))
        window_end = parse_absolute_time(str(end_text))
        return (
            float((window_start - recording_start).total_seconds()),
            float((window_end - recording_start).total_seconds()),
        )

    if window.get("eeg_onset_seconds") is not None:
        onset = float(window["eeg_onset_seconds"])
        start_offset = float(window.get("window_start_offset_seconds", -10))
        end_offset = float(window.get("window_end_offset_seconds", 0))
        return onset + start_offset, onset + end_offset

    raise ValueError("cannot determine EEG window offsets")


def _validate_window_data(eeg_window: EEGWindowData) -> np.ndarray:
    data = np.asarray(eeg_window.data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected EEG window shape [channels, samples], got {data.shape}")
    if data.shape[0] <= 0:
        raise ValueError("EEG window has no channels")
    if data.shape[1] != TARGET_WINDOW_SAMPLES:
        raise ValueError(
            f"expected EEG window samples {TARGET_WINDOW_SAMPLES}, got {data.shape[1]}"
        )
    if not np.isfinite(data).all():
        raise ValueError("EEG window contains NaN or infinite values")
    return data


def _validate_deep_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("deep EEG feature vector is empty")
    if not np.isfinite(values).all():
        raise ValueError("deep EEG feature vector contains NaN or infinite values")
    return values


def _bandpower_statistics(data: np.ndarray, *, sfreq: float) -> np.ndarray:
    features: list[float] = []
    features.extend(_aggregate_channel_stats(data))
    for low_hz, high_hz in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)):
        powers = _band_power_per_channel(data, sfreq=sfreq, low_hz=low_hz, high_hz=high_hz)
        features.append(float(np.log1p(np.mean(powers))))
        features.append(float(np.log1p(np.std(powers))))
    features.extend([float(data.shape[0]), float(data.shape[1]), float(sfreq)])
    return np.asarray(features, dtype=np.float32)


def _aggregate_channel_stats(data: np.ndarray) -> list[float]:
    channel_means = data.mean(axis=1)
    channel_stds = data.std(axis=1)
    channel_rms = np.sqrt(np.mean(np.square(data), axis=1))
    return [
        float(channel_means.mean()),
        float(channel_means.std()),
        float(channel_stds.mean()),
        float(channel_stds.std()),
        float(channel_rms.mean()),
        float(channel_rms.std()),
        float(data.min()),
        float(data.max()),
    ]


def _band_power_per_channel(
    data: np.ndarray,
    *,
    sfreq: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / float(sfreq))
    spectrum = np.abs(np.fft.rfft(data, axis=1)) ** 2
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not mask.any():
        return np.zeros(data.shape[0], dtype=np.float32)
    return spectrum[:, mask].mean(axis=1).astype(np.float32)


def _project_to_256(vector: np.ndarray, *, seed: int, salt: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("EEG feature vector is empty")
    if not np.isfinite(values).all():
        raise ValueError("EEG feature vector contains NaN or infinite values")
    normalized = values.copy()
    std = float(normalized.std())
    if std > 0:
        normalized = (normalized - float(normalized.mean())) / std
    rng_seed = seed + int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    scale = 1.0 / max(1.0, float(np.sqrt(normalized.size)))
    weights = rng.normal(0.0, scale, size=(normalized.size, EMBEDDING_DIM)).astype(np.float32)
    return np.tanh(normalized @ weights).astype(np.float32)


def _write_eeg_npz(samples: list[dict[str, Any]], output_npz: Path | str) -> Path:
    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.array([sample["sample_id"] for sample in samples], dtype=object),
        event_id=np.array([sample["event_id"] for sample in samples], dtype=object),
        subject_id=np.array([sample["subject_id"] for sample in samples], dtype=object),
        eeg_emb=np.stack([sample["eeg_emb"] for sample in samples]).astype(np.float32)
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
    channel_counts = [
        int(sample["quality_flags"]["channel_count"])
        for sample in samples
        if sample["quality_flags"].get("channel_count") is not None
    ]
    return {
        "stage": 15,
        "modality": "eeg",
        "encoder_profile": encoder_profile,
        "success_count": len(samples),
        "failure_count": len(failures),
        "failure_types": _count_by_error_type(failures),
        "mean_channel_count": None if not channel_counts else float(np.mean(channel_counts)),
        "target_window_samples": TARGET_WINDOW_SAMPLES,
        "nan_count": int(sum(np.isnan(sample["eeg_emb"]).sum() for sample in samples)),
    }


def _count_by_error_type(failures: list[EmbeddingFailure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.error_type] = counts.get(failure.error_type, 0) + 1
    return counts


def _runtime_error_type(exc: Exception) -> str:
    message = str(exc).lower()
    if "out of memory" in message or "oom" in message:
        return "oom"
    if "timed out" in message or "timeout" in message:
        return "timeout"
    return "decode_failed"


def _value_error_type(
    exc: ValueError,
    *,
    window: dict[str, Any],
    cache: dict[str, Any],
    start_offset: float | None,
    end_offset: float | None,
) -> str:
    message = str(exc)
    if "expected EEG window samples" not in message:
        return "shape_mismatch"
    if start_offset is None or end_offset is None:
        return "eeg_window_shape_mismatch"
    duration = eeg_duration_seconds({**window, **cache})
    if duration is None:
        return "eeg_window_shape_mismatch"
    coverage = classify_eeg_window_coverage(
        start_offset_seconds=start_offset,
        end_offset_seconds=end_offset,
        bdf_duration_seconds=duration,
    )
    classification = coverage["classification"]
    if classification == "negative_offset":
        return "eeg_window_before_recording"
    if classification == "after_recording_end":
        return "eeg_window_after_recording"
    if classification == "partial_overlap":
        return "eeg_window_partial_overlap"
    return "eeg_window_shape_mismatch"


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
        modality="eeg",
        encoder_profile=encoder_profile,
        stage=stage,
        error_type=error_type,
        error=error,
        source_path=source_path,
        recoverable=True,
    )
