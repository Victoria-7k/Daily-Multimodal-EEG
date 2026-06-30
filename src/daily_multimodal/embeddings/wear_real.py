from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.alignment.time_utils import parse_absolute_time
from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape
from daily_multimodal.embeddings.failures import EmbeddingFailure, write_failure_list


TARGET_SAMPLE_RATES_HZ = {"ppg": 64, "gsr": 32, "acc": 32}
TARGET_COLUMNS = {
    "ppg": ("csv_time_PPG", ("PPG",)),
    "gsr": ("csv_time_GSR", ("GSR",)),
    "acc": ("csv_time_motion", ("Motion_dataX", "Motion_dataY", "Motion_dataZ")),
}


@dataclass
class WearSeries:
    timestamps_seconds: np.ndarray
    values: np.ndarray
    duplicate_timestamps: bool
    nonmonotonic_timestamps: bool
    invalid_rows: int
    source_rows: int


def extract_wear_real_embeddings(
    windows: list[dict[str, Any]],
    *,
    cache_root: Path | str,
    output_npz: Path | str,
    failures_out: Path | str,
    encoder_profile: str,
    projection_seed: int = 16016,
) -> dict[str, Any]:
    failures: list[EmbeddingFailure] = []
    samples: list[dict[str, Any]] = []

    for window in windows:
        cache = _read_wear_cache(window, cache_root=cache_root, encoder_profile=encoder_profile)
        if cache is None:
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="read_wear_cache",
                    error_type="source_missing",
                    error="wear cache metadata or CSV source is missing",
                    source_path=str(_wear_cache_dir(window, cache_root, encoder_profile)),
                )
            )
            continue

        cache_dir = _wear_cache_dir(window, cache_root, encoder_profile)
        try:
            window_start = parse_absolute_time(str(cache.get("window_start_time") or window["window_start_time"]))
            window_end = parse_absolute_time(str(cache.get("window_end_time") or window["window_end_time"]))
            duration = (window_end - window_start).total_seconds()
            if duration <= 0:
                raise ValueError("wear window duration must be positive")

            sequences: dict[str, np.ndarray] = {}
            qualities: dict[str, Any] = {}
            stats_features: list[float] = []
            for modality in ("ppg", "gsr", "acc"):
                series = _read_wear_series(
                    Path(cache["source_paths"][modality]),
                    modality=modality,
                    window_start=window_start,
                    window_end=window_end,
                )
                target_rate = int(cache.get("target_sample_rates_hz", {}).get(modality) or TARGET_SAMPLE_RATES_HZ[modality])
                sequence = _resample_series(series, duration_seconds=duration, target_rate_hz=target_rate)
                sequences[modality] = sequence
                _add_quality_flags(qualities, modality, series, sequence, duration_seconds=duration)
                stats_features.extend(_sequence_stats(sequence))

            usable = any(qualities[f"{modality}_rows_in_window"] > 0 for modality in ("ppg", "gsr", "acc"))
            if not usable:
                embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                failures.append(
                    _failure(
                        window,
                        encoder_profile,
                        stage="quality_gate",
                        error_type="quality_threshold_failed",
                        error="no wearable rows found inside window",
                        source_path=str(cache_dir),
                    )
                )
            else:
                sequence_features = _sequence_encoder_features(sequences)
                embedding = _project_to_256(
                    np.concatenate([np.asarray(stats_features, dtype=np.float32), sequence_features]),
                    seed=projection_seed,
                    salt=encoder_profile,
                )
                embedding = validate_embedding_shape("wear_emb", embedding)

            quality_flags = {
                **qualities,
                "sequence_cache_path": str(cache_dir / "sequence.npz"),
                "stats_cache_path": str(cache_dir / "stats.json"),
                "target_sample_rates_hz": {
                    modality: int(cache.get("target_sample_rates_hz", {}).get(modality) or TARGET_SAMPLE_RATES_HZ[modality])
                    for modality in ("ppg", "gsr", "acc")
                },
                "motion_intensity": _motion_intensity(sequences["acc"]),
                "stationary_ratio": _stationary_ratio(sequences["acc"]),
                "masked": not usable,
            }
            _write_sequence_cache(cache_dir, sequences, quality_flags, stats_features)
        except ValueError as exc:
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_wear",
                    error_type="shape_mismatch",
                    error=str(exc),
                    source_path=str(cache_dir),
                )
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_wear",
                    error_type="decode_failed",
                    error=str(exc),
                    source_path=str(cache_dir),
                )
            )
            continue

        samples.append(
            {
                "sample_id": window.get("sample_id", cache.get("sample_id", "")),
                "event_id": window.get("event_id", cache.get("event_id", "")),
                "subject_id": window.get("subject_id", cache.get("subject_id", "")),
                "wear_emb": embedding,
                "modality_mask": np.array([0, 1 if usable else 0, 0, 0], dtype=np.int8),
                "quality_flags": quality_flags,
                "encoder_version": encoder_profile,
            }
        )

    _write_wear_npz(samples, output_npz)
    write_failure_list(failures, failures_out)
    return _summary(samples, failures, encoder_profile)


def write_wear_quality_summary(summary: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _read_wear_cache(
    window: dict[str, Any],
    *,
    cache_root: Path | str,
    encoder_profile: str,
) -> dict[str, Any] | None:
    cache_dir = _wear_cache_dir(window, cache_root, encoder_profile)
    metadata_path = cache_dir / "window.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_paths = metadata.get("source_paths") or {}
    for modality in ("ppg", "gsr", "acc"):
        path = Path(str(source_paths.get(modality) or ""))
        if not path.is_file():
            return None
        source_paths[modality] = str(path)
    metadata["source_paths"] = source_paths
    return metadata


def _wear_cache_dir(window: dict[str, Any], cache_root: Path | str, encoder_profile: str) -> Path:
    return Path(cache_root) / "wear_windows" / str(window.get("sample_id", "")) / encoder_profile


def _read_wear_series(
    path: Path,
    *,
    modality: str,
    window_start,
    window_end,
) -> WearSeries:
    table = _load_wear_source_table(str(path), modality)
    start_ts = _naive_epoch_seconds(window_start)
    end_ts = _naive_epoch_seconds(window_end)
    in_window = (table["timestamps"] >= start_ts) & (table["timestamps"] < end_ts)
    selected_timestamps = table["timestamps"][in_window]
    selected_values = table["values"][in_window]
    seconds = (selected_timestamps - start_ts).astype(np.float32)
    duplicate = bool(np.unique(seconds).shape[0] != seconds.shape[0])

    if selected_values.size:
        order = np.argsort(seconds.astype(np.float64), kind="stable")
        sorted_seconds = seconds[order]
        sorted_values = selected_values[order]
        unique_seconds, unique_indices = np.unique(sorted_seconds, return_index=True)
        sorted_values = sorted_values[unique_indices]
    else:
        unique_seconds = np.zeros((0,), dtype=np.float32)
        sorted_values = np.zeros((0, len(TARGET_COLUMNS[modality][1])), dtype=np.float32)
    return WearSeries(
        timestamps_seconds=unique_seconds.astype(np.float32),
        values=sorted_values.astype(np.float32),
        duplicate_timestamps=duplicate,
        nonmonotonic_timestamps=bool(table["nonmonotonic"]),
        invalid_rows=int(table["invalid_rows"]),
        source_rows=int(table["source_rows"]),
    )


@lru_cache(maxsize=32)
def _load_wear_source_table(path: str, modality: str) -> dict[str, Any]:
    try:
        return _load_wear_source_table_pandas(path, modality)
    except ImportError:  # pragma: no cover - pandas is available in supported envs
        return _load_wear_source_table_csv(path, modality)


def _load_wear_source_table_pandas(path: str, modality: str) -> dict[str, Any]:
    import pandas as pd

    time_column, value_columns = TARGET_COLUMNS[modality]
    columns = [time_column, *value_columns]
    frame = pd.read_csv(path, usecols=lambda column: column in set(columns))
    source_rows = int(len(frame))
    if time_column not in frame.columns or any(column not in frame.columns for column in value_columns):
        return {
            "timestamps": np.zeros((0,), dtype=np.float64),
            "values": np.zeros((0, len(value_columns)), dtype=np.float32),
            "invalid_rows": source_rows,
            "source_rows": source_rows,
            "nonmonotonic": False,
        }

    try:
        parsed_time = pd.to_datetime(frame[time_column], errors="coerce", format="mixed")
    except TypeError:  # pragma: no cover - older pandas fallback
        parsed_time = pd.to_datetime(frame[time_column], errors="coerce")
    numeric_values = frame.loc[:, list(value_columns)].apply(pd.to_numeric, errors="coerce")
    valid_mask = parsed_time.notna() & numeric_values.notna().all(axis=1)
    invalid_rows = int((~valid_mask).sum())
    valid_times = parsed_time[valid_mask]
    values = numeric_values[valid_mask].to_numpy(dtype=np.float32, copy=True)
    if valid_times.empty:
        timestamps = np.zeros((0,), dtype=np.float64)
        nonmonotonic = False
    else:
        epoch = pd.Timestamp("1970-01-01")
        timestamps = (valid_times - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
        nonmonotonic = bool(np.any(np.diff(timestamps) < 0.0))
    return {
        "timestamps": timestamps,
        "values": values if values.size else np.zeros((0, len(value_columns)), dtype=np.float32),
        "invalid_rows": invalid_rows,
        "source_rows": source_rows,
        "nonmonotonic": nonmonotonic,
    }


def _load_wear_source_table_csv(path: str, modality: str) -> dict[str, Any]:
    time_column, value_columns = TARGET_COLUMNS[modality]
    timestamps: list[float] = []
    values: list[list[float]] = []
    invalid_rows = 0
    source_rows = 0
    previous_absolute = None
    nonmonotonic = False

    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_rows += 1
            raw_time = row.get(time_column)
            if not raw_time:
                invalid_rows += 1
                continue
            try:
                timestamp = parse_absolute_time(raw_time)
            except ValueError:
                invalid_rows += 1
                continue
            if previous_absolute is not None and timestamp < previous_absolute:
                nonmonotonic = True
            previous_absolute = timestamp
            row_values: list[float] = []
            ok = True
            for column in value_columns:
                try:
                    value = float(row.get(column, ""))
                except ValueError:
                    ok = False
                    value = math.nan
                row_values.append(value)
            if not ok or not all(math.isfinite(value) for value in row_values):
                invalid_rows += 1
                continue
            timestamps.append(_naive_epoch_seconds(timestamp))
            values.append(row_values)

    value_dim = len(value_columns)
    return {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "values": np.asarray(values, dtype=np.float32)
        if values
        else np.zeros((0, value_dim), dtype=np.float32),
        "invalid_rows": int(invalid_rows),
        "source_rows": int(source_rows),
        "nonmonotonic": bool(nonmonotonic),
    }


def _naive_epoch_seconds(value: datetime) -> float:
    return (value - datetime(1970, 1, 1)).total_seconds()


def _resample_series(
    series: WearSeries,
    *,
    duration_seconds: float,
    target_rate_hz: int,
) -> np.ndarray:
    sample_count = int(round(duration_seconds * target_rate_hz))
    if sample_count <= 0:
        raise ValueError("target sequence sample count must be positive")
    channels = series.values.shape[1] if series.values.ndim == 2 else 1
    target_seconds = np.arange(sample_count, dtype=np.float32) / float(target_rate_hz)
    if series.values.shape[0] == 0:
        return np.zeros((sample_count, channels), dtype=np.float32)
    if series.values.shape[0] == 1:
        return np.repeat(series.values[:1], sample_count, axis=0).astype(np.float32)
    out = np.empty((sample_count, channels), dtype=np.float32)
    for column in range(channels):
        out[:, column] = np.interp(
            target_seconds,
            series.timestamps_seconds,
            series.values[:, column],
            left=series.values[0, column],
            right=series.values[-1, column],
        )
    return out.astype(np.float32)


def _add_quality_flags(
    quality: dict[str, Any],
    modality: str,
    series: WearSeries,
    sequence: np.ndarray,
    *,
    duration_seconds: float,
) -> None:
    rows = int(series.values.shape[0])
    quality[f"{modality}_rows_in_window"] = rows
    quality[f"{modality}_source_rows"] = int(series.source_rows)
    quality[f"{modality}_invalid_rows"] = int(series.invalid_rows)
    quality[f"{modality}_duplicate_timestamps"] = bool(series.duplicate_timestamps)
    quality[f"{modality}_nonmonotonic_timestamps"] = bool(series.nonmonotonic_timestamps)
    quality[f"{modality}_effective_sampling_rate_hz"] = float(rows / duration_seconds) if duration_seconds > 0 else None
    quality[f"{modality}_sequence_shape"] = list(sequence.shape)
    quality[f"{modality}_missing_ratio"] = 1.0 if rows == 0 else 0.0


def _sequence_stats(sequence: np.ndarray) -> list[float]:
    features: list[float] = []
    if sequence.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    for column in range(sequence.shape[1]):
        values = sequence[:, column]
        features.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(values.min()),
                float(values.max()),
                float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
            ]
        )
    return features


def _sequence_encoder_features(sequences: dict[str, np.ndarray]) -> np.ndarray:
    features: list[float] = []
    for modality in ("ppg", "gsr", "acc"):
        seq = sequences[modality]
        features.extend(_sequence_stats(seq))
        if seq.shape[0] >= 4:
            splits = np.array_split(seq, 4, axis=0)
            for split in splits:
                features.extend(split.mean(axis=0).astype(np.float32).tolist())
                features.extend(split.std(axis=0).astype(np.float32).tolist())
    return np.asarray(features, dtype=np.float32)


def _motion_intensity(acc_sequence: np.ndarray) -> float:
    if acc_sequence.size == 0:
        return 0.0
    magnitude = np.linalg.norm(acc_sequence, axis=1)
    return float(np.mean(magnitude))


def _stationary_ratio(acc_sequence: np.ndarray, *, threshold: float = 0.05) -> float:
    if acc_sequence.shape[0] < 2:
        return 1.0
    delta = np.linalg.norm(np.diff(acc_sequence, axis=0), axis=1)
    return float(np.mean(delta < threshold))


def _write_sequence_cache(
    cache_dir: Path,
    sequences: dict[str, np.ndarray],
    quality_flags: dict[str, Any],
    stats_features: list[float],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / "sequence.npz",
        ppg=sequences["ppg"].astype(np.float32),
        gsr=sequences["gsr"].astype(np.float32),
        acc=sequences["acc"].astype(np.float32),
    )
    payload = {
        "target_sample_rates_hz": quality_flags["target_sample_rates_hz"],
        "quality_flags": quality_flags,
        "statistics_features": [float(value) for value in stats_features],
    }
    (cache_dir / "stats.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _project_to_256(vector: np.ndarray, *, seed: int, salt: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("wear feature vector is empty")
    if not np.isfinite(values).all():
        raise ValueError("wear feature vector contains NaN or infinite values")
    normalized = values.copy()
    std = float(normalized.std())
    if std > 0:
        normalized = (normalized - float(normalized.mean())) / std
    rng_seed = seed + int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    scale = 1.0 / max(1.0, float(np.sqrt(normalized.size)))
    weights = rng.normal(0.0, scale, size=(normalized.size, EMBEDDING_DIM)).astype(np.float32)
    return np.tanh(normalized @ weights).astype(np.float32)


def _write_wear_npz(samples: list[dict[str, Any]], output_npz: Path | str) -> Path:
    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.array([sample["sample_id"] for sample in samples], dtype=object),
        event_id=np.array([sample["event_id"] for sample in samples], dtype=object),
        subject_id=np.array([sample["subject_id"] for sample in samples], dtype=object),
        wear_emb=np.stack([sample["wear_emb"] for sample in samples]).astype(np.float32)
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
    usable = [sample for sample in samples if int(sample["modality_mask"][1]) == 1]
    qualities = [sample["quality_flags"] for sample in samples]
    return {
        "stage": 16,
        "modality": "wear",
        "encoder_profile": encoder_profile,
        "embedded_count": len(samples),
        "success_count": len(usable),
        "failure_count": len(failures),
        "failure_types": _count_by_error_type(failures),
        "masked_count": len(samples) - len(usable),
        "mean_motion_intensity": _mean_quality(qualities, "motion_intensity"),
        "mean_stationary_ratio": _mean_quality(qualities, "stationary_ratio"),
        "mean_ppg_effective_sampling_rate_hz": _mean_quality(qualities, "ppg_effective_sampling_rate_hz"),
        "mean_gsr_effective_sampling_rate_hz": _mean_quality(qualities, "gsr_effective_sampling_rate_hz"),
        "mean_acc_effective_sampling_rate_hz": _mean_quality(qualities, "acc_effective_sampling_rate_hz"),
        "nan_count": int(sum(np.isnan(sample["wear_emb"]).sum() for sample in samples)),
    }


def _mean_quality(qualities: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in qualities if item.get(key) is not None]
    return None if not values else float(np.mean(values))


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
        modality="wear",
        encoder_profile=encoder_profile,
        stage=stage,
        error_type=error_type,
        error=error,
        source_path=source_path,
        recoverable=True,
    )
