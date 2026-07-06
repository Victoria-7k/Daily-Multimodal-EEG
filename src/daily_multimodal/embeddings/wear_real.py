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
PHYSIO_FEATURE_ENCODER_PROFILES = {
    "wear_physio_features_v2",
    "wear_deep_sequence_v1",
    "wear_physio_features_preprocessed_v1",
    "wear_deep_sequence_preprocessed_v1",
}
PREPROCESSED_ENCODER_PROFILES = {
    "wear_physio_features_preprocessed_v1",
    "wear_deep_sequence_preprocessed_v1",
}
PLAUSIBLE_HEART_RATE_RANGE_BPM = (40.0, 180.0)
GSR_SLOPE_ABNORMAL_ABS_THRESHOLD = 0.003
GSR_SCR_ABNORMAL_HIGH_THRESHOLD = 52.0
ACC_MOTION_HIGH_THRESHOLD = 0.25
ACC_STABLE_STATIONARY_RATIO_THRESHOLD = 0.5
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
    duplicate_timestamp_rows: int
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
    mask_low_quality_wear: bool = False,
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
                if encoder_profile in PREPROCESSED_ENCODER_PROFILES:
                    sequence, preprocessing_flags = _preprocess_wear_sequence(
                        modality,
                        sequence,
                        target_rate_hz=target_rate,
                    )
                    qualities.update(preprocessing_flags)
                sequences[modality] = sequence
                _add_quality_flags(qualities, modality, series, sequence, duration_seconds=duration)
                stats_features.extend(_sequence_stats(sequence))

            usable = any(qualities[f"{modality}_rows_in_window"] > 0 for modality in ("ppg", "gsr", "acc"))
            feature_names: list[str] | None = None
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
                if encoder_profile in PHYSIO_FEATURE_ENCODER_PROFILES:
                    feature_names, feature_values, physio_flags = _physio_features_v2(
                        sequences,
                        qualities,
                    )
                    qualities.update(physio_flags)
                    if encoder_profile in {"wear_physio_features_v2", "wear_physio_features_preprocessed_v1"}:
                        stats_features = feature_values.astype(np.float32).tolist()
                        sequence_features = feature_values
                    else:
                        sequence_features = _deep_sequence_encoder_features(sequences)
                else:
                    feature_names = None
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
                "motion_intensity": qualities.get("motion_intensity", _motion_intensity(sequences["acc"])),
                "stationary_ratio": qualities.get("stationary_ratio", _stationary_ratio(sequences["acc"])),
                "masked": not usable,
            }
            if encoder_profile in PREPROCESSED_ENCODER_PROFILES:
                quality_flags["wear_preprocessing_applied"] = True
                quality_flags["wear_preprocessing_version"] = "wear_signal_preprocessing_v1"
            if encoder_profile in PHYSIO_FEATURE_ENCODER_PROFILES:
                quality_flags["physio_feature_names"] = feature_names or []
                quality_flags["physio_feature_values"] = [float(value) for value in stats_features]
            if usable:
                _add_wear_quality_grade(quality_flags)
                if mask_low_quality_wear and quality_flags["wear_quality_grade"] == "C":
                    usable = False
                    quality_flags["wear_low_quality_masked"] = True
                    quality_flags["masked"] = True
                else:
                    quality_flags["wear_low_quality_masked"] = False
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
    fallback_profile: str | None = None
    if not metadata_path.is_file():
        if encoder_profile == "wear_sequence_v1":
            return _read_wear_cache_from_window(window, encoder_profile=encoder_profile)
        fallback_profile = "wear_sequence_v1"
        metadata_path = _wear_cache_dir(window, cache_root, fallback_profile) / "window.json"
        if not metadata_path.is_file():
            return _read_wear_cache_from_window(window, encoder_profile=encoder_profile)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if fallback_profile is not None:
        metadata["metadata_source_encoder_profile"] = fallback_profile
        metadata["requested_encoder_profile"] = encoder_profile
    source_paths = metadata.get("source_paths") or {}
    for modality in ("ppg", "gsr", "acc"):
        path = Path(str(source_paths.get(modality) or ""))
        if not path.is_file():
            return None
        source_paths[modality] = str(path)
    metadata["source_paths"] = source_paths
    return metadata


def _read_wear_cache_from_window(
    window: dict[str, Any],
    *,
    encoder_profile: str,
) -> dict[str, Any] | None:
    source_paths = {
        "ppg": Path(str(window.get("wear_ppg_path") or "")),
        "gsr": Path(str(window.get("wear_gsr_path") or "")),
        "acc": Path(str(window.get("wear_acc_path") or "")),
    }
    if any(not path.is_file() for path in source_paths.values()):
        return None
    return {
        "sample_id": window.get("sample_id", ""),
        "event_id": window.get("event_id", ""),
        "subject_id": window.get("subject_id", ""),
        "modality": "wear",
        "encoder_profile": encoder_profile,
        "metadata_source": "window_index",
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "window_start_time": window.get("window_start_time", ""),
        "window_end_time": window.get("window_end_time", ""),
        "target_sample_rates_hz": TARGET_SAMPLE_RATES_HZ,
    }


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
    if not table["nonmonotonic"]:
        left = int(np.searchsorted(table["timestamps"], start_ts, side="left"))
        right = int(np.searchsorted(table["timestamps"], end_ts, side="left"))
        selected_timestamps = table["timestamps"][left:right]
        selected_values = table["values"][left:right]
    else:
        in_window = (table["timestamps"] >= start_ts) & (table["timestamps"] < end_ts)
        selected_timestamps = table["timestamps"][in_window]
        selected_values = table["values"][in_window]
    seconds = (selected_timestamps - start_ts).astype(np.float32)
    unique_count = int(np.unique(seconds).shape[0])
    duplicate_rows = int(seconds.shape[0] - unique_count)
    duplicate = duplicate_rows > 0

    if selected_values.size:
        order = np.argsort(seconds.astype(np.float64), kind="stable")
        sorted_seconds = seconds[order]
        sorted_values = selected_values[order]
        output_seconds = _spread_duplicate_seconds(
            sorted_seconds,
            duration_seconds=float(end_ts - start_ts),
        )
    else:
        output_seconds = np.zeros((0,), dtype=np.float32)
        sorted_values = np.zeros((0, len(TARGET_COLUMNS[modality][1])), dtype=np.float32)
    return WearSeries(
        timestamps_seconds=output_seconds.astype(np.float32),
        values=sorted_values.astype(np.float32),
        duplicate_timestamps=duplicate,
        duplicate_timestamp_rows=duplicate_rows,
        nonmonotonic_timestamps=bool(table["nonmonotonic"]),
        invalid_rows=int(table["invalid_rows"]),
        source_rows=int(table["source_rows"]),
    )


def _spread_duplicate_seconds(seconds: np.ndarray, *, duration_seconds: float) -> np.ndarray:
    values = np.asarray(seconds, dtype=np.float32).copy()
    if values.size < 2:
        return values
    unique_values, starts, counts = np.unique(values, return_index=True, return_counts=True)
    previous_interval = 1.0
    for index, (base, start, count) in enumerate(zip(unique_values, starts, counts)):
        if int(count) <= 1:
            continue
        if index + 1 < len(unique_values):
            interval = float(unique_values[index + 1] - base)
        else:
            interval = float(duration_seconds - float(base))
        if interval <= 0.0:
            interval = previous_interval
        else:
            previous_interval = interval
        stop = int(start + count)
        values[int(start):stop] = float(base) + (np.arange(int(count), dtype=np.float32) / float(count)) * float(interval)
    if duration_seconds > 0:
        values = np.clip(values, 0.0, np.nextafter(np.float32(duration_seconds), np.float32(0.0)))
    return values


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


def _preprocess_wear_sequence(
    modality: str,
    sequence: np.ndarray,
    *,
    target_rate_hz: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(sequence, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    processed = raw.copy()
    steps: list[str] = []
    if processed.size:
        processed = _robust_winsorize(processed)
        steps.append("robust_winsorize_1_99pct")
        if modality == "ppg":
            baseline = _moving_average(processed, max(3, int(round(float(target_rate_hz) * 2.0))))
            high_passed = processed - baseline
            processed = _moving_average(high_passed, max(3, int(round(float(target_rate_hz) / 5.0))))
            steps.append("bandpass_approx_0.5_5hz")
        elif modality == "gsr":
            processed = _moving_average(processed, max(3, int(round(float(target_rate_hz) / 1.0))))
            steps.append("lowpass_approx_1hz")
        elif modality == "acc":
            processed = processed - np.median(processed, axis=0, keepdims=True)
            steps.append("gravity_median_removed")
            processed = _moving_average(processed, max(3, int(round(float(target_rate_hz) / 5.0))))
            steps.append("lowpass_approx_5hz")
    input_std = float(np.std(raw)) if raw.size else 0.0
    output_std = float(np.std(processed)) if processed.size else 0.0
    mean_abs_delta = float(np.mean(np.abs(processed - raw))) if raw.size else 0.0
    flags = {
        f"{modality}_preprocessing_steps": steps,
        f"{modality}_preprocessing_input_mean": float(np.mean(raw)) if raw.size else 0.0,
        f"{modality}_preprocessing_output_mean": float(np.mean(processed)) if processed.size else 0.0,
        f"{modality}_preprocessing_input_std": input_std,
        f"{modality}_preprocessing_output_std": output_std,
        f"{modality}_preprocessing_mean_abs_delta": mean_abs_delta,
        f"{modality}_preprocessing_changed": bool(mean_abs_delta > 1e-6),
    }
    return processed.astype(np.float32), flags


def _robust_winsorize(values: np.ndarray, *, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()
    clipped = arr.copy()
    for column in range(clipped.shape[1]):
        low, high = np.percentile(clipped[:, column], [lower_pct, upper_pct])
        if np.isfinite(low) and np.isfinite(high) and high >= low:
            clipped[:, column] = np.clip(clipped[:, column], low, high)
    return clipped.astype(np.float32)


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()
    window = max(1, int(window_size))
    if window <= 1 or arr.shape[0] <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(arr, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    smoothed = np.empty_like(arr)
    for column in range(arr.shape[1]):
        smoothed[:, column] = np.convolve(padded[:, column], kernel, mode="valid")
    return smoothed.astype(np.float32)


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
    quality[f"{modality}_duplicate_timestamp_rows"] = int(series.duplicate_timestamp_rows)
    quality[f"{modality}_nonmonotonic_timestamps"] = bool(series.nonmonotonic_timestamps)
    quality[f"{modality}_effective_sampling_rate_hz"] = float(rows / duration_seconds) if duration_seconds > 0 else None
    quality[f"{modality}_sequence_shape"] = list(sequence.shape)
    quality[f"{modality}_missing_ratio"] = 1.0 if rows == 0 else 0.0
    quality[f"{modality}_flatline_ratio"] = _flatline_ratio(sequence)
    quality[f"{modality}_flatline"] = bool(quality[f"{modality}_flatline_ratio"] >= 0.95)


def _flatline_ratio(sequence: np.ndarray, *, tolerance: float = 1e-6) -> float:
    values = np.asarray(sequence, dtype=np.float32)
    if values.size == 0 or values.shape[0] < 2:
        return 1.0
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    ranges = np.ptp(values, axis=0)
    return float(np.mean(ranges <= tolerance)) if ranges.size else 1.0


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


def _deep_sequence_encoder_features(sequences: dict[str, np.ndarray]) -> np.ndarray:
    matrix = _aligned_sequence_matrix(sequences, target_steps=320)
    features: list[float] = []
    features.extend(_sequence_stats(matrix))
    filters = _frozen_conv_filters(channel_count=matrix.shape[1])
    for kernel_size, weights in filters:
        conv = _valid_conv1d(matrix, weights)
        if conv.size == 0:
            continue
        activated = np.tanh(conv)
        features.extend(activated.mean(axis=0).astype(np.float32).tolist())
        features.extend(activated.std(axis=0).astype(np.float32).tolist())
        features.extend(activated.max(axis=0).astype(np.float32).tolist())
        features.extend(activated.min(axis=0).astype(np.float32).tolist())
        features.append(float(np.mean(np.square(activated))))
        features.append(float(kernel_size))
    return np.asarray(features, dtype=np.float32)


def _aligned_sequence_matrix(sequences: dict[str, np.ndarray], *, target_steps: int) -> np.ndarray:
    columns = [
        _resample_array_to_steps(sequences["ppg"][:, :1], target_steps),
        _resample_array_to_steps(sequences["gsr"][:, :1], target_steps),
        _resample_array_to_steps(sequences["acc"], target_steps),
    ]
    matrix = np.concatenate(columns, axis=1).astype(np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((matrix - mean) / std).astype(np.float32)


def _resample_array_to_steps(values: np.ndarray, target_steps: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == target_steps:
        return arr
    if arr.shape[0] == 0:
        return np.zeros((target_steps, arr.shape[1] if arr.ndim == 2 else 1), dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_steps, dtype=np.float32)
    out = np.empty((target_steps, arr.shape[1]), dtype=np.float32)
    for column in range(arr.shape[1]):
        out[:, column] = np.interp(target_x, source_x, arr[:, column])
    return out


def _frozen_conv_filters(*, channel_count: int) -> list[tuple[int, np.ndarray]]:
    filters: list[tuple[int, np.ndarray]] = []
    rng = np.random.default_rng(23017)
    for kernel_size in (3, 5, 9, 15):
        weights = rng.normal(
            0.0,
            1.0 / max(1.0, float(np.sqrt(kernel_size * channel_count))),
            size=(kernel_size, channel_count, 16),
        ).astype(np.float32)
        filters.append((kernel_size, weights))
    return filters


def _valid_conv1d(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    steps = matrix.shape[0] - weights.shape[0] + 1
    if steps <= 0:
        return np.zeros((0, weights.shape[2]), dtype=np.float32)
    out = np.empty((steps, weights.shape[2]), dtype=np.float32)
    for index in range(steps):
        window = matrix[index : index + weights.shape[0], :]
        out[index, :] = np.tensordot(window, weights, axes=([0, 1], [0, 1]))
    return out


def _physio_features_v2(
    sequences: dict[str, np.ndarray],
    qualities: dict[str, Any],
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    ppg = sequences["ppg"][:, 0] if sequences["ppg"].size else np.zeros((0,), dtype=np.float32)
    gsr = sequences["gsr"][:, 0] if sequences["gsr"].size else np.zeros((0,), dtype=np.float32)
    acc = sequences["acc"] if sequences["acc"].size else np.zeros((0, 3), dtype=np.float32)
    ppg_features, ppg_flags = _ppg_physio_features(
        ppg,
        sample_rate_hz=TARGET_SAMPLE_RATES_HZ["ppg"],
        missing_ratio=float(qualities.get("ppg_missing_ratio", 1.0)),
    )
    gsr_features, gsr_flags = _gsr_physio_features(
        gsr,
        sample_rate_hz=TARGET_SAMPLE_RATES_HZ["gsr"],
        missing_ratio=float(qualities.get("gsr_missing_ratio", 1.0)),
    )
    acc_features, acc_flags = _acc_physio_features(acc)
    feature_values = {**ppg_features, **gsr_features, **acc_features}
    feature_names = list(feature_values)
    flags = {**feature_values, **ppg_flags, **gsr_flags, **acc_flags}
    return feature_names, np.asarray([feature_values[name] for name in feature_names], dtype=np.float32), flags


def _ppg_physio_features(
    values: np.ndarray,
    *,
    sample_rate_hz: int,
    missing_ratio: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if values.size == 0:
        peaks = np.zeros((0,), dtype=np.int64)
    else:
        centered = values - float(np.mean(values))
        threshold = float(np.mean(values) + 0.5 * np.std(values))
        candidates = np.flatnonzero(
            (values[1:-1] > values[:-2])
            & (values[1:-1] >= values[2:])
            & (values[1:-1] > threshold)
        ) + 1
        peaks = _enforce_min_peak_distance(candidates, values, min_distance=max(1, int(0.35 * sample_rate_hz)))
        if float(np.std(centered)) < 1e-6:
            peaks = np.zeros((0,), dtype=np.int64)
    peak_count = int(len(peaks))
    if peak_count >= 2:
        ibis = np.diff(peaks).astype(np.float32) / float(sample_rate_hz)
        heart_rate = float(60.0 / np.mean(ibis)) if float(np.mean(ibis)) > 0 else 0.0
        ibi_mean = float(np.mean(ibis))
        ibi_std = float(np.std(ibis))
        rmssd = float(np.sqrt(np.mean(np.square(np.diff(ibis))))) if len(ibis) > 1 else 0.0
    else:
        heart_rate = 0.0
        ibi_mean = 0.0
        ibi_std = 0.0
        rmssd = 0.0
    low_bpm, high_bpm = PLAUSIBLE_HEART_RATE_RANGE_BPM
    heart_rate_plausible = bool(peak_count >= 2 and low_bpm <= heart_rate <= high_bpm)
    return (
        {
            "heart_rate": heart_rate,
            "ibi_mean": ibi_mean,
            "ibi_std": ibi_std,
            "rmssd": rmssd,
            "peak_count": float(peak_count),
            "ppg_missing_ratio": float(missing_ratio),
        },
        {
            "ppg_peak_insufficient": peak_count < 2,
            "heart_rate_plausible": heart_rate_plausible,
            "heart_rate_plausible_range_bpm": [low_bpm, high_bpm],
        },
    )


def _enforce_min_peak_distance(
    candidates: np.ndarray,
    values: np.ndarray,
    *,
    min_distance: int,
) -> np.ndarray:
    if candidates.size == 0:
        return candidates.astype(np.int64)
    selected: list[int] = []
    for candidate in candidates[np.argsort(values[candidates])[::-1]]:
        if all(abs(int(candidate) - existing) >= min_distance for existing in selected):
            selected.append(int(candidate))
    return np.asarray(sorted(selected), dtype=np.int64)


def _gsr_physio_features(
    values: np.ndarray,
    *,
    sample_rate_hz: int,
    missing_ratio: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if values.size == 0:
        tonic_mean = phasic_std = slope = 0.0
        scr_count = 0
    else:
        seconds = np.arange(values.shape[0], dtype=np.float32) / float(sample_rate_hz)
        slope = float(np.polyfit(seconds, values.astype(np.float32), deg=1)[0]) if values.shape[0] > 1 else 0.0
        tonic_mean = float(np.mean(values))
        trend = tonic_mean + slope * (seconds - float(seconds.mean())) if seconds.size else values
        phasic = values - trend
        phasic_std = float(np.std(phasic))
        diff = np.diff(values)
        threshold = float(np.mean(diff) + np.std(diff)) if diff.size else math.inf
        scr_count = int(np.sum(diff > threshold)) if diff.size else 0
    return (
        {
            "tonic_mean": tonic_mean,
            "phasic_std": phasic_std,
            "scr_count": float(scr_count),
            "gsr_slope": slope,
            "gsr_missing_ratio": float(missing_ratio),
        },
        {},
    )


def _acc_physio_features(acc_sequence: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    if acc_sequence.size == 0:
        motion_intensity = stationary_ratio = axis_std = spectral_energy = 0.0
    else:
        delta = np.diff(acc_sequence, axis=0)
        delta_norm = np.linalg.norm(delta, axis=1) if delta.size else np.zeros((0,), dtype=np.float32)
        motion_intensity = float(np.mean(delta_norm)) if delta_norm.size else 0.0
        stationary_ratio = float(np.mean(delta_norm < 0.05)) if delta_norm.size else 1.0
        axis_std = float(np.mean(np.std(acc_sequence, axis=0)))
        centered_norm = np.linalg.norm(acc_sequence - np.mean(acc_sequence, axis=0, keepdims=True), axis=1)
        spectrum = np.abs(np.fft.rfft(centered_norm)) if centered_norm.size else np.zeros((0,), dtype=np.float32)
        spectral_energy = float(np.mean(np.square(spectrum))) if spectrum.size else 0.0
    return (
        {
            "motion_intensity": motion_intensity,
            "stationary_ratio": stationary_ratio,
            "axis_std": axis_std,
            "spectral_energy": spectral_energy,
        },
        {},
    )


def _add_wear_quality_grade(quality: dict[str, Any]) -> None:
    invalid_ratio_zero = all(
        float(quality.get(f"{modality}_invalid_rows", 0.0)) == 0.0
        for modality in ("ppg", "gsr", "acc")
    )
    flatline = any(bool(quality.get(f"{modality}_flatline", False)) for modality in ("ppg", "gsr"))
    peak_sufficient = not bool(quality.get("ppg_peak_insufficient", True))
    hr_plausible = bool(quality.get("heart_rate_plausible", False))
    gsr_slope_abnormal = abs(float(quality.get("gsr_slope", 0.0))) > GSR_SLOPE_ABNORMAL_ABS_THRESHOLD
    gsr_scr_abnormal = float(quality.get("scr_count", 0.0)) > GSR_SCR_ABNORMAL_HIGH_THRESHOLD
    motion_intensity = float(quality.get("motion_intensity", 0.0))
    stationary_ratio = float(quality.get("stationary_ratio", 0.0))
    acc_motion_high = motion_intensity > ACC_MOTION_HIGH_THRESHOLD
    acc_stable = stationary_ratio >= ACC_STABLE_STATIONARY_RATIO_THRESHOLD
    motion_artifact_risk = bool(acc_motion_high or (not hr_plausible and motion_intensity > 0.1 and not acc_stable))
    risk_count = sum(
        [
            not hr_plausible,
            not peak_sufficient,
            gsr_slope_abnormal,
            gsr_scr_abnormal,
            acc_motion_high,
            flatline,
            not invalid_ratio_zero,
        ]
    )
    if (
        hr_plausible
        and peak_sufficient
        and invalid_ratio_zero
        and not flatline
        and not acc_motion_high
        and (not gsr_slope_abnormal or acc_stable)
    ):
        grade = "A"
        label = "high"
        recommended = ["wear_physio_features_v2", "wear_deep_sequence_v1", "multimodal_fusion"]
    elif (not peak_sufficient) or (gsr_slope_abnormal and acc_motion_high) or risk_count >= 3:
        grade = "C"
        label = "low"
        recommended = ["quality_audit_only"]
    else:
        grade = "B"
        label = "medium"
        recommended = ["wear_deep_sequence_v1", "quality_flagged_training"]

    quality.update(
        {
            "wear_quality_grade": grade,
            "wear_quality_label": label,
            "ppg_hr_plausible": hr_plausible,
            "ppg_peak_sufficient": peak_sufficient,
            "gsr_slope_abnormal": gsr_slope_abnormal,
            "gsr_scr_abnormal": gsr_scr_abnormal,
            "acc_motion_high": acc_motion_high,
            "acc_stable": acc_stable,
            "motion_artifact_risk": motion_artifact_risk,
            "wear_invalid_ratio_zero": invalid_ratio_zero,
            "wear_quality_risk_count": int(risk_count),
            "wear_quality_recommended_use": recommended,
            "wear_quality_thresholds": {
                "gsr_slope_abs_abnormal": GSR_SLOPE_ABNORMAL_ABS_THRESHOLD,
                "gsr_scr_count_abnormal_high": GSR_SCR_ABNORMAL_HIGH_THRESHOLD,
                "acc_motion_high": ACC_MOTION_HIGH_THRESHOLD,
                "acc_stable_stationary_ratio": ACC_STABLE_STATIONARY_RATIO_THRESHOLD,
                "heart_rate_plausible_range_bpm": list(PLAUSIBLE_HEART_RATE_RANGE_BPM),
            },
        }
    )


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
        "quality_audit": _quality_audit(samples, failures),
    }


def _mean_quality(qualities: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in qualities if item.get(key) is not None]
    return None if not values else float(np.mean(values))


def _quality_audit(samples: list[dict[str, Any]], failures: list[EmbeddingFailure]) -> dict[str, Any]:
    qualities = [sample["quality_flags"] for sample in samples]
    return {
        "window_count": len(samples) + len([failure for failure in failures if failure.stage != "quality_gate"]),
        "embedded_count": len(samples),
        "success_count": sum(int(sample["modality_mask"][1]) == 1 for sample in samples),
        "failure_count": len(failures),
        "failure_types": _count_by_error_type(failures),
        "masked_count": sum(int(sample["modality_mask"][1]) == 0 for sample in samples),
        "modalities": {
            modality: _modality_quality_audit(qualities, modality)
            for modality in ("ppg", "gsr", "acc")
        },
        "ppg": {
            "peak_count": _numeric_summary(qualities, "peak_count"),
            "heart_rate": _numeric_summary(qualities, "heart_rate"),
            "heart_rate_plausible_range_bpm": list(PLAUSIBLE_HEART_RATE_RANGE_BPM),
            "heart_rate_plausible_count": _bool_count(qualities, "heart_rate_plausible", True),
            "heart_rate_implausible_count": _bool_count(qualities, "heart_rate_plausible", False),
            "peak_insufficient_count": _bool_count(qualities, "ppg_peak_insufficient", True),
        },
        "gsr": {
            "slope": _numeric_summary(qualities, "gsr_slope"),
            "scr_count": _numeric_summary(qualities, "scr_count"),
            **_iqr_abnormal_summary(qualities, "gsr_slope", prefix="slope"),
            **_iqr_abnormal_summary(qualities, "scr_count", prefix="scr_count"),
        },
        "acc": {
            "motion_intensity": _numeric_summary(qualities, "motion_intensity"),
            "stationary_ratio": _numeric_summary(qualities, "stationary_ratio"),
        },
        "wear_quality_grade_counts": {
            grade: _string_count(qualities, "wear_quality_grade", grade)
            for grade in ("A", "B", "C")
        },
        "motion_artifact_risk_count": _bool_count(qualities, "motion_artifact_risk", True),
        "low_quality_masked_count": _bool_count(qualities, "wear_low_quality_masked", True),
    }


def _modality_quality_audit(qualities: list[dict[str, Any]], modality: str) -> dict[str, Any]:
    return {
        "rows_in_window": _numeric_summary(qualities, f"{modality}_rows_in_window"),
        "effective_sampling_rate_hz": _numeric_summary(qualities, f"{modality}_effective_sampling_rate_hz"),
        "invalid_rows": _numeric_summary(qualities, f"{modality}_invalid_rows"),
        "source_rows": _numeric_summary(qualities, f"{modality}_source_rows"),
        "invalid_rows_per_source_row": _ratio_summary(
            qualities,
            numerator_key=f"{modality}_invalid_rows",
            denominator_key=f"{modality}_source_rows",
        ),
        "duplicate_timestamps_count": _bool_count(qualities, f"{modality}_duplicate_timestamps", True),
        "duplicate_timestamp_rows": _numeric_summary(qualities, f"{modality}_duplicate_timestamp_rows"),
        "nonmonotonic_timestamps_count": _bool_count(qualities, f"{modality}_nonmonotonic_timestamps", True),
        "flatline_ratio": _numeric_summary(qualities, f"{modality}_flatline_ratio"),
        "flatline_window_count": _bool_count(qualities, f"{modality}_flatline", True),
    }


def _numeric_summary(qualities: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(item[key]) for item in qualities if item.get(key) is not None]
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
            "sum": 0.0,
            "zero_count": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "zero_count": int(np.sum(arr == 0.0)),
    }


def _ratio_summary(
    qualities: list[dict[str, Any]],
    *,
    numerator_key: str,
    denominator_key: str,
) -> dict[str, float | int | None]:
    ratios: list[dict[str, float]] = []
    for item in qualities:
        numerator = item.get(numerator_key)
        denominator = item.get(denominator_key)
        if numerator is None or denominator is None or float(denominator) <= 0.0:
            continue
        ratios.append({"ratio": float(numerator) / float(denominator)})
    return _numeric_summary(ratios, "ratio")


def _bool_count(qualities: list[dict[str, Any]], key: str, expected: bool) -> int:
    return sum(1 for item in qualities if item.get(key) is expected)


def _string_count(qualities: list[dict[str, Any]], key: str, expected: str) -> int:
    return sum(1 for item in qualities if str(item.get(key, "")) == expected)


def _iqr_abnormal_summary(
    qualities: list[dict[str, Any]],
    key: str,
    *,
    prefix: str,
) -> dict[str, Any]:
    values = np.asarray([float(item[key]) for item in qualities if item.get(key) is not None], dtype=np.float64)
    if values.size < 4:
        return {
            f"{prefix}_abnormal_count": 0,
            f"{prefix}_abnormal_low": None,
            f"{prefix}_abnormal_high": None,
        }
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        low = high = float(np.median(values))
        count = int(np.sum(values != low))
    else:
        low = float(q1 - 1.5 * iqr)
        high = float(q3 + 1.5 * iqr)
        count = int(np.sum((values < low) | (values > high)))
    return {
        f"{prefix}_abnormal_count": count,
        f"{prefix}_abnormal_low": low,
        f"{prefix}_abnormal_high": high,
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
        modality="wear",
        encoder_profile=encoder_profile,
        stage=stage,
        error_type=error_type,
        error=error,
        source_path=source_path,
        recoverable=True,
    )
