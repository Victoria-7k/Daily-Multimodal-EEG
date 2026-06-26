from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.alignment.time_utils import parse_absolute_time


EMBED_DIM = 256
MODALITY_ORDER = ("eeg", "wear", "face", "audio")


@dataclass
class EmbeddingSample:
    sample_id: str
    event_id: str
    subject_id: str
    session_id: str
    window_start_time: str
    window_end_time: str
    eeg_emb: np.ndarray
    wear_emb: np.ndarray
    face_emb: np.ndarray
    audio_emb: np.ndarray
    modality_mask: np.ndarray
    labels: dict[str, Any]
    source_paths: dict[str, Any]
    encoder_versions: dict[str, str]
    quality_flags: dict[str, Any]


def extract_basic_embedding(window: dict[str, Any]) -> EmbeddingSample:
    """Extract deterministic stage-5 smoke embeddings for one window."""
    eeg_emb, eeg_quality, eeg_available = _metadata_embedding(
        "eeg",
        [window.get("eeg_bdf_path", "")],
        window,
        enabled=bool(window.get("has_eeg")),
    )
    wear_emb, wear_quality, wear_available = _wear_embedding(window)
    face_emb, face_quality, face_available = _metadata_embedding(
        "face",
        _as_list(window.get("candidate_mp4_paths")),
        window,
        enabled=bool(window.get("has_face")),
    )
    audio_emb, audio_quality, audio_available = _metadata_embedding(
        "audio",
        _as_list(window.get("candidate_audio_paths")) or _as_list(window.get("candidate_mp4_paths")),
        window,
        enabled=bool(window.get("has_audio")),
    )

    return EmbeddingSample(
        sample_id=window["sample_id"],
        event_id=window.get("event_id", ""),
        subject_id=window.get("subject_id", ""),
        session_id=window.get("session_id", ""),
        window_start_time=window.get("window_start_time", ""),
        window_end_time=window.get("window_end_time", ""),
        eeg_emb=eeg_emb,
        wear_emb=wear_emb,
        face_emb=face_emb,
        audio_emb=audio_emb,
        modality_mask=np.array(
            [eeg_available, wear_available, face_available, audio_available],
            dtype=np.int8,
        ),
        labels=window.get("label_columns") or window.get("labels") or {},
        source_paths={
            "eeg": window.get("eeg_bdf_path", ""),
            "wear_ppg": window.get("wear_ppg_path", ""),
            "wear_gsr": window.get("wear_gsr_path", ""),
            "wear_acc": window.get("wear_acc_path", ""),
            "face": _as_list(window.get("candidate_mp4_paths")),
            "audio": _as_list(window.get("candidate_audio_paths")) or _as_list(window.get("candidate_mp4_paths")),
        },
        encoder_versions={
            "eeg": "basic_smoke_metadata_v1",
            "wear": "basic_wear_statistics_v1",
            "face": "basic_smoke_metadata_v1",
            "audio": "basic_smoke_metadata_v1",
        },
        quality_flags={
            "eeg": eeg_quality,
            "wear": wear_quality,
            "face": face_quality,
            "audio": audio_quality,
        },
    )


def _wear_embedding(window: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], bool]:
    ppg_stats = _read_numeric_window_stats(
        window.get("wear_ppg_path", ""),
        time_column="csv_time_PPG",
        value_columns=["PPG"],
        start_time=window.get("window_start_time", ""),
        end_time=window.get("window_end_time", ""),
    )
    gsr_stats = _read_numeric_window_stats(
        window.get("wear_gsr_path", ""),
        time_column="csv_time_GSR",
        value_columns=["GSR"],
        start_time=window.get("window_start_time", ""),
        end_time=window.get("window_end_time", ""),
    )
    acc_stats = _read_numeric_window_stats(
        window.get("wear_acc_path", ""),
        time_column="csv_time_motion",
        value_columns=["Motion_dataX", "Motion_dataY", "Motion_dataZ"],
        start_time=window.get("window_start_time", ""),
        end_time=window.get("window_end_time", ""),
    )
    feature_values = (
        ppg_stats["features"]
        + gsr_stats["features"]
        + acc_stats["features"]
    )
    available = any(item["rows"] > 0 for item in [ppg_stats, gsr_stats, acc_stats])
    quality = {
        "ppg_rows": ppg_stats["rows"],
        "gsr_rows": gsr_stats["rows"],
        "acc_rows": acc_stats["rows"],
        "available": available,
    }
    if not available:
        return np.zeros(EMBED_DIM, dtype=np.float32), quality, False
    return _features_to_embedding(feature_values, salt="wear"), quality, True


def _read_numeric_window_stats(
    path_text: str,
    *,
    time_column: str,
    value_columns: list[str],
    start_time: str,
    end_time: str,
) -> dict[str, Any]:
    path = Path(path_text) if path_text else None
    if not path or not path.is_file():
        return {"rows": 0, "features": [0.0] * (len(value_columns) * 4)}
    start = parse_absolute_time(start_time)
    end = parse_absolute_time(end_time)
    values: list[list[float]] = [[] for _ in value_columns]
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_time = row.get(time_column)
            if not raw_time:
                continue
            try:
                timestamp = parse_absolute_time(raw_time)
            except ValueError:
                continue
            if timestamp < start:
                continue
            if timestamp >= end:
                break
            for index, column in enumerate(value_columns):
                try:
                    values[index].append(float(row.get(column, "")))
                except ValueError:
                    continue
    features: list[float] = []
    rows = max((len(column_values) for column_values in values), default=0)
    for column_values in values:
        features.extend(_basic_stats(column_values))
    return {"rows": rows, "features": features}


def _basic_stats(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.array(values, dtype=np.float32)
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
    ]


def _metadata_embedding(
    modality: str,
    paths: list[str],
    window: dict[str, Any],
    *,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    existing = [path for path in paths if path and Path(path).is_file()]
    available = bool(enabled and existing)
    quality = {
        "candidate_count": len([path for path in paths if path]),
        "existing_count": len(existing),
        "available": available,
        "smoke_encoder_note": "metadata-derived placeholder until raw encoder is enabled",
    }
    if not available:
        return np.zeros(EMBED_DIM, dtype=np.float32), quality, False
    sizes = [float(Path(path).stat().st_size) for path in existing[:4]]
    seconds = _window_duration_seconds(window)
    seed_text = f"{modality}|{window.get('sample_id', '')}|{'|'.join(existing[:4])}"
    return _features_to_embedding([seconds, *sizes], salt=seed_text), quality, True


def _window_duration_seconds(window: dict[str, Any]) -> float:
    try:
        start = parse_absolute_time(window["window_start_time"])
        end = parse_absolute_time(window["window_end_time"])
        return (end - start).total_seconds()
    except Exception:
        return float(window.get("window_size_seconds") or 0.0)


def _features_to_embedding(features: list[float], *, salt: str) -> np.ndarray:
    cleaned = np.array([_finite(value) for value in features], dtype=np.float32)
    if cleaned.size == 0:
        cleaned = np.zeros(1, dtype=np.float32)
    normalized = cleaned.copy()
    std = float(normalized.std())
    if std > 0:
        normalized = (normalized - float(normalized.mean())) / std
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    projection = np.empty(EMBED_DIM, dtype=np.float32)
    for index in range(EMBED_DIM):
        base = normalized[index % normalized.size]
        phase = digest[index % len(digest)] / 255.0
        projection[index] = math.tanh(float(base) + phase)
    return projection.astype(np.float32)


def _finite(value: float) -> float:
    number = float(value)
    if math.isfinite(number):
        return number
    return 0.0


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]
