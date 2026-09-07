from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EMBEDDING_DIM = 256
TOKEN_SECONDS = 2
MODALITY_ORDER = ("eeg", "wear", "video", "audio")
TOKEN_KEYS = {name: f"{name}_tokens" for name in MODALITY_ORDER}


def load_npz_dict(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def validate_modality_temporal_npz(
    path: Path | str,
    *,
    modality: str,
    expected_sample_id: np.ndarray | None = None,
    expected_token_count: int = 5,
    expected_dim: int = EMBEDDING_DIM,
) -> dict[str, Any]:
    if modality not in MODALITY_ORDER:
        raise ValueError(f"unsupported modality {modality!r}; expected one of {MODALITY_ORDER}")
    data = load_npz_dict(path)
    token_key = TOKEN_KEYS[modality]
    required = {"sample_id", token_key}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    sample_id = data["sample_id"].astype(str)
    if expected_sample_id is not None and not np.array_equal(sample_id, expected_sample_id.astype(str)):
        raise ValueError(f"{path} sample_id order does not match temporal index")
    tokens = _validate_tokens(data[token_key], key=token_key, row_count=len(sample_id), token_count=expected_token_count, dim=expected_dim)
    mask = _modality_mask_from_data(data, modality=modality, row_count=len(sample_id), token_count=expected_token_count)
    quality = _quality_from_data(data, modality=modality, row_count=len(sample_id), token_count=expected_token_count)
    return {
        "path": str(path),
        "modality": modality,
        "row_count": int(len(sample_id)),
        "token_shape": list(tokens.shape),
        "mask_available_ratio": float(mask.mean()) if mask.size else 0.0,
        "quality_shape": list(quality.shape),
    }


def validate_packed_temporal_npz(
    path: Path | str,
    *,
    expected_sample_id: np.ndarray | None = None,
    expected_token_count: int = 5,
    expected_dim: int = EMBEDDING_DIM,
) -> dict[str, Any]:
    data = load_npz_dict(path)
    required = {
        "sample_id",
        "token_start_seconds",
        "token_end_seconds",
        "modality_order",
        "token_mask",
        "modality_mask",
        "quality_features",
    } | set(TOKEN_KEYS.values())
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    sample_id = data["sample_id"].astype(str)
    if expected_sample_id is not None and not np.array_equal(sample_id, expected_sample_id.astype(str)):
        raise ValueError(f"{path} sample_id order does not match expected sample_id")
    modality_order = tuple(str(value) for value in data["modality_order"].tolist())
    if modality_order != MODALITY_ORDER:
        raise ValueError(f"{path} modality_order expected {MODALITY_ORDER}, got {modality_order}")
    for modality in MODALITY_ORDER:
        _validate_tokens(data[TOKEN_KEYS[modality]], key=TOKEN_KEYS[modality], row_count=len(sample_id), token_count=expected_token_count, dim=expected_dim)
    token_mask = _validate_token_mask(data["token_mask"], row_count=len(sample_id), token_count=expected_token_count)
    modality_mask = np.asarray(data["modality_mask"])
    if modality_mask.shape != (len(sample_id), len(MODALITY_ORDER)):
        raise ValueError(f"modality_mask expected shape {(len(sample_id), len(MODALITY_ORDER))}, got {modality_mask.shape}")
    quality = np.asarray(data["quality_features"])
    if quality.ndim != 4 or quality.shape[:3] != (len(sample_id), len(MODALITY_ORDER), expected_token_count):
        raise ValueError(
            f"quality_features expected shape (N,{len(MODALITY_ORDER)},{expected_token_count},Q), got {quality.shape}"
        )
    if np.isnan(quality).any() or not np.isfinite(quality).all():
        raise ValueError("quality_features contains NaN or infinite values")
    return {
        "path": str(path),
        "row_count": int(len(sample_id)),
        "token_count": int(expected_token_count),
        "embedding_dim": int(expected_dim),
        "modality_order": list(modality_order),
        "available_token_ratio": float(token_mask.mean()) if token_mask.size else 0.0,
        "quality_feature_dim": int(quality.shape[-1]),
    }


def write_packed_temporal_tokens(
    *,
    output_path: Path | str,
    temporal_index_rows: list[dict[str, Any]],
    modality_arrays: Mapping[str, Mapping[str, np.ndarray]],
    encoder_versions: Mapping[str, Any] | None = None,
    source_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not temporal_index_rows:
        raise ValueError("temporal_index_rows must not be empty")
    sample_id = np.asarray([str(row["sample_id"]) for row in temporal_index_rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in temporal_index_rows], dtype=str)
    subject_id = np.asarray([str(row.get("subject_id", "")) for row in temporal_index_rows], dtype=str)
    day_id = np.asarray([_day_id(row) for row in temporal_index_rows], dtype=str)
    labels, label_names = _labels_from_rows(temporal_index_rows)
    token_start = _shared_time_boundaries(temporal_index_rows, "token_start_seconds")
    token_end = _shared_time_boundaries(temporal_index_rows, "token_end_seconds")
    token_count = len(token_start)

    packed: dict[str, Any] = {
        "sample_id": sample_id,
        "event_id": event_id,
        "subject_id": subject_id,
        "day_id": day_id,
        "label_names": np.asarray(label_names, dtype=object),
        "labels": labels,
        "token_start_seconds": np.asarray(token_start, dtype=np.float32),
        "token_end_seconds": np.asarray(token_end, dtype=np.float32),
        "modality_order": np.asarray(MODALITY_ORDER, dtype=object),
    }
    token_masks: list[np.ndarray] = []
    quality_blocks: list[np.ndarray] = []
    quality_names: dict[str, list[str]] = {}
    for modality in MODALITY_ORDER:
        if modality not in modality_arrays:
            raise ValueError(f"missing modality arrays for {modality}")
        arrays = modality_arrays[modality]
        tokens = _validate_tokens(
            arrays[TOKEN_KEYS[modality]],
            key=TOKEN_KEYS[modality],
            row_count=len(sample_id),
            token_count=token_count,
            dim=EMBEDDING_DIM,
        )
        mask = _modality_mask_from_data(arrays, modality=modality, row_count=len(sample_id), token_count=token_count)
        quality = _quality_from_data(arrays, modality=modality, row_count=len(sample_id), token_count=token_count)
        packed[TOKEN_KEYS[modality]] = tokens
        token_masks.append(mask.astype(np.int8))
        quality_blocks.append(quality.astype(np.float32))
        quality_names[modality] = _quality_names_from_data(arrays, modality=modality, feature_dim=quality.shape[-1])
    token_mask = np.stack(token_masks, axis=1).astype(np.int8)
    modality_mask = token_mask.any(axis=2).astype(np.int8)
    quality_features = _pad_quality_blocks(quality_blocks)
    packed.update(
        {
            "token_mask": token_mask,
            "modality_mask": modality_mask,
            "quality_features": quality_features,
            "quality_feature_names_json": np.asarray([json.dumps(quality_names, ensure_ascii=False)], dtype=object),
            "encoder_versions_json": np.asarray([json.dumps(dict(encoder_versions or {}), ensure_ascii=False)], dtype=object),
            "source_paths_json": np.asarray([json.dumps(dict(source_paths or {}), ensure_ascii=False)], dtype=object),
        }
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **packed)
    return validate_packed_temporal_npz(out, expected_sample_id=sample_id, expected_token_count=token_count)


def _validate_tokens(array: np.ndarray, *, key: str, row_count: int, token_count: int, dim: int) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (row_count, token_count, dim):
        raise ValueError(f"{key} expected shape {(row_count, token_count, dim)}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{key} must use a floating dtype, got {value.dtype}")
    if np.isnan(value).any() or not np.isfinite(value).all():
        raise ValueError(f"{key} contains NaN or infinite values")
    return value.astype(np.float32, copy=False)


def _validate_token_mask(array: np.ndarray, *, row_count: int, token_count: int) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (row_count, len(MODALITY_ORDER), token_count):
        raise ValueError(f"token_mask expected shape {(row_count, len(MODALITY_ORDER), token_count)}, got {value.shape}")
    return value.astype(bool, copy=False)


def _modality_mask_from_data(data: Mapping[str, np.ndarray], *, modality: str, row_count: int, token_count: int) -> np.ndarray:
    for key in (f"{modality}_token_mask", "token_mask"):
        if key in data:
            value = np.asarray(data[key])
            if value.shape == (row_count, token_count):
                return value.astype(bool, copy=False)
            if value.shape == (row_count, len(MODALITY_ORDER), token_count):
                return value[:, MODALITY_ORDER.index(modality)].astype(bool, copy=False)
            raise ValueError(f"{key} has unsupported shape {value.shape}")
    return np.ones((row_count, token_count), dtype=bool)


def _quality_from_data(data: Mapping[str, np.ndarray], *, modality: str, row_count: int, token_count: int) -> np.ndarray:
    for key in (f"{modality}_quality_features", "quality_features"):
        if key in data:
            value = np.asarray(data[key])
            if value.shape[:2] == (row_count, token_count) and value.ndim == 3:
                quality = value
            elif value.ndim == 4 and value.shape[:3] == (row_count, len(MODALITY_ORDER), token_count):
                quality = value[:, MODALITY_ORDER.index(modality)]
            else:
                raise ValueError(f"{key} has unsupported shape {value.shape}")
            if np.isnan(quality).any() or not np.isfinite(quality).all():
                raise ValueError(f"{key} contains NaN or infinite values")
            return quality.astype(np.float32, copy=False)
    return np.ones((row_count, token_count, 1), dtype=np.float32)


def _quality_names_from_data(data: Mapping[str, np.ndarray], *, modality: str, feature_dim: int) -> list[str]:
    for key in (f"{modality}_quality_feature_names_json", "quality_feature_names_json"):
        if key in data:
            raw = data[key].tolist()
            text = raw[0] if isinstance(raw, list) else raw
            try:
                parsed = json.loads(str(text))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and modality in parsed:
                return [str(value) for value in parsed[modality]]
            if isinstance(parsed, list):
                return [str(value) for value in parsed]
    return [f"{modality}_quality_{idx}" for idx in range(feature_dim)]


def _pad_quality_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    max_dim = max(block.shape[-1] for block in blocks)
    padded = []
    for block in blocks:
        if block.shape[-1] == max_dim:
            padded.append(block)
            continue
        pad_width = [(0, 0), (0, 0), (0, max_dim - block.shape[-1])]
        padded.append(np.pad(block, pad_width=pad_width, mode="constant"))
    return np.stack(padded, axis=1).astype(np.float32)


def _shared_time_boundaries(rows: list[dict[str, Any]], key: str) -> list[float]:
    first = [float(value) for value in rows[0][key]]
    for row in rows[1:]:
        values = [float(value) for value in row[key]]
        if values != first:
            raise ValueError(f"{key} differs across temporal index rows")
    return first


def _labels_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = rows[0].get("label_names")
    if isinstance(names, list) and names:
        label_names = tuple(str(value) for value in names)
    else:
        labels = rows[0].get("labels") or rows[0].get("label_columns")
        if isinstance(labels, dict):
            label_names = tuple(str(key) for key in labels.keys())
        elif isinstance(labels, list):
            label_names = tuple(f"label_{idx}" for idx in range(len(labels)))
        else:
            label_names = tuple()
    values: list[list[float]] = []
    for row in rows:
        labels = row.get("labels") or row.get("label_columns") or {}
        if isinstance(labels, dict):
            values.append([float(labels[name]) for name in label_names])
        elif isinstance(labels, list):
            values.append([float(labels[idx]) for idx in range(len(label_names))])
        elif label_names:
            raise ValueError("row has unsupported labels field")
        else:
            values.append([])
    return np.asarray(values, dtype=np.float32), label_names


def _day_id(row: Mapping[str, Any]) -> str:
    for key in ("day_id", "session_id", "date", "recording_day"):
        value = row.get(key)
        if value:
            return str(value)
    onset = str(row.get("absolute_onset_time", ""))
    return onset[:10] if len(onset) >= 10 else ""
