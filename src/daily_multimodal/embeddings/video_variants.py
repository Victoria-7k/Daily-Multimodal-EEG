from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape
from daily_multimodal.embeddings.video_behavior_flags import BEHAVIOR_RATIO_NAMES


FACE_MASK_INDEX = 2
BEHAVIOR_FLAG_VECTOR = tuple(BEHAVIOR_RATIO_NAMES)
VIDEO_VARIANTS = {
    "openface_only_v1",
    "behavior_flags_only_v2_probe",
    "openface_behavior_flags_v2",
}


def build_video_variant_embeddings(
    *,
    variant: str,
    window_index_path: Path | str,
    out_path: Path | str,
    behavior_flags_path: Path | str | None = None,
    openface_embeddings_path: Path | str | None = None,
    sample_mode: str = "behavior_retained",
) -> dict[str, Any]:
    if variant not in VIDEO_VARIANTS:
        raise ValueError(f"unsupported video variant: {variant}")
    windows = _read_jsonl(Path(window_index_path))
    behavior_rows = _load_behavior_rows(behavior_flags_path)
    openface = _load_openface_bundle(openface_embeddings_path)

    rows: list[dict[str, Any]] = []
    for window in windows:
        sample_id = str(window.get("sample_id", ""))
        behavior = behavior_rows.get(sample_id)
        openface_record = openface.get(sample_id)
        if variant == "openface_only_v1":
            row = _openface_only_row(window, openface_record)
        elif variant == "behavior_flags_only_v2_probe":
            row = _behavior_only_row(window, behavior)
        else:
            row = _openface_behavior_row(
                window,
                behavior,
                openface_record,
                sample_mode=sample_mode,
            )
        rows.append(row)

    _write_variant_npz(rows, out_path)
    return {
        "variant": variant,
        "sample_mode": sample_mode,
        "row_count": len(rows),
        "mask_sum": int(sum(int(row["modality_mask"][FACE_MASK_INDEX]) for row in rows)),
        "out_path": str(out_path),
    }


def _openface_only_row(
    window: dict[str, Any],
    openface_record: dict[str, Any] | None,
) -> dict[str, Any]:
    mask_value = _face_mask(openface_record)
    embedding = (
        openface_record["face_emb"].copy()
        if openface_record is not None and mask_value
        else np.zeros(EMBEDDING_DIM, dtype=np.float32)
    )
    quality = _base_quality("openface_only_v1")
    if openface_record is None:
        quality["missing_openface"] = True
    else:
        quality["openface_mask_value"] = mask_value
        quality["openface_quality_flags"] = openface_record["quality_flags"]
    return _variant_row(
        window,
        embedding=embedding,
        mask_value=mask_value,
        quality_flags=quality,
        encoder_version="openface_only_v1",
    )


def _behavior_only_row(
    window: dict[str, Any],
    behavior: dict[str, Any] | None,
) -> dict[str, Any]:
    behavior_usable = _behavior_usable(behavior)
    vector = _behavior_vector(behavior) if behavior_usable else np.zeros(len(BEHAVIOR_FLAG_VECTOR), dtype=np.float32)
    embedding = (
        _project_to_256(vector, salt="behavior_flags_only_v2_probe")
        if behavior_usable
        else np.zeros(EMBEDDING_DIM, dtype=np.float32)
    )
    quality = _base_quality("behavior_flags_only_v2_probe")
    quality["behavior_flags_present"] = behavior is not None
    quality["behavior_usable"] = behavior_usable
    return _variant_row(
        window,
        embedding=embedding,
        mask_value=1 if behavior_usable else 0,
        quality_flags=quality,
        encoder_version="behavior_flags_only_v2_probe",
    )


def _openface_behavior_row(
    window: dict[str, Any],
    behavior: dict[str, Any] | None,
    openface_record: dict[str, Any] | None,
    *,
    sample_mode: str,
) -> dict[str, Any]:
    if sample_mode not in {"strict_aligned", "behavior_retained"}:
        raise ValueError(f"unsupported sample_mode: {sample_mode}")
    behavior_usable = _behavior_usable(behavior)
    openface_mask = _face_mask(openface_record)
    strict_usable = bool(behavior_usable and openface_record is not None and openface_mask)
    retained_usable = bool(behavior_usable)
    mask_value = 1 if (strict_usable if sample_mode == "strict_aligned" else retained_usable) else 0

    if mask_value:
        openface_vector = (
            openface_record["face_emb"]
            if openface_record is not None and openface_mask
            else np.zeros(EMBEDDING_DIM, dtype=np.float32)
        )
        vector = np.concatenate([openface_vector.astype(np.float32), _behavior_vector(behavior)])
        embedding = _project_to_256(vector, salt=f"openface_behavior_flags_v2:{sample_mode}")
    else:
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    quality = _base_quality("openface_behavior_flags_v2")
    quality["sample_mode"] = sample_mode
    quality["behavior_flags_present"] = behavior is not None
    quality["behavior_usable"] = behavior_usable
    quality["openface_present"] = openface_record is not None
    quality["openface_mask_value"] = openface_mask
    if openface_record is not None:
        quality["openface_quality_flags"] = openface_record["quality_flags"]
    quality["behavior_retained_without_openface_mask"] = bool(
        sample_mode == "behavior_retained"
        and behavior_usable
        and (openface_record is None or not openface_mask)
    )
    return _variant_row(
        window,
        embedding=embedding,
        mask_value=mask_value,
        quality_flags=quality,
        encoder_version="openface_behavior_flags_v2",
    )


def _variant_row(
    window: dict[str, Any],
    *,
    embedding: np.ndarray,
    mask_value: int,
    quality_flags: dict[str, Any],
    encoder_version: str,
) -> dict[str, Any]:
    mask = np.zeros(4, dtype=np.int8)
    mask[FACE_MASK_INDEX] = int(mask_value)
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "labels": _window_labels(window),
        "face_emb": validate_embedding_shape("face_emb", np.asarray(embedding, dtype=np.float32)),
        "modality_mask": mask,
        "quality_flags": quality_flags,
        "encoder_version": encoder_version,
    }


def _load_behavior_rows(path: Path | str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(Path(path)):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in records:
            raise ValueError(f"{path} contains duplicate sample_id {sample_id!r}")
        records[sample_id] = row
    return records


def _load_openface_bundle(path: Path | str | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as loaded:
        sample_ids = loaded["sample_id"].astype(str).tolist()
        embeddings = validate_embedding_shape("face_emb", loaded["face_emb"])
        masks = loaded["modality_mask"].astype(np.int8)
        quality_values = loaded["quality_flags"].tolist() if "quality_flags" in loaded.files else ["{}"] * len(sample_ids)
        if embeddings.shape[0] != len(sample_ids) or masks.shape[0] != len(sample_ids):
            raise ValueError(f"{path} has inconsistent row counts for openface embeddings")
        if len(quality_values) != len(sample_ids):
            raise ValueError(f"{path} has inconsistent row counts for openface quality_flags")
        for idx, sample_id in enumerate(sample_ids):
            if sample_id in records:
                raise ValueError(f"{path} contains duplicate sample_id {sample_id!r}")
            records[sample_id] = {
                "face_emb": embeddings[idx].astype(np.float32, copy=False),
                "mask_value": int(masks[idx, FACE_MASK_INDEX]) if masks.ndim == 2 else 0,
                "quality_flags": _json_ready(_parse_json_object(quality_values[idx])),
            }
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _behavior_vector(row: dict[str, Any] | None) -> np.ndarray:
    return np.asarray(
        [float((row or {}).get(name, 0.0) or 0.0) for name in BEHAVIOR_FLAG_VECTOR],
        dtype=np.float32,
    )


def _behavior_usable(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    try:
        return int(row.get("usable_frame_count", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _face_mask(openface_record: dict[str, Any] | None) -> int:
    if openface_record is None:
        return 0
    return 1 if int(openface_record.get("mask_value", 0)) else 0


def _project_to_256(vector: np.ndarray, *, salt: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    std = float(values.std())
    normalized = values.copy()
    if std > 0:
        normalized = (normalized - float(values.mean())) / std
    digest = hashlib.sha256(salt.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(1.0, float(np.sqrt(normalized.size)))
    weights = rng.normal(0.0, scale, size=(normalized.size, EMBEDDING_DIM)).astype(np.float32)
    return np.tanh(normalized @ weights).astype(np.float32)


def _write_variant_npz(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.array([row["sample_id"] for row in rows], dtype=object),
        event_id=np.array([row["event_id"] for row in rows], dtype=object),
        subject_id=np.array([row["subject_id"] for row in rows], dtype=object),
        labels=np.array([json.dumps(_json_ready(row["labels"]), ensure_ascii=False, allow_nan=False) for row in rows], dtype=object),
        face_emb=np.stack([row["face_emb"] for row in rows]).astype(np.float32)
        if rows
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        modality_mask=np.stack([row["modality_mask"] for row in rows]).astype(np.int8)
        if rows
        else np.zeros((0, 4), dtype=np.int8),
        quality_flags=np.array(
            [json.dumps(_json_ready(row["quality_flags"]), ensure_ascii=False, allow_nan=False) for row in rows],
            dtype=object,
        ),
        encoder_version=np.array([row["encoder_version"] for row in rows], dtype=object),
    )
    return out


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _window_labels(window: dict[str, Any]) -> dict[str, Any]:
    values = window.get("label_columns") or window.get("labels") or {}
    return _json_ready(values) if isinstance(values, dict) else _json_ready(_parse_json_object(values))


def _base_quality(variant: str) -> dict[str, Any]:
    return {"variant": variant}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
