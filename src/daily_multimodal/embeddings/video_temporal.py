from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape


FACE_MASK_INDEX = 2
TEMPORAL_ENCODER_VERSIONS = {
    "tcn": "video_v4b_tcn_dinov2_2xroi",
    "temporal_transformer": "video_v4b_temporal_transformer_dinov2_2xroi",
}


def build_video_temporal_embeddings(
    *,
    frame_sequences: Path | str,
    out_path: Path | str,
    temporal_encoder: str,
    seed: int = 41,
) -> dict[str, Any]:
    if temporal_encoder not in TEMPORAL_ENCODER_VERSIONS:
        raise ValueError(f"unsupported temporal_encoder: {temporal_encoder}")
    version = TEMPORAL_ENCODER_VERSIONS[temporal_encoder]
    bundle = _load_frame_sequence_bundle(Path(frame_sequences))
    rows = []
    for index in range(bundle["frame_embeddings"].shape[0]):
        sequence = bundle["frame_embeddings"][index]
        rows.append(
            _temporal_row(
                sample_id=bundle["sample_id"][index],
                event_id=bundle["event_id"][index],
                subject_id=bundle["subject_id"][index],
                labels=bundle["labels"][index],
                sequence=sequence,
                temporal_encoder=temporal_encoder,
                encoder_version=version,
                source_encoder_version=bundle["source_encoder_version"][index],
                source_mask_value=int(bundle["source_mask_value"][index]),
                seed=seed,
            )
        )
    _write_temporal_npz(rows, out_path)
    return {
        "temporal_encoder": temporal_encoder,
        "encoder_version": version,
        "row_count": int(len(rows)),
        "mask_sum": int(sum(int(row["modality_mask"][FACE_MASK_INDEX]) for row in rows)),
        "out_path": str(out_path),
    }


def _temporal_row(
    *,
    sample_id: str,
    event_id: str,
    subject_id: str,
    labels: str,
    sequence: np.ndarray,
    temporal_encoder: str,
    encoder_version: str,
    source_encoder_version: str,
    source_mask_value: int,
    seed: int,
) -> dict[str, Any]:
    quality = {
        "variant": encoder_version,
        "temporal_encoder": temporal_encoder,
        "input_frame_count": int(sequence.shape[0]),
        "frame_embedding_dim": int(sequence.shape[1]),
        "source_encoder_version": source_encoder_version,
        "source_mask_value": int(source_mask_value),
    }
    if not source_mask_value:
        quality["masked_reason"] = "source_video_mask_zero"
        return _row(
            sample_id=sample_id,
            event_id=event_id,
            subject_id=subject_id,
            labels=labels,
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            mask_value=0,
            quality_flags=quality,
            encoder_version=encoder_version,
        )
    if not np.isfinite(sequence).all():
        quality["masked_reason"] = "nonfinite_frame_embeddings"
        return _row(
            sample_id=sample_id,
            event_id=event_id,
            subject_id=subject_id,
            labels=labels,
            embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            mask_value=0,
            quality_flags=quality,
            encoder_version=encoder_version,
        )

    features = _temporal_features(sequence, temporal_encoder=temporal_encoder)
    quality["projected_from_dim"] = int(features.reshape(-1).shape[0])
    embedding = _project_to_256(features, salt=f"{encoder_version}:{seed}")
    return _row(
        sample_id=sample_id,
        event_id=event_id,
        subject_id=subject_id,
        labels=labels,
        embedding=embedding,
        mask_value=1,
        quality_flags=quality,
        encoder_version=encoder_version,
    )


def _temporal_features(sequence: np.ndarray, *, temporal_encoder: str) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"frame sequence must have shape [frames, dim], got {values.shape}")
    if temporal_encoder == "tcn":
        return _tcn_features(values)
    if temporal_encoder == "temporal_transformer":
        return _temporal_transformer_features(values)
    raise ValueError(f"unsupported temporal_encoder: {temporal_encoder}")


def _tcn_features(sequence: np.ndarray) -> np.ndarray:
    padded = np.pad(sequence, ((1, 1), (0, 0)), mode="edge")
    smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    delta = np.diff(sequence, axis=0)
    if delta.size == 0:
        delta = np.zeros_like(sequence)
    return np.concatenate(
        [
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.max(axis=0),
            sequence[-1] - sequence[0],
            smoothed.mean(axis=0),
            delta.std(axis=0),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _temporal_transformer_features(sequence: np.ndarray) -> np.ndarray:
    positions = np.linspace(-1.0, 1.0, sequence.shape[0], dtype=np.float32).reshape(-1, 1)
    positional = sequence + positions
    centered = positional - positional.mean(axis=0, keepdims=True)
    scores = np.linalg.norm(centered, axis=1) / math.sqrt(float(sequence.shape[1]))
    weights = _softmax(scores)
    attended = weights @ positional
    return np.concatenate(
        [
            attended,
            positional.mean(axis=0),
            positional.std(axis=0),
            positional.max(axis=0),
            positional[-1] - positional[0],
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    shifted = values - float(values.max())
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0:
        return np.full(values.shape, 1.0 / float(max(1, values.size)), dtype=np.float32)
    return (exp / total).astype(np.float32, copy=False)


def _row(
    *,
    sample_id: str,
    event_id: str,
    subject_id: str,
    labels: str,
    embedding: np.ndarray,
    mask_value: int,
    quality_flags: dict[str, Any],
    encoder_version: str,
) -> dict[str, Any]:
    mask = np.zeros(4, dtype=np.int8)
    mask[FACE_MASK_INDEX] = int(mask_value)
    return {
        "sample_id": sample_id,
        "event_id": event_id,
        "subject_id": subject_id,
        "labels": labels,
        "face_emb": validate_embedding_shape("face_emb", np.asarray(embedding, dtype=np.float32)),
        "modality_mask": mask,
        "quality_flags": quality_flags,
        "encoder_version": encoder_version,
    }


def _load_frame_sequence_bundle(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as loaded:
        frame_embeddings = _frame_sequences_from_npz(loaded)
        row_count = int(frame_embeddings.shape[0])
        sample_id = _string_array(loaded, "sample_id", row_count)
        event_id = _string_array(loaded, "event_id", row_count)
        subject_id = _string_array(loaded, "subject_id", row_count)
        labels = _string_array(loaded, "labels", row_count, default="{}")
        source_encoder_version = _string_array(loaded, "encoder_version", row_count, default="")
        source_mask_value = _source_mask_values(loaded, row_count)
    return {
        "sample_id": sample_id,
        "event_id": event_id,
        "subject_id": subject_id,
        "labels": labels,
        "frame_embeddings": frame_embeddings,
        "source_encoder_version": source_encoder_version,
        "source_mask_value": source_mask_value,
    }


def _frame_sequences_from_npz(loaded: Any) -> np.ndarray:
    for key in ("frame_embeddings", "dinov2_frame_embeddings", "frame_emb"):
        if key in loaded.files:
            values = np.asarray(loaded[key], dtype=np.float32)
            break
    else:
        raise ValueError("frame sequence bundle must contain frame_embeddings")
    if values.ndim != 3:
        raise ValueError(f"frame_embeddings expected shape [N, frames, dim], got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError(f"frame_embeddings must be non-empty, got {values.shape}")
    return values.astype(np.float32, copy=False)


def _string_array(loaded: Any, key: str, row_count: int, *, default: str | None = None) -> np.ndarray:
    if key in loaded.files:
        values = loaded[key].astype(str)
    elif default is not None:
        values = np.asarray([default] * row_count, dtype=object).astype(str)
    else:
        values = np.asarray([""] * row_count, dtype=object).astype(str)
    if values.shape[0] != row_count:
        raise ValueError(f"{key} row count {values.shape[0]} does not match frame_embeddings row count {row_count}")
    return values


def _source_mask_values(loaded: Any, row_count: int) -> np.ndarray:
    if "modality_mask" not in loaded.files:
        return np.ones(row_count, dtype=np.int8)
    values = np.asarray(loaded["modality_mask"], dtype=np.int8)
    if values.shape != (row_count, 4):
        raise ValueError(f"modality_mask expected shape ({row_count}, 4), got {values.shape}")
    return values[:, FACE_MASK_INDEX].astype(np.int8, copy=False)


def _project_to_256(vector: np.ndarray, *, salt: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    std = float(values.std())
    normalized = values.copy()
    if std > 0:
        normalized = (normalized - float(values.mean())) / std
    digest = hashlib.sha256(f"{salt}:{values.size}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))
    weights = rng.normal(
        0.0,
        1.0 / math.sqrt(float(max(1, values.size))),
        size=(values.size, EMBEDDING_DIM),
    ).astype(np.float32)
    projected = np.tanh(normalized @ weights)
    return validate_embedding_shape("face_emb", projected.astype(np.float32))


def _write_temporal_npz(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.asarray([row["sample_id"] for row in rows], dtype=object),
        event_id=np.asarray([row["event_id"] for row in rows], dtype=object),
        subject_id=np.asarray([row["subject_id"] for row in rows], dtype=object),
        labels=np.asarray([row["labels"] for row in rows], dtype=object),
        face_emb=np.stack([row["face_emb"] for row in rows]).astype(np.float32)
        if rows
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        modality_mask=np.stack([row["modality_mask"] for row in rows]).astype(np.int8)
        if rows
        else np.zeros((0, 4), dtype=np.int8),
        quality_flags=np.asarray(
            [json.dumps(_json_ready(row["quality_flags"]), ensure_ascii=False, allow_nan=False) for row in rows],
            dtype=object,
        ),
        encoder_version=np.asarray([row["encoder_version"] for row in rows], dtype=object),
    )
    return out


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
