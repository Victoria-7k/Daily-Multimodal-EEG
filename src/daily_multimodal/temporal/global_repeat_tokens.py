from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.temporal.token_contract import EMBEDDING_DIM, MODALITY_ORDER, TOKEN_KEYS


DEFAULT_EMB_KEYS = {
    "eeg": ("eeg_emb",),
    "wear": ("wear_emb",),
    "video": ("video_emb", "face_emb"),
    "audio": ("audio_emb",),
}


def build_global_repeat_temporal_tokens(
    *,
    temporal_index_rows: list[dict[str, Any]],
    source_npz: Path | str,
    output_npz: Path | str,
    modality: str,
    emb_key: str | None = None,
    mask_key: str | None = None,
    encoder_version: str = "global_repeat_smoke_v1",
) -> dict[str, Any]:
    """Repeat window-level 256D embeddings into 5 temporal slots for pipeline smoke tests."""

    if modality not in MODALITY_ORDER:
        raise ValueError(f"unsupported modality {modality!r}; expected one of {MODALITY_ORDER}")
    if not temporal_index_rows:
        raise ValueError("temporal_index_rows must not be empty")
    expected_sample_id = np.asarray([str(row["sample_id"]) for row in temporal_index_rows], dtype=str)
    token_start = _shared_time_boundaries(temporal_index_rows, "token_start_seconds")
    token_end = _shared_time_boundaries(temporal_index_rows, "token_end_seconds")
    with np.load(source_npz, allow_pickle=True) as loaded:
        if "sample_id" not in loaded.files:
            raise ValueError(f"{source_npz} missing sample_id")
        loaded_sample_id = loaded["sample_id"].astype(str)
        if not np.array_equal(loaded_sample_id, expected_sample_id):
            raise ValueError(f"{source_npz} sample_id order does not match temporal index")
        resolved_emb_key = emb_key or _first_present_key(loaded.files, DEFAULT_EMB_KEYS[modality])
        embedding = loaded[resolved_emb_key].astype(np.float32)
        if embedding.shape != (len(expected_sample_id), EMBEDDING_DIM):
            raise ValueError(f"{resolved_emb_key} expected shape {(len(expected_sample_id), EMBEDDING_DIM)}, got {embedding.shape}")
        if np.isnan(embedding).any() or not np.isfinite(embedding).all():
            raise ValueError(f"{resolved_emb_key} contains NaN or infinite values")
        mask = _load_mask(loaded, modality=modality, mask_key=mask_key, row_count=len(expected_sample_id))
    token_count = len(token_start)
    tokens = np.repeat(embedding[:, None, :], token_count, axis=1).astype(np.float32)
    token_mask = np.repeat(mask[:, None], token_count, axis=1).astype(np.int8)
    quality = token_mask[:, :, None].astype(np.float32)
    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=expected_sample_id,
        event_id=np.asarray([str(row.get("event_id", "")) for row in temporal_index_rows], dtype=str),
        subject_id=np.asarray([str(row.get("subject_id", "")) for row in temporal_index_rows], dtype=str),
        token_start_seconds=np.asarray(token_start, dtype=np.float32),
        token_end_seconds=np.asarray(token_end, dtype=np.float32),
        **{
            TOKEN_KEYS[modality]: tokens,
            f"{modality}_token_mask": token_mask,
            f"{modality}_quality_features": quality,
            f"{modality}_quality_feature_names_json": np.asarray(
                [json.dumps(["available_from_window_mask"], ensure_ascii=False)],
                dtype=object,
            ),
            "encoder_versions_json": np.asarray(
                [json.dumps({modality: encoder_version}, ensure_ascii=False)],
                dtype=object,
            ),
            "source_paths_json": np.asarray(
                [json.dumps({modality: str(source_npz)}, ensure_ascii=False)],
                dtype=object,
            ),
        },
    )
    return {
        "path": str(out),
        "modality": modality,
        "row_count": int(len(expected_sample_id)),
        "token_shape": list(tokens.shape),
        "mask_available_ratio": float(token_mask.mean()) if token_mask.size else 0.0,
        "source_npz": str(source_npz),
        "source_embedding_key": resolved_emb_key,
        "source_mask_key": mask_key or "auto",
        "encoder_version": encoder_version,
        "interpretation": "pipeline_smoke_only_not_lag_evidence",
    }


def _first_present_key(files: list[str], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in files:
            return key
    raise ValueError(f"source NPZ missing any embedding key from {candidates}")


def _load_mask(loaded: Any, *, modality: str, mask_key: str | None, row_count: int) -> np.ndarray:
    if mask_key:
        if mask_key not in loaded.files:
            raise ValueError(f"source NPZ missing requested mask key {mask_key}")
        mask = loaded[mask_key]
    elif f"{modality}_mask" in loaded.files:
        mask = loaded[f"{modality}_mask"]
    elif "modality_mask" in loaded.files:
        mask = loaded["modality_mask"][:, MODALITY_ORDER.index(modality)]
    else:
        mask = np.ones((row_count,), dtype=np.int8)
    value = np.asarray(mask)
    if value.shape != (row_count,):
        raise ValueError(f"mask expected shape {(row_count,)}, got {value.shape}")
    return value.astype(bool, copy=False)


def _shared_time_boundaries(rows: list[dict[str, Any]], key: str) -> list[float]:
    first = [float(value) for value in rows[0][key]]
    for row in rows[1:]:
        values = [float(value) for value in row[key]]
        if values != first:
            raise ValueError(f"{key} differs across temporal index rows")
    return first
