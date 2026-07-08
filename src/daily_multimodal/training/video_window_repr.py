from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.embeddings.contracts import validate_embedding_shape
from daily_multimodal.training.video_event_aggregation import _project_to_256


FACE_MASK_INDEX = 2


def build_window_repr_embedding_bundle(
    *,
    representations: Path | str,
    variant: str,
    out: Path | str,
    target_label: str = "fatigue",
) -> dict[str, Any]:
    data = _load_window_repr(representations, variant=variant)
    embeddings = np.vstack(
        [_project_to_256(row, salt=f"video_window_{variant}_adapter_repr") for row in data["representation"]]
    ).astype(np.float32)
    validate_embedding_shape("face_emb", embeddings)
    row_count = int(len(data["sample_id"]))
    modality_mask = np.zeros((row_count, 4), dtype=np.int8)
    modality_mask[:, FACE_MASK_INDEX] = 1
    labels = np.asarray([json.dumps({target_label: float(value)}, ensure_ascii=False) for value in data["target"]], dtype=object)
    quality_flags = np.asarray(
        [
            json.dumps(
                {
                    "source_variant": variant,
                    "representation_source": "window_level_adapter_repr",
                    "input_dim": int(data["representation"].shape[1]),
                },
                ensure_ascii=False,
            )
            for _ in range(row_count)
        ],
        dtype=object,
    )
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_id=data["sample_id"].astype(object),
        subject_id=data["subject_id"].astype(object),
        event_id=data["event_id"].astype(object),
        labels=labels,
        face_emb=embeddings,
        modality_mask=modality_mask,
        quality_flags=quality_flags,
        source_paths=np.asarray(
            [json.dumps({"representation_source": str(representations)}, ensure_ascii=False) for _ in range(row_count)],
            dtype=object,
        ),
        encoder_version=np.asarray([f"video_window_{variant}_adapter_repr_projected_256"] * row_count, dtype=object),
    )
    return {
        "representations": str(representations),
        "variant": variant,
        "output": str(output),
        "row_count": row_count,
        "event_count": int(len(set(data["event_id"].astype(str).tolist()))),
        "subject_count": int(len(set(data["subject_id"].astype(str).tolist()))),
        "input_dim": int(data["representation"].shape[1]),
    }


def _load_window_repr(path: Path | str, *, variant: str) -> dict[str, Any]:
    key = f"repr__{variant}"
    path = Path(path)
    with np.load(path, allow_pickle=True) as loaded:
        required = {"sample_id", "subject_id", "event_id", "target", key}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing required arrays: {', '.join(missing)}")
        representation = np.asarray(loaded[key], dtype=np.float32)
        if representation.ndim != 2:
            raise ValueError(f"{key} expected shape (N, D), got {representation.shape}")
        if not np.isfinite(representation).all():
            raise ValueError(f"{key} contains non-finite values")
        row_count = representation.shape[0]
        data = {
            "sample_id": loaded["sample_id"].astype(str),
            "subject_id": loaded["subject_id"].astype(str),
            "event_id": loaded["event_id"].astype(str),
            "target": np.asarray(loaded["target"], dtype=np.float32),
            "representation": representation,
        }
    for name, value in data.items():
        if isinstance(value, np.ndarray) and len(value) != row_count:
            raise ValueError(f"{path} array {name} row count {len(value)} != representation rows {row_count}")
    return data
