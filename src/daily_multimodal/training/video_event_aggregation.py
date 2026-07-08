from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape


FACE_MASK_INDEX = 2
DEFAULT_AGGREGATIONS = ("E1_mean", "E2_mean_std", "E3_mean_std_max")


@dataclass(frozen=True)
class EventAggregationSpec:
    name: str
    stats: tuple[str, ...]


AGGREGATION_SPECS = {
    "E1_mean": EventAggregationSpec("E1_mean", ("mean",)),
    "E2_mean_std": EventAggregationSpec("E2_mean_std", ("mean", "std")),
    "E3_mean_std_max": EventAggregationSpec("E3_mean_std_max", ("mean", "std", "max")),
}


def build_event_embedding_bundles(
    *,
    representations: Path | str,
    variant: str,
    out_dir: Path | str,
    min_windows: int = 8,
    target_label: str = "fatigue",
    aggregations: Iterable[str] = DEFAULT_AGGREGATIONS,
) -> dict[str, Any]:
    data = load_representation_bundle(representations, variant=variant)
    specs = [AGGREGATION_SPECS[name] for name in aggregations]
    events, dropped_events = _group_events(data, min_windows=min_windows)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for spec in specs:
        path = output_dir / f"{spec.name}_embeddings.npz"
        _write_event_bundle(
            path,
            data=data,
            events=events,
            spec=spec,
            variant=variant,
            target_label=target_label,
        )
        outputs[spec.name] = str(path)

    result = {
        "representations": str(representations),
        "source_variant": variant,
        "target_label": target_label,
        "min_windows": int(min_windows),
        "input_window_count": int(len(data["sample_id"])),
        "event_count": int(len(events)),
        "dropped_event_count": int(len(dropped_events)),
        "dropped_events": dropped_events,
        "outputs": outputs,
    }
    _write_summary(result, output_dir / "event_aggregation_summary.json", output_dir / "event_aggregation_summary.md")
    return result


def load_representation_bundle(path: Path | str, *, variant: str) -> dict[str, Any]:
    key = f"repr__{variant}"
    path = Path(path)
    with np.load(path, allow_pickle=True) as loaded:
        required = {"sample_id", "subject_id", "event_id", "session_id", "target", key}
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
            "session_id": loaded["session_id"].astype(str),
            "target": np.asarray(loaded["target"], dtype=np.float32),
            "representation": representation,
        }
    for name, value in data.items():
        if isinstance(value, np.ndarray) and len(value) != row_count:
            raise ValueError(f"{path} array {name} row count {len(value)} != representation rows {row_count}")
    return data


def _group_events(data: dict[str, Any], *, min_windows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_ids = data["event_id"].astype(str)
    events: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for event_id in dict.fromkeys(event_ids.tolist()):
        indices = np.flatnonzero(event_ids == event_id)
        subject_values = list(dict.fromkeys(data["subject_id"][indices].astype(str).tolist()))
        session_values = list(dict.fromkeys(data["session_id"][indices].astype(str).tolist()))
        target_values = data["target"][indices].astype(float)
        row = {
            "event_id": str(event_id),
            "subject_id": subject_values[0] if subject_values else "",
            "session_id": session_values[0] if session_values else "",
            "window_count": int(len(indices)),
            "target": float(target_values[0]) if len(target_values) else math.nan,
            "target_std": float(np.std(target_values)) if len(target_values) else math.nan,
            "indices": indices,
        }
        if len(indices) < int(min_windows):
            dropped.append({key: value for key, value in row.items() if key != "indices"})
        else:
            events.append(row)
    return events, dropped


def _write_event_bundle(
    path: Path,
    *,
    data: dict[str, Any],
    events: list[dict[str, Any]],
    spec: EventAggregationSpec,
    variant: str,
    target_label: str,
) -> None:
    sample_ids = []
    event_ids = []
    subject_ids = []
    labels = []
    quality_flags = []
    source_paths = []
    encoder_versions = []
    face_emb = []
    for event in events:
        event_id = str(event["event_id"])
        indices = event["indices"]
        values = data["representation"][indices]
        vector = _event_stat_vector(values, spec.stats)
        projected = _project_to_256(vector, salt=f"video_event_{variant}_{spec.name}")
        face_emb.append(projected)
        sample_ids.append(event_id)
        event_ids.append(event_id)
        subject_ids.append(str(event["subject_id"]))
        labels.append(json.dumps({target_label: float(event["target"])}, ensure_ascii=False))
        quality_flags.append(
            json.dumps(
                {
                    "source_variant": variant,
                    "aggregation": spec.name,
                    "stats": list(spec.stats),
                    "window_count": int(event["window_count"]),
                    "target_std_within_event": float(event["target_std"]),
                    "source_sample_ids": data["sample_id"][indices].astype(str).tolist(),
                },
                ensure_ascii=False,
            )
        )
        source_paths.append(json.dumps({"representation_source": variant, "event_id": event_id}, ensure_ascii=False))
        encoder_versions.append(f"video_event_{variant}_{spec.name}_projected_256")

    row_count = len(events)
    modality_mask = np.zeros((row_count, 4), dtype=np.int8)
    modality_mask[:, FACE_MASK_INDEX] = 1
    face_array = validate_embedding_shape("face_emb", np.vstack(face_emb).astype(np.float32)) if face_emb else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        event_id=np.asarray(event_ids, dtype=object),
        subject_id=np.asarray(subject_ids, dtype=object),
        labels=np.asarray(labels, dtype=object),
        face_emb=face_array,
        modality_mask=modality_mask,
        quality_flags=np.asarray(quality_flags, dtype=object),
        source_paths=np.asarray(source_paths, dtype=object),
        encoder_version=np.asarray(encoder_versions, dtype=object),
    )


def _event_stat_vector(values: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    parts = []
    for stat in stats:
        if stat == "mean":
            parts.append(values.mean(axis=0))
        elif stat == "std":
            parts.append(values.std(axis=0))
        elif stat == "max":
            parts.append(values.max(axis=0))
        else:  # pragma: no cover - guarded by spec constants.
            raise ValueError(f"unsupported event aggregation stat: {stat}")
    return np.concatenate(parts).astype(np.float32)


def _project_to_256(vector: np.ndarray, *, salt: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size == 0:
        raise ValueError("cannot project empty event vector")
    mean = float(value.mean())
    std = float(value.std())
    normalized = (value - mean) / (std if std > 1e-6 else 1.0)
    seed = int.from_bytes(hashlib.sha256(salt.encode("utf-8")).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0 / math.sqrt(float(max(1, normalized.size))), size=(normalized.size, EMBEDDING_DIM))
    projected = normalized @ weights
    norm = float(np.linalg.norm(projected))
    if norm > 0:
        projected = projected / norm
    return validate_embedding_shape("face_emb", projected.astype(np.float32))


def _write_summary(result: dict[str, Any], out_json: Path, out_table: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows = [
        "# Video Event Aggregation",
        "",
        f"- representations: `{result['representations']}`",
        f"- source variant: `{result['source_variant']}`",
        f"- input windows: `{result['input_window_count']}`",
        f"- retained events: `{result['event_count']}`",
        f"- dropped events: `{result['dropped_event_count']}`",
        "",
        "| aggregation | output |",
        "| --- | --- |",
    ]
    for name, path in result["outputs"].items():
        rows.append(f"| {name} | `{path}` |")
    if result["dropped_events"]:
        rows.extend(["", "## Dropped Events", "", "| event_id | subject_id | session_id | windows |", "| --- | --- | --- | ---: |"])
        for event in result["dropped_events"]:
            rows.append(
                "| {event_id} | {subject_id} | {session_id} | {window_count} |".format(
                    event_id=event["event_id"],
                    subject_id=event["subject_id"],
                    session_id=event["session_id"],
                    window_count=event["window_count"],
                )
            )
    out_table.write_text("\n".join(rows) + "\n", encoding="utf-8")
