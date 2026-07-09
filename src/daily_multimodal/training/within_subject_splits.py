from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


EventKey = tuple[str, str, str]


@dataclass(frozen=True)
class WindowMetadata:
    sample_id: str
    event_id: str
    subject_id: str
    session_id: str
    start: datetime
    end: datetime


def build_global_paired_cohort(
    sample_ids_by_experiment: Mapping[str, np.ndarray],
    *,
    reference_order: np.ndarray,
) -> np.ndarray:
    if len(sample_ids_by_experiment) != 12:
        raise ValueError("paired fusion cohort requires exactly 12 experiments")
    common = set(np.asarray(reference_order).astype(str).tolist())
    for name, values in sample_ids_by_experiment.items():
        ids = np.asarray(values).astype(str)
        if len(ids) != len(set(ids.tolist())):
            raise ValueError(f"{name} contains duplicate sample_id values")
        common &= set(ids.tolist())
    ordered = [
        value
        for value in np.asarray(reference_order).astype(str).tolist()
        if value in common
    ]
    if not ordered:
        raise ValueError("global paired cohort is empty")
    return np.asarray(ordered, dtype=str)


def load_window_metadata(
    path: Path | str,
    required_sample_ids: Sequence[str] | np.ndarray,
) -> list[WindowMetadata]:
    required = np.asarray(required_sample_ids).astype(str)
    required_set = set(required.tolist())
    rows_by_sample: dict[str, WindowMetadata] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            sample_id = str(raw.get("sample_id", ""))
            if sample_id not in required_set:
                continue
            if sample_id in rows_by_sample:
                raise ValueError(f"{path} contains duplicate sample_id {sample_id!r}")
            rows_by_sample[sample_id] = WindowMetadata(
                sample_id=sample_id,
                event_id=_required_str(raw, "event_id", path, line_number),
                subject_id=_required_str(raw, "subject_id", path, line_number),
                session_id=_required_str(raw, "session_id", path, line_number),
                start=_parse_datetime(
                    _required_str(raw, "window_start_time", path, line_number),
                    path,
                    line_number,
                    "window_start_time",
                ),
                end=_parse_datetime(
                    _required_str(raw, "window_end_time", path, line_number),
                    path,
                    line_number,
                    "window_end_time",
                ),
            )
    missing = [sample_id for sample_id in required.tolist() if sample_id not in rows_by_sample]
    if missing:
        raise ValueError(f"{path} missing required sample_id values: {missing[:5]}")
    return [rows_by_sample[sample_id] for sample_id in required.tolist()]


def build_overlap_components(
    rows: Sequence[WindowMetadata],
) -> tuple[dict[EventKey, str], list[dict]]:
    event_keys = sorted({_event_key(row) for row in rows})
    parent: dict[EventKey, EventKey] = {key: key for key in event_keys}

    def find(key: EventKey) -> EventKey:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: EventKey, right: EventKey) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root

    overlaps: list[dict] = []
    by_subject_session: dict[tuple[str, str], list[WindowMetadata]] = {}
    for row in rows:
        by_subject_session.setdefault((row.subject_id, row.session_id), []).append(row)

    for group_rows in by_subject_session.values():
        sorted_rows = sorted(group_rows, key=lambda row: (row.start, row.end, row.event_id, row.sample_id))
        for left_index, left in enumerate(sorted_rows):
            for right in sorted_rows[left_index + 1 :]:
                if right.start >= left.end:
                    break
                overlap_seconds = (min(left.end, right.end) - max(left.start, right.start)).total_seconds()
                if overlap_seconds <= 0:
                    continue
                left_key = _event_key(left)
                right_key = _event_key(right)
                if left_key == right_key:
                    continue
                union(left_key, right_key)
                event_a, event_b = sorted([left.event_id, right.event_id])
                overlaps.append(
                    {
                        "subject_id": left.subject_id,
                        "session_id": left.session_id,
                        "event_a": event_a,
                        "event_b": event_b,
                        "overlap_seconds": float(overlap_seconds),
                    }
                )

    component_roots = {key: find(key) for key in event_keys}
    root_names = {
        root: f"{root[0]}|{root[1]}|component-{index:04d}"
        for index, root in enumerate(sorted(set(component_roots.values())))
    }
    component_by_event = {
        key: root_names[root]
        for key, root in component_roots.items()
    }
    overlaps.sort(
        key=lambda item: (
            item["subject_id"],
            item["session_id"],
            item["event_a"],
            item["event_b"],
        )
    )
    return component_by_event, overlaps


def write_cohort_manifest(
    *,
    cohort: Sequence[str] | np.ndarray,
    native_counts: Mapping[str, int],
    source_hashes: Mapping[str, str],
) -> dict:
    sample_ids = np.asarray(cohort).astype(str).tolist()
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("cohort contains duplicate sample_id values")
    return {
        "schema_version": 1,
        "cohort_count": len(sample_ids),
        "sample_ids": sample_ids,
        "sample_id_sha256": _sha256_lines(sample_ids),
        "native_counts": {str(key): int(value) for key, value in sorted(native_counts.items())},
        "source_hashes": {str(key): str(value) for key, value in sorted(source_hashes.items())},
    }


def _event_key(row: WindowMetadata) -> EventKey:
    return (str(row.subject_id), str(row.session_id), str(row.event_id))


def _required_str(raw: Mapping[str, object], key: str, path: Path | str, line_number: int) -> str:
    if key not in raw or raw[key] is None or str(raw[key]) == "":
        raise ValueError(f"{path}:{line_number} missing required field {key}")
    return str(raw[key])


def _parse_datetime(value: str, path: Path | str, line_number: int, key: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} invalid {key}: {value!r}") from exc


def _sha256_lines(values: Sequence[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
