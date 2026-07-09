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


def build_split_manifest(
    cohort: Sequence[str] | np.ndarray,
    metadata: Sequence[WindowMetadata],
    *,
    split_seed: int = 17,
) -> dict:
    ordered_sample_ids = np.asarray(cohort).astype(str).tolist()
    row_by_sample = {row.sample_id: row for row in metadata}
    missing = [sample_id for sample_id in ordered_sample_ids if sample_id not in row_by_sample]
    if missing:
        raise ValueError(f"metadata missing cohort sample_id values: {missing[:5]}")
    ordered_rows = [row_by_sample[sample_id] for sample_id in ordered_sample_ids]
    subjects: dict[str, list[WindowMetadata]] = {}
    for row in ordered_rows:
        subjects.setdefault(row.subject_id, []).append(row)
    subject_order = list(subjects)
    return {
        "schema_version": 1,
        "split_seed": int(split_seed),
        "cohort_count": len(ordered_sample_ids),
        "protocols": {
            "event_grouped_5fold": {
                "subjects": [
                    _event_grouped_subject_manifest(subject, subjects[subject], split_seed)
                    for subject in subject_order
                ]
            },
            "session_held_out": {
                "subjects": [
                    _session_subject_manifest(subject, subjects[subject])
                    for subject in subject_order
                ]
            },
        },
    }


def validate_split_manifest(
    manifest: Mapping[str, object],
    *,
    cohort_hash: str,
    window_index_hash: str,
) -> None:
    if str(manifest.get("cohort_sha256")) != str(cohort_hash):
        raise ValueError("cohort hash mismatch")
    if str(manifest.get("window_index_sha256")) != str(window_index_hash):
        raise ValueError("window index hash mismatch")
    protocols = manifest.get("protocols")
    if not isinstance(protocols, Mapping):
        raise ValueError("split manifest missing protocols")
    for required in ("event_grouped_5fold", "session_held_out"):
        if required not in protocols:
            raise ValueError(f"split manifest missing protocol {required}")


def _event_key(row: WindowMetadata) -> EventKey:
    return (str(row.subject_id), str(row.session_id), str(row.event_id))


def _event_grouped_subject_manifest(
    subject_id: str,
    rows: Sequence[WindowMetadata],
    split_seed: int,
) -> dict:
    component_by_event, _overlaps = build_overlap_components(rows)
    rows_by_component: dict[str, list[WindowMetadata]] = {}
    for row in rows:
        rows_by_component.setdefault(component_by_event[_event_key(row)], []).append(row)
    base = _subject_base(subject_id, rows)
    if len(rows_by_component) < 5:
        return {
            **base,
            "status": "insufficient_split_units",
            "split_unit_count": len(rows_by_component),
            "folds": [],
        }
    components = _balanced_components(rows_by_component, split_seed)
    buckets: list[list[str]] = [[] for _ in range(5)]
    bucket_sizes = [0] * 5
    for component_id in components:
        bucket_index = min(range(5), key=lambda idx: (bucket_sizes[idx], idx))
        buckets[bucket_index].append(component_id)
        bucket_sizes[bucket_index] += len(rows_by_component[component_id])
    folds = []
    for fold_index in range(5):
        test_components = set(buckets[fold_index])
        val_components = set(buckets[(fold_index + 1) % 5])
        train_components = set(rows_by_component) - test_components - val_components
        folds.append(
            _fold_manifest(
                fold_index=fold_index,
                train_rows=_rows_for_components(rows_by_component, train_components),
                val_rows=_rows_for_components(rows_by_component, val_components),
                test_rows=_rows_for_components(rows_by_component, test_components),
                train_split_unit_ids=sorted(train_components),
                val_split_unit_ids=sorted(val_components),
                test_split_unit_ids=sorted(test_components),
            )
        )
    return {
        **base,
        "status": "eligible",
        "split_unit_count": len(rows_by_component),
        "folds": folds,
    }


def _session_subject_manifest(subject_id: str, rows: Sequence[WindowMetadata]) -> dict:
    base = _subject_base(subject_id, rows)
    sessions = _ordered_unique(row.session_id for row in rows)
    if len(sessions) < 3:
        return {
            **base,
            "status": "insufficient_sessions",
            "session_ids": sessions,
            "folds": [],
        }
    folds = []
    for fold_index, test_session in enumerate(sessions):
        val_session = sessions[(fold_index + 1) % len(sessions)]
        train_sessions = set(sessions) - {test_session, val_session}
        folds.append(
            _fold_manifest(
                fold_index=fold_index,
                train_rows=[row for row in rows if row.session_id in train_sessions],
                val_rows=[row for row in rows if row.session_id == val_session],
                test_rows=[row for row in rows if row.session_id == test_session],
                train_split_unit_ids=sorted(train_sessions),
                val_split_unit_ids=[val_session],
                test_split_unit_ids=[test_session],
            )
        )
    return {
        **base,
        "status": "eligible",
        "session_ids": sessions,
        "folds": folds,
    }


def _subject_base(subject_id: str, rows: Sequence[WindowMetadata]) -> dict:
    return {
        "subject_id": subject_id,
        "sample_count": len(rows),
        "event_count": len({_event_key(row) for row in rows}),
        "session_ids": _ordered_unique(row.session_id for row in rows),
    }


def _balanced_components(
    rows_by_component: Mapping[str, Sequence[WindowMetadata]],
    split_seed: int,
) -> list[str]:
    rng = np.random.default_rng(int(split_seed))
    by_size: dict[int, list[str]] = {}
    for component_id, rows in rows_by_component.items():
        by_size.setdefault(len(rows), []).append(component_id)
    ordered: list[str] = []
    for size in sorted(by_size, reverse=True):
        values = sorted(by_size[size])
        rng.shuffle(values)
        ordered.extend(values)
    return ordered


def _rows_for_components(
    rows_by_component: Mapping[str, Sequence[WindowMetadata]],
    component_ids: set[str],
) -> list[WindowMetadata]:
    rows = [
        row
        for component_id in sorted(component_ids)
        for row in rows_by_component[component_id]
    ]
    return sorted(rows, key=lambda row: row.sample_id)


def _fold_manifest(
    *,
    fold_index: int,
    train_rows: Sequence[WindowMetadata],
    val_rows: Sequence[WindowMetadata],
    test_rows: Sequence[WindowMetadata],
    train_split_unit_ids: Sequence[str],
    val_split_unit_ids: Sequence[str],
    test_split_unit_ids: Sequence[str],
) -> dict:
    return {
        "fold_id": f"fold-{fold_index:02d}",
        "train_sample_ids": _sample_ids(train_rows),
        "val_sample_ids": _sample_ids(val_rows),
        "test_sample_ids": _sample_ids(test_rows),
        "train_event_keys": _event_keys(train_rows),
        "val_event_keys": _event_keys(val_rows),
        "test_event_keys": _event_keys(test_rows),
        "train_split_unit_ids": list(train_split_unit_ids),
        "val_split_unit_ids": list(val_split_unit_ids),
        "test_split_unit_ids": list(test_split_unit_ids),
        "train_session_ids": _ordered_unique(row.session_id for row in train_rows),
        "val_session_ids": _ordered_unique(row.session_id for row in val_rows),
        "test_session_ids": _ordered_unique(row.session_id for row in test_rows),
        "train_window_count": len(train_rows),
        "val_window_count": len(val_rows),
        "test_window_count": len(test_rows),
        "train_event_count": len(_event_keys(train_rows)),
        "val_event_count": len(_event_keys(val_rows)),
        "test_event_count": len(_event_keys(test_rows)),
        "cross_partition_time_overlap_count": _cross_partition_overlap_count(
            train_rows,
            val_rows,
            test_rows,
        ),
    }


def _sample_ids(rows: Sequence[WindowMetadata]) -> list[str]:
    return [row.sample_id for row in sorted(rows, key=lambda row: row.sample_id)]


def _event_keys(rows: Sequence[WindowMetadata]) -> list[list[str]]:
    return [
        list(key)
        for key in sorted({_event_key(row) for row in rows})
    ]


def _ordered_unique(values: Sequence[str] | object) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _cross_partition_overlap_count(*partitions: Sequence[WindowMetadata]) -> int:
    labeled_rows = [
        (partition_index, row)
        for partition_index, rows in enumerate(partitions)
        for row in rows
    ]
    count = 0
    for left_index, (left_partition, left) in enumerate(labeled_rows):
        for right_partition, right in labeled_rows[left_index + 1 :]:
            if left_partition == right_partition:
                continue
            if (left.subject_id, left.session_id) != (right.subject_id, right.session_id):
                continue
            if max(left.start, right.start) < min(left.end, right.end):
                count += 1
    return count


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
