#!/usr/bin/env python3
"""Audit the fixed EEGPT EEG-aligned split protocols without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROTOCOLS = ("cross_subject", "cross_day", "within_subject_day")
SPLIT_NAMES = ("pretrain", "finetune", "train", "val", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-index", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_jsonl(args.window_index)
    if not rows:
        raise ValueError("window index is empty")
    audit = {
        "stage": 1,
        "window_index": str(args.window_index),
        "row_count": len(rows),
        "protocols": {},
    }
    for protocol in args.protocols:
        audit["protocols"][protocol] = _audit_protocol(
            rows,
            args.splits_root / protocol,
        )
    _write_json(audit, args.out_json)
    _write_markdown(audit, args.out_md)
    print(f"protocols={','.join(args.protocols)}")
    print(f"row_count={len(rows)}")
    print(f"out_json={args.out_json}")
    print(f"out_md={args.out_md}")
    return 0


def _audit_protocol(rows: list[dict[str, Any]], protocol_root: Path) -> dict[str, Any]:
    subjects = np.asarray([_norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    days = np.asarray([_norm_day(row.get("day_id", row.get("day"))) for row in rows], dtype=str)
    events = np.asarray([str(row.get("event_id", row.get("event_key", ""))) for row in rows], dtype=str)
    split = {
        name: _load_indices(protocol_root / f"{name}.json", n_rows=len(rows))
        for name in ("pretrain", "finetune", "val", "test")
    }
    split["train"] = np.asarray(
        split["pretrain"].tolist() + split["finetune"].tolist(),
        dtype=np.int64,
    )
    split_report: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        indices = split[name]
        pair_values = sorted(
            {_subject_day(subjects[index], days[index]) for index in indices.tolist()},
            key=str,
        )
        split_report[name] = {
            "window_count": int(len(indices)),
            "event_count": int(len(set(events[indices].tolist()))),
            "subject_count": int(len(set(subjects[indices].tolist()))),
            "day_count": int(len(set(days[indices].tolist()))),
            "subject_day_count": int(len(pair_values)),
            "subject_ids": sorted(set(subjects[indices].tolist()), key=str),
            "day_ids": sorted(set(days[indices].tolist()), key=str),
            "subject_day_pairs": pair_values,
            "index_min": None if len(indices) == 0 else int(indices.min()),
            "index_max": None if len(indices) == 0 else int(indices.max()),
        }

    train = split["train"]
    val = split["val"]
    test = split["test"]
    train_subjects = set(subjects[train].tolist())
    val_subjects = set(subjects[val].tolist())
    test_subjects = set(subjects[test].tolist())
    train_days = set(days[train].tolist())
    val_days = set(days[val].tolist())
    test_days = set(days[test].tolist())
    train_pairs = {_subject_day(subjects[i], days[i]) for i in train.tolist()}
    val_pairs = {_subject_day(subjects[i], days[i]) for i in val.tolist()}
    test_pairs = {_subject_day(subjects[i], days[i]) for i in test.tolist()}
    return {
        "protocol_root": str(protocol_root),
        "split": split_report,
        "overlap": {
            "index_train_val": _sorted_ints(set(train.tolist()) & set(val.tolist())),
            "index_train_test": _sorted_ints(set(train.tolist()) & set(test.tolist())),
            "index_val_test": _sorted_ints(set(val.tolist()) & set(test.tolist())),
            "subject_overlap_train_val": sorted(train_subjects & val_subjects, key=str),
            "subject_overlap_train_test": sorted(train_subjects & test_subjects, key=str),
            "day_overlap_train_val": sorted(train_days & val_days, key=str),
            "day_overlap_train_test": sorted(train_days & test_days, key=str),
            "subject_day_overlap_train_val": sorted(train_pairs & val_pairs, key=str),
            "subject_day_overlap_train_test": sorted(train_pairs & test_pairs, key=str),
            "subject_day_overlap_val_test": sorted(val_pairs & test_pairs, key=str),
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"window index row is not an object: {path}")
                rows.append(value)
    return rows


def _load_indices(path: Path, *, n_rows: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("indices", "index", "rows"):
            if key in value:
                value = value[key]
                break
    indices = np.asarray(value, dtype=np.int64).reshape(-1)
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_rows):
        raise ValueError(f"{path} contains out-of-range indices for {n_rows} rows")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(f"{path} contains duplicate indices")
    return indices


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def _norm_day(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _subject_day(subject: str, day: str) -> str:
    return f"{subject}::{day}"


def _sorted_ints(values: set[int]) -> list[int]:
    return sorted(int(value) for value in values)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# EEGPT Split Audit",
        "",
        f"window_index: `{audit['window_index']}`",
        f"row_count: `{audit['row_count']}`",
        "",
    ]
    for protocol, report in audit["protocols"].items():
        lines.extend([f"## `{protocol}`", "", "| split | windows | events | subjects | days | subject-day pairs |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for split_name in SPLIT_NAMES:
            row = report["split"][split_name]
            lines.append(
                f"| {split_name} | {row['window_count']} | {row['event_count']} | {row['subject_count']} | {row['day_count']} | {row['subject_day_count']} |"
            )
        lines.extend([
            "",
            "| overlap check | count | examples |",
            "| --- | ---: | --- |",
        ])
        for key, values in report["overlap"].items():
            preview = ", ".join(map(str, values[:8]))
            if len(values) > 8:
                preview += ", ..."
            lines.append(f"| {key} | {len(values)} | {preview} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
