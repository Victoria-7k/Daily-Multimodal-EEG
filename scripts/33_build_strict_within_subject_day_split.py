#!/usr/bin/env python3
"""Build a strict within-subject held-out-day split for EEGPT windows."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--pretrain-ratio-of-train", type=float, default=0.35)
    parser.add_argument("--name", default="within_subject_day_strict")
    args = parser.parse_args()

    rows = _load_jsonl(args.window_index)
    split = build_strict_within_subject_day_split(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        pretrain_ratio_of_train=args.pretrain_ratio_of_train,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pretrain", "finetune", "val", "test"):
        (out_dir / f"{name}.json").write_text(
            json.dumps(split[name], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    info = _split_info(rows, split, name=args.name, source_window_index=args.window_index)
    (out_dir / "split_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(f"train={len(split['pretrain']) + len(split['finetune'])}")
    print(f"val={len(split['val'])}")
    print(f"test={len(split['test'])}")
    print("subject_day_overlap_train_val=0")
    print("subject_day_overlap_train_test=0")
    print("subject_day_overlap_val_test=0")
    return 0


def build_strict_within_subject_day_split(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    pretrain_ratio_of_train: float = 0.35,
) -> dict[str, list[int]]:
    if not rows:
        raise ValueError("window index is empty")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    if not 0.0 <= pretrain_ratio_of_train <= 1.0:
        raise ValueError("pretrain_ratio_of_train must be in [0, 1]")

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        subject = _norm_subject(row.get("subject_id"))
        day = _norm_day(row.get("day_id", row.get("day")))
        grouped[subject][day].append(index)

    pretrain: list[int] = []
    finetune: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for subject in sorted(grouped, key=str):
        day_items = sorted(grouped[subject].items(), key=lambda item: _day_sort_key(item[0]))
        day_count = len(day_items)
        if day_count < 3:
            raise ValueError(f"{subject} has fewer than three days; cannot build train/val/test split")
        train_days = max(1, int(math.floor(day_count * train_ratio)))
        val_days = max(1, int(math.floor(day_count * val_ratio)))
        if train_days + val_days >= day_count:
            val_days = max(1, day_count - train_days - 1)
        if train_days + val_days >= day_count:
            train_days = max(1, day_count - val_days - 1)
        test_days = day_count - train_days - val_days
        if test_days <= 0:
            raise ValueError(f"{subject} split has no held-out test days")

        train_items = day_items[:train_days]
        val_items = day_items[train_days : train_days + val_days]
        test_items = day_items[train_days + val_days :]
        pretrain_days = int(round(len(train_items) * pretrain_ratio_of_train))
        if len(train_items) > 1:
            pretrain_days = min(max(1, pretrain_days), len(train_items) - 1)
        else:
            pretrain_days = len(train_items)
        for _, indices in train_items[:pretrain_days]:
            pretrain.extend(indices)
        for _, indices in train_items[pretrain_days:]:
            finetune.extend(indices)
        for _, indices in val_items:
            val.extend(indices)
        for _, indices in test_items:
            test.extend(indices)

    split = {
        "pretrain": sorted(pretrain),
        "finetune": sorted(finetune),
        "val": sorted(val),
        "test": sorted(test),
    }
    _validate_no_overlap(rows, split)
    return split


def _validate_no_overlap(rows: list[dict[str, Any]], split: dict[str, list[int]]) -> None:
    index_sets = {name: set(values) for name, values in split.items()}
    train = index_sets["pretrain"] | index_sets["finetune"]
    combined = {
        "train": train,
        "val": index_sets["val"],
        "test": index_sets["test"],
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = combined[left] & combined[right]
        if overlap:
            raise ValueError(f"index overlap between {left} and {right}: {len(overlap)}")
    pair_sets = {name: {_subject_day(rows[index]) for index in values} for name, values in combined.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = pair_sets[left] & pair_sets[right]
        if overlap:
            preview = ", ".join(sorted(overlap, key=str)[:5])
            raise ValueError(f"subject-day overlap between {left} and {right}: {len(overlap)} ({preview})")


def _split_info(
    rows: list[dict[str, Any]],
    split: dict[str, list[int]],
    *,
    name: str,
    source_window_index: Path,
) -> dict[str, Any]:
    train = sorted(split["pretrain"] + split["finetune"])
    combined = {
        "pretrain": split["pretrain"],
        "finetune": split["finetune"],
        "train": train,
        "val": split["val"],
        "test": split["test"],
    }
    return {
        "name": name,
        "source_window_index": str(source_window_index),
        "row_count": len(rows),
        "unit": "subject_day_pair",
        "rule": "within each subject, sorted days are assigned to train, val, and test; all windows from one subject-day pair stay in one split",
        "counts": {key: len(value) for key, value in combined.items()},
        "subject_day_counts": {key: len({_subject_day(rows[index]) for index in value}) for key, value in combined.items()},
        "subject_counts": {key: len({_norm_subject(rows[index].get("subject_id")) for index in value}) for key, value in combined.items()},
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


def _subject_day(row: dict[str, Any]) -> str:
    return f"{_norm_subject(row.get('subject_id'))}::{_norm_day(row.get('day_id', row.get('day')))}"


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


def _day_sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(float(value)))
    except ValueError:
        return (1, value)


if __name__ == "__main__":
    raise SystemExit(main())
