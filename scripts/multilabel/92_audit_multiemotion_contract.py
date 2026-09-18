#!/usr/bin/env python3
"""Audit the data, split, label, and token contracts for multi-emotion training."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.split_paths import resolve_protocol_split_root
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases


def load_base_module():
    path = Path(__file__).with_name("48_run_fixed_tokens_multilabel_fusion.py")
    spec = importlib.util.spec_from_file_location("_fixed_multilabel_audit_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
install_numpy_core_pickle_aliases()
DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_ROOT / "outputs/splits")
    parser.add_argument("--protocols", default="cross_day,within_subject_day")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT / "outputs/multiemotion_20260913/phase0_contract")
    args = parser.parse_args()

    rows = BASE._load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    targets = BASE._load_targets(rows)
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    subject_id = np.asarray([BASE._norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    day_id = np.asarray([str(row.get("day_id", "")) for row in rows], dtype=str)
    event_window_id = np.asarray([int(row.get("event_window_id", -1)) for row in rows], dtype=np.int64)
    label_audit = BASE._audit_labels(targets, rows, event_window_id)
    event_audit = audit_events(event_id, subject_id, day_id, event_window_id, targets)
    split_audit: dict[str, Any] = {}
    canonical_splits: dict[str, dict[str, np.ndarray]] = {}
    split_gate = True
    for protocol in split_csv(args.protocols):
        protocol_root = resolve_protocol_split_root(args.splits_root, protocol)
        split = BASE._load_split(protocol_root, len(rows))
        canonical_splits[protocol] = split
        report = audit_split(protocol_root, split, event_id, subject_id, day_id)
        split_audit[protocol] = report
        split_gate = split_gate and report["gate_passed"]
    token_inventory = audit_tokens(args.embeddings_root, sample_id, canonical_splits)
    token_gate = all(row["gate_passed"] for row in token_inventory["fixed_tokens"])
    partial_ft_gate = all(row["gate_passed"] for row in token_inventory["fatigue_partial_ft_tokens"])
    contract = {
        "row_count": len(rows),
        "event_count": int(np.unique(event_id).size),
        "label_names": list(BASE.LABEL_NAMES),
        "label_gate_passed": bool(label_audit["gate_passed"]),
        "event_gate_passed": bool(event_audit["gate_passed"]),
        "split_gate_passed": bool(split_gate),
        "token_gate_passed": bool(token_gate),
        "fatigue_partial_ft_token_gate_passed": bool(partial_ft_gate),
        "gate_passed": bool(
            label_audit["gate_passed"] and event_audit["gate_passed"] and split_gate and token_gate and partial_ft_gate
        ),
        "event_audit": event_audit,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(contract, args.out_root / "phase0_contract.json")
    write_json(split_audit, args.out_root / "phase0_split_overlap.json")
    write_json(token_inventory, args.out_root / "phase0_token_inventory.json")
    write_label_csv(label_audit, args.out_root / "phase0_label_distribution.csv")
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0 if contract["gate_passed"] else 2


def audit_events(
    event_id: np.ndarray,
    subject_id: np.ndarray,
    day_id: np.ndarray,
    event_window_id: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    problems: list[str] = []
    counts = []
    for event in np.unique(event_id):
        idx = np.flatnonzero(event_id == event)
        counts.append(len(idx))
        if len(idx) != 23 or set(event_window_id[idx].tolist()) != set(range(23)):
            problems.append(f"{event}:invalid_windows")
        if np.unique(subject_id[idx]).size != 1 or np.unique(day_id[idx]).size != 1:
            problems.append(f"{event}:cross_subject_or_day")
        if not np.allclose(targets[idx], targets[idx[0]], atol=1e-6):
            problems.append(f"{event}:inconsistent_labels")
    return {
        "event_count": int(np.unique(event_id).size),
        "window_count_min": int(min(counts)),
        "window_count_max": int(max(counts)),
        "problem_count": len(problems),
        "problems": problems[:100],
        "gate_passed": not problems,
    }


def audit_split(
    protocol_root: Path,
    split: dict[str, np.ndarray],
    event_id: np.ndarray,
    subject_id: np.ndarray,
    day_id: np.ndarray,
) -> dict[str, Any]:
    leaves = {name: split[name] for name in ("train", "val", "test")}
    overlap: dict[str, Any] = {}
    gate = True
    names = tuple(leaves)
    for left_pos, left_name in enumerate(names):
        for right_name in names[left_pos + 1 :]:
            left = leaves[left_name]
            right = leaves[right_name]
            window_overlap = len(set(left.tolist()) & set(right.tolist()))
            event_overlap = len(set(event_id[left].tolist()) & set(event_id[right].tolist()))
            subject_day_overlap = len(
                set(zip(subject_id[left].tolist(), day_id[left].tolist()))
                & set(zip(subject_id[right].tolist(), day_id[right].tolist()))
            )
            overlap[f"{left_name}_{right_name}"] = {
                "window_overlap": window_overlap,
                "event_overlap": event_overlap,
                "subject_day_overlap": subject_day_overlap,
            }
            gate = gate and window_overlap == 0 and event_overlap == 0 and subject_day_overlap == 0
    return {
        "protocol_root": str(protocol_root),
        "counts": {name: int(len(index)) for name, index in split.items()},
        "event_counts": {name: int(np.unique(event_id[index]).size) for name, index in leaves.items()},
        "subject_day_counts": {
            name: len(set(zip(subject_id[index].tolist(), day_id[index].tolist()))) for name, index in leaves.items()
        },
        "overlap": overlap,
        "gate_passed": bool(gate),
    }


def audit_tokens(
    embeddings_root: Path,
    sample_id: np.ndarray,
    canonical_splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    fixed = []
    for name, branch in BASE.BRANCHES.items():
        path = embeddings_root / branch.filename
        row: dict[str, Any] = {"name": name, "path": str(path), "exists": path.is_file(), "gate_passed": False}
        if path.is_file():
            with np.load(path, allow_pickle=True) as loaded:
                ids = loaded["sample_id"].astype(str)
                emb_key = branch.emb_key if branch.emb_key in loaded.files else "face_emb"
                shape = tuple(int(value) for value in loaded[emb_key].shape)
                row.update(
                    {
                        "embedding_shape": shape,
                        "sample_id_aligned": bool(np.array_equal(ids, sample_id)),
                        "gate_passed": bool(shape == (len(sample_id), 256) and np.array_equal(ids, sample_id)),
                    }
                )
        fixed.append(row)
    partial = []
    for protocol, expected_split in canonical_splits.items():
        path = embeddings_root / f"eeg_encoder_256d_tokens/{protocol}/eegpt_partial_ft_v1/seed_240800.npz"
        row = {"protocol": protocol, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            with np.load(path, allow_pickle=True) as loaded:
                row["files"] = list(loaded.files)
                row["eeg_emb_shape"] = list(loaded["eeg_emb"].shape) if "eeg_emb" in loaded.files else None
                row["sample_id_aligned"] = bool(
                    "sample_id" in loaded.files and np.array_equal(loaded["sample_id"].astype(str), sample_id)
                )
                embedded_split = {
                    name: loaded[f"{name}_index"].astype(np.int64)
                    for name in ("train", "val", "test")
                    if f"{name}_index" in loaded.files
                }
                row["embedded_split_counts"] = {name: int(len(index)) for name, index in embedded_split.items()}
                row["split_indices_match"] = bool(
                    set(embedded_split) == {"train", "val", "test"}
                    and all(np.array_equal(embedded_split[name], expected_split[name]) for name in embedded_split)
                )
                row["gate_passed"] = bool(row["sample_id_aligned"] and row["split_indices_match"])
        else:
            row["gate_passed"] = False
        partial.append(row)
    return {"fixed_tokens": fixed, "fatigue_partial_ft_tokens": partial}


def write_label_csv(label_audit: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "n", "mean", "std", "min", "median", "max", "flags", "count_1", "count_2", "count_3", "count_4", "count_5"])
        for label in BASE.LABEL_NAMES:
            row = label_audit["per_label"][label]
            writer.writerow(
                [label, row["n"], row["mean"], row["std"], row["min"], row["median"], row["max"], ";".join(row["flags"])]
                + [row["rounded_label_counts"][str(score)] for score in range(1, 6)]
            )


def write_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
