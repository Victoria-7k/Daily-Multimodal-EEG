#!/usr/bin/env python3
"""Run a train-only subject-mean fatigue baseline on EEGPT split protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.training.centered_metrics import (
    evaluate_regression_with_centered,
    predict_subject_train_mean,
)


LABEL_NAMES = [
    "inspired",
    "alert",
    "determined",
    "attentive",
    "active",
    "hostile",
    "nervous",
    "upset",
    "afraid",
    "ashamed",
    "fatigue",
]
DEFAULT_PROTOCOLS = ("cross_day", "within_subject_day")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-index", type=Path, required=True)
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--target", "--target-label", default="fatigue")
    parser.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS))
    parser.add_argument("--fusion-summary", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_jsonl(args.window_index)
    y = np.asarray([_target_value(row, args.target) for row in rows], dtype=np.float32)
    subjects = np.asarray([_norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    results = []
    current = _load_current_metrics(args.fusion_summary) if args.fusion_summary else {}
    for protocol in args.protocols:
        split = _load_split(args.splits_root / protocol, len(rows))
        train = split["train"]
        test = split["test"]
        prediction = predict_subject_train_mean(y[train], subjects[train], subjects[test])
        metrics = evaluate_regression_with_centered(y[test], prediction, subjects[test])
        train_subjects = set(subjects[train].tolist())
        test_subjects = set(subjects[test].tolist())
        result = {
            "protocol": protocol,
            "target_label": args.target,
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "train_subject_count": int(len(train_subjects)),
            "test_subject_count": int(len(test_subjects)),
            "subject_mean_coverage": float(len(train_subjects & test_subjects) / len(test_subjects)) if test_subjects else None,
            "unseen_test_subjects": sorted(test_subjects - train_subjects, key=str),
            "metrics": metrics,
            "current_cross_attention": current.get(protocol, {}),
        }
        result["raw_r_gap_to_current_rmse_winner"] = _difference(
            metrics.get("raw_r"), result["current_cross_attention"].get("best_rmse", {}).get("raw_r")
        )
        results.append(result)

    output = {
        "stage": 1,
        "window_index": str(args.window_index),
        "splits_root": str(args.splits_root),
        "target_label": args.target,
        "protocols": results,
    }
    _write_json(output, args.out_json)
    _write_markdown(output, args.out_md)
    for row in results:
        metrics = row["metrics"]
        print(
            f"protocol={row['protocol']} train={row['train_count']} test={row['test_count']} "
            f"rmse={_fmt(metrics['rmse'])} raw_r={_fmt(metrics['raw_r'])} "
            f"centered_r={_fmt(metrics['within_subject_centered_r'])} "
            f"coverage={_fmt(row['subject_mean_coverage'])}"
        )
    print(f"out_json={args.out_json}")
    print(f"out_md={args.out_md}")
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _target_value(row: dict[str, Any], target: str) -> float:
    labels = row.get("labels")
    if isinstance(labels, dict):
        return float(labels[target])
    if isinstance(labels, list):
        if target not in LABEL_NAMES:
            raise ValueError(f"target {target!r} is not in canonical label names")
        return float(labels[LABEL_NAMES.index(target)])
    if target in row:
        return float(row[target])
    raise ValueError("window index row has no supported labels field")


def _load_split(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    split = {name: _load_indices(path / f"{name}.json", n_rows) for name in ("pretrain", "finetune", "val", "test")}
    split["train"] = np.asarray(split["pretrain"].tolist() + split["finetune"].tolist(), dtype=np.int64)
    return split


def _load_indices(path: Path, n_rows: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    indices = np.asarray(value, dtype=np.int64).reshape(-1)
    if indices.size and (indices.min() < 0 or indices.max() >= n_rows):
        raise ValueError(f"{path} contains out-of-range indices")
    return indices


def _load_current_metrics(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("experiments", [])
    result: dict[str, Any] = {}
    for protocol in sorted({row.get("protocol") for row in rows if row.get("protocol")}):
        candidates = [row for row in rows if row.get("protocol") == protocol]
        best_rmse = min(candidates, key=lambda row: float(row["test"]["rmse"]))
        best_raw = max(candidates, key=lambda row: _none_to_inf(row.get("pooled_raw_pearson_r"), negative=True))
        result[protocol] = {
            "best_rmse": _metric_row(best_rmse),
            "best_raw_r": _metric_row(best_raw),
        }
    return result


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    test = row.get("test", {})
    return {
        "experiment": row.get("experiment"),
        "rmse": test.get("rmse"),
        "mae": test.get("mae"),
        "raw_r": row.get("pooled_raw_pearson_r", test.get("pearson_r")),
        "centered_r": row.get("within_subject_centered_r"),
    }


def _none_to_inf(value: Any, *, negative: bool = False) -> float:
    if value is None:
        return float("-inf" if negative else "inf")
    return float(value)


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# EEGPT Subject-Mean Baseline",
        "",
        f"target_label: `{output['target_label']}`",
        "",
        "| protocol | train/test | train/test subjects | coverage | RMSE | MAE | raw r | centered r | current best RMSE experiment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in output["protocols"]:
        metrics = row["metrics"]
        current = row.get("current_cross_attention", {}).get("best_rmse", {})
        lines.append(
            f"| {row['protocol']} | {row['train_count']}/{row['test_count']} | "
            f"{row['train_subject_count']}/{row['test_subject_count']} | {_fmt(row['subject_mean_coverage'])} | "
            f"{_fmt(metrics['rmse'])} | {_fmt(metrics['mae'])} | {_fmt(metrics['raw_r'])} | "
            f"{_fmt(metrics['within_subject_centered_r'])} | {current.get('experiment', 'NA')} |"
        )
    lines.extend(["", "## Interpretation fields", "", "- `raw r` is computed on the test rows pooled together.", "- `centered r` centers both prediction and target by the test-split subject mean.", "- Subject means are estimated from `pretrain + finetune` only; unseen subjects use the train global mean.", ""])
    for row in output["protocols"]:
        lines.append(f"### `{row['protocol']}`")
        lines.append("")
        lines.append(f"unseen_test_subjects: `{row['unseen_test_subjects']}`")
        lines.append("")
        current = row.get("current_cross_attention", {})
        if current:
            lines.append(f"current_best_rmse: `{json.dumps(current.get('best_rmse'), ensure_ascii=False)}`")
            lines.append(f"current_best_raw_r: `{json.dumps(current.get('best_raw_r'), ensure_ascii=False)}`")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
