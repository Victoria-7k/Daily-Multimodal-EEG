#!/usr/bin/env python3
"""Summarize paired event-level raw-r scalar-regression comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.training.centered_metrics import safe_pearsonr


BASELINE = "window_attention_regression_full_mean"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    args = parser.parse_args()
    out_dir = args.out_dir or args.runs_root.parent / "summary"
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.runs_root.rglob("metrics.json"))]
    if not rows:
        raise FileNotFoundError(f"no metrics.json beneath {args.runs_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "run_metrics.csv", flatten_rows(rows))
    paired = paired_rows(rows, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
    write_csv(out_dir / "paired_window_deltas.csv", paired)
    summary = aggregate(paired)
    write_csv(out_dir / "condition_summary.csv", summary)
    (out_dir / "summary.json").write_text(json.dumps({"run_count": len(rows), "paired_rows": paired, "condition_summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", summary)
    print(f"runs={len(rows)} paired={len(paired)} out_dir={out_dir}")
    return 0


def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "protocol": row["protocol"], "route_id": row["route_id"], "condition_id": row["condition_id"],
            "family": row["family"], "model_id": row["model_id"], "seed": row["seed"],
            "objective_id": row["objective_id"], "supervision_unit": row["supervision_unit"],
            "test_raw_r": row["test"]["raw_r"], "test_rmse": row["test"]["rmse"], "test_mae": row["test"]["mae"],
            "test_centered_r": row["test"]["within_subject_centered_r"], "prediction_std": row["test"]["prediction_std"],
            "best_epoch": row["train_audit"]["best_epoch"], "prediction_path": row["prediction_path"],
        }
        for row in rows
    ]


def paired_rows(rows: list[dict[str, Any]], *, replicates: int, seed: int) -> list[dict[str, Any]]:
    keyed = {(r["protocol"], r["route_id"], int(r["seed"]), r["condition_id"]): r for r in rows}
    output: list[dict[str, Any]] = []
    for key, candidate in keyed.items():
        protocol, route_id, run_seed, condition_id = key
        if condition_id == BASELINE:
            continue
        baseline = keyed.get((protocol, route_id, run_seed, BASELINE))
        if baseline is None:
            continue
        boot = paired_subject_day_bootstrap(baseline["prediction_path"], candidate["prediction_path"], replicates=replicates, seed=seed + run_seed)
        output.append(
            {
                "protocol": protocol, "route_id": route_id, "seed": run_seed, "condition_id": condition_id,
                "model_id": candidate["model_id"], "raw_r_window": baseline["test"]["raw_r"], "raw_r_ema_bag": candidate["test"]["raw_r"],
                "delta_raw_r": delta(candidate["test"]["raw_r"], baseline["test"]["raw_r"]),
                "rmse_window": baseline["test"]["rmse"], "rmse_ema_bag": candidate["test"]["rmse"],
                "delta_rmse": delta(candidate["test"]["rmse"], baseline["test"]["rmse"]), **boot,
            }
        )
    return output


def paired_subject_day_bootstrap(window_path: str, bag_path: str, *, replicates: int, seed: int) -> dict[str, Any]:
    with np.load(window_path, allow_pickle=False) as window, np.load(bag_path, allow_pickle=False) as bag:
        test = window["test_index"].astype(np.int64)
        for key in ("event_id", "label", "subject_id", "day_id"):
            if not np.array_equal(window[key][test], bag[key][bag["test_index"].astype(np.int64)]):
                raise ValueError(f"paired prediction identity mismatch for {key}: {window_path} vs {bag_path}")
        true, subject, day = window["label"][test], window["subject_id"][test].astype(str), window["day_id"][test].astype(str)
        window_pred, bag_pred = window["test_event_prediction"], bag["test_event_prediction"]
    groups = [np.flatnonzero((subject == s) & (day == d)) for s, d in sorted(set(zip(subject.tolist(), day.tolist())))]
    rng, values = np.random.default_rng(seed), []
    for _ in range(max(1, int(replicates))):
        selected = np.concatenate([groups[index] for index in rng.integers(0, len(groups), size=len(groups))])
        values.append(delta(safe_pearsonr(true[selected], bag_pred[selected]), safe_pearsonr(true[selected], window_pred[selected])))
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return {
        "bootstrap_unit": "subject_day", "bootstrap_replicates": int(replicates), "bootstrap_valid": int(len(finite)),
        "bootstrap_delta_raw_r_mean": float(np.mean(finite)) if len(finite) else None,
        "bootstrap_delta_raw_r_ci_low": float(np.quantile(finite, 0.025)) if len(finite) else None,
        "bootstrap_delta_raw_r_ci_high": float(np.quantile(finite, 0.975)) if len(finite) else None,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["protocol"], row["condition_id"]), []).append(row)
    output = []
    for (protocol, condition), values in sorted(grouped.items()):
        deltas = numeric(values, "delta_raw_r")
        output.append(
            {
                "protocol": protocol, "condition_id": condition, "seed_count": len(values),
                "mean_delta_raw_r": mean_or_none(deltas), "std_delta_raw_r": std_or_none(deltas),
                "positive_seed_count": int(sum(value > 0 for value in deltas)),
                "mean_bootstrap_ci_low": mean_or_none(numeric(values, "bootstrap_delta_raw_r_ci_low")),
                "mean_bootstrap_ci_high": mean_or_none(numeric(values, "bootstrap_delta_raw_r_ci_high")),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Daily-affect scalar-regression paired summary", "", "| protocol | condition | seeds | mean delta raw r | std | positive seeds | mean bootstrap 95% CI |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['protocol']} | {row['condition_id']} | {row['seed_count']} | {fmt(row['mean_delta_raw_r'])} | {fmt(row['std_delta_raw_r'])} | {row['positive_seed_count']} | [{fmt(row['mean_bootstrap_ci_low'])}, {fmt(row['mean_bootstrap_ci_high'])}] |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def std_or_none(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
