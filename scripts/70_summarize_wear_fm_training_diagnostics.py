#!/usr/bin/env python3
"""Summarize Wear-only FM training curves and early-stopping behavior."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_PHASE2_ROOT = Path("outputs/server_sync/wear_fm_phase2_20260825")
DEFAULT_INTERNAL_ROOT = Path("outputs/server_sync/wear_fm_internal_ablation_20260825")
DEFAULT_OUT_DIR = Path("outputs/server_sync/wear_fm_training_diagnostics_20260827")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--internal-root", type=Path, default=DEFAULT_INTERNAL_ROOT)
    parser.add_argument("--extra-root", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    roots = [("phase2", args.phase2_root), ("internal", args.internal_root)]
    roots.extend((root.name, root) for root in args.extra_root)
    rows = []
    for stage, root in roots:
        rows.extend(_read_run_rows(stage, root))
    if not rows:
        raise SystemExit("no run metrics found")

    summary = _summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "training_curve_detail.csv", rows)
    _write_csv(args.out_dir / "training_curve_summary.csv", summary)
    _write_markdown(args.out_dir / "wear_fm_training_diagnostics.md", rows, summary)
    print(f"run_count={len(rows)}")
    print(f"out_dir={args.out_dir}")
    return 0


def _read_run_rows(stage: str, root: Path) -> list[dict[str, Any]]:
    runs_root = root / "runs"
    if not runs_root.is_dir():
        return []
    rows = []
    for metrics_path in sorted(runs_root.glob("*/*/seed_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = _read_history(metrics_path.parent / "val_history.csv")
        if not history:
            continue
        first = history[0]
        last = history[-1]
        best_val = min(_finite_values(row.get("val_rmse") for row in history), default=math.nan)
        first_train = _to_float(first.get("train_loss"))
        last_train = _to_float(last.get("train_loss"))
        train_drop = (first_train - last_train) / first_train if first_train and math.isfinite(first_train) else math.nan
        row = {
            "stage": stage,
            "protocol": metrics["protocol"],
            "route": metrics["route"],
            "seed": int(metrics["seed"]),
            "best_epoch": int(metrics["train_audit"]["best_epoch"]),
            "epoch_count": int(metrics["train_audit"]["epoch_count"]),
            "selection_metric": metrics["train_audit"].get("selection_metric", "rmse"),
            "train_rmse": float(metrics["train"]["rmse"]),
            "val_rmse": float(metrics["val"]["rmse"]),
            "test_rmse": float(metrics["test"]["rmse"]),
            "train_raw_r": float(metrics["train"]["raw_r"]),
            "val_raw_r": float(metrics["val"]["raw_r"]),
            "test_raw_r": float(metrics["test"]["raw_r"]),
            "train_centered_r": float(metrics["train"]["within_subject_centered_r"]),
            "val_centered_r": float(metrics["val"]["within_subject_centered_r"]),
            "test_centered_r": float(metrics["test"]["within_subject_centered_r"]),
            "first_train_loss": first_train,
            "last_train_loss": last_train,
            "train_loss_drop_frac": train_drop,
            "first_val_rmse": _to_float(first.get("val_rmse")),
            "best_val_rmse": best_val,
            "last_val_rmse": _to_float(last.get("val_rmse")),
            "val_rmse_drift_after_best": _to_float(last.get("val_rmse")) - best_val if math.isfinite(best_val) else math.nan,
            "val_train_rmse_gap": float(metrics["val"]["rmse"]) - float(metrics["train"]["rmse"]),
            "test_train_rmse_gap": float(metrics["test"]["rmse"]) - float(metrics["train"]["rmse"]),
            "fast_overfit_flag": False,
        }
        row["fast_overfit_flag"] = bool(row["train_loss_drop_frac"] >= 0.3 and row["val_rmse_drift_after_best"] >= 0.03)
        rows.append(row)
    return rows


def _read_history(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: _to_float(value) for key, value in row.items()})
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["stage"], row["protocol"], row["route"]), []).append(row)
    out = []
    for (stage, protocol, route), values in sorted(groups.items()):
        out.append(
            {
                "stage": stage,
                "protocol": protocol,
                "route": route,
                "run_count": len(values),
                "best_epoch_mean": _mean(values, "best_epoch"),
                "best_epoch_min": min(row["best_epoch"] for row in values),
                "best_epoch_max": max(row["best_epoch"] for row in values),
                "epoch_count_mean": _mean(values, "epoch_count"),
                "train_rmse_mean": _mean(values, "train_rmse"),
                "val_rmse_mean": _mean(values, "val_rmse"),
                "test_rmse_mean": _mean(values, "test_rmse"),
                "val_train_rmse_gap_mean": _mean(values, "val_train_rmse_gap"),
                "test_train_rmse_gap_mean": _mean(values, "test_train_rmse_gap"),
                "train_loss_drop_frac_mean": _mean(values, "train_loss_drop_frac"),
                "val_rmse_drift_after_best_mean": _mean(values, "val_rmse_drift_after_best"),
                "fast_overfit_count": sum(1 for row in values if row["fast_overfit_flag"]),
            }
        )
    return out


def _write_markdown(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    w3fm_like = [row for row in rows if row["route"].startswith("W3FM")]
    baseline_like = [row for row in rows if not row["route"].startswith("W3FM")]
    lines = [
        "# Wear-only FM Training Diagnostics",
        "",
        "## Diagnosis",
        "",
        f"- analyzed_runs: `{len(rows)}`",
        f"- w3fm_like_fast_overfit: `{sum(row['fast_overfit_flag'] for row in w3fm_like)}/{len(w3fm_like)}`",
        f"- baseline_like_fast_overfit: `{sum(row['fast_overfit_flag'] for row in baseline_like)}/{len(baseline_like)}`",
        f"- w3fm_like_train_loss_drop_mean: `{_fmt(mean(row['train_loss_drop_frac'] for row in w3fm_like))}`",
        f"- baseline_like_train_loss_drop_mean: `{_fmt(mean(row['train_loss_drop_frac'] for row in baseline_like))}`",
        "",
        "Early best epochs are checkpoint-selection symptoms, not short training runs: most runs continue for patience windows after the best epoch. W3FM-like routes usually keep lowering train loss while validation RMSE drifts upward, which is consistent with fast overfitting or too-aggressive optimization.",
        "",
        "## Group Summary",
        "",
        "| stage | protocol | route | n | best epoch | train RMSE | val RMSE | test RMSE | train loss drop | val drift | overfit flags |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['stage']} | {row['protocol']} | {row['route']} | {row['run_count']} | "
            f"{_fmt(row['best_epoch_mean'])} ({row['best_epoch_min']}-{row['best_epoch_max']}) | "
            f"{_fmt(row['train_rmse_mean'])} | {_fmt(row['val_rmse_mean'])} | {_fmt(row['test_rmse_mean'])} | "
            f"{_fmt(row['train_loss_drop_frac_mean'])} | {_fmt(row['val_rmse_drift_after_best_mean'])} | "
            f"{row['fast_overfit_count']}/{row['run_count']} |"
        )
    lines.extend(
        [
            "",
            "## Suggested Sweep",
            "",
            "| config | purpose | key args |",
            "| --- | --- | --- |",
            "| `lr3e4` | slow down W3FM projection/gate/head fitting | `--learning-rate 3e-4 --epochs 120 --patience 25` |",
            "| `lr1e4` | test whether epoch-1 best is mainly step size | `--learning-rate 1e-4 --epochs 160 --patience 35` |",
            "| `lr3e4_reg` | add regularization and reduce head capacity | `--learning-rate 3e-4 --dropout 0.3 --weight-decay 1e-3 --hidden-dim 64 --epochs 120 --patience 25` |",
            "| `centered_select` | diagnostic only for within-subject centered-r conflict | `--selection-metric centered_r` |",
            "",
            "Keep these as diagnostic sweeps until paired multi-seed results beat the frozen Phase 2 baselines under the declared protocol metric.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(values):
    return [value for value in values if math.isfinite(_to_float(value))]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(mean(float(row[key]) for row in rows))


def _std(rows: list[dict[str, Any]], key: str) -> float:
    return float(pstdev(float(row[key]) for row in rows))


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def _fmt(value: Any) -> str:
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
