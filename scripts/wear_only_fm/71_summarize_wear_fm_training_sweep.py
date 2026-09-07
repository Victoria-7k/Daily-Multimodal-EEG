#!/usr/bin/env python3
"""Summarize Wear-only FM W3FM training-parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_PHASE2_ROOT = Path("outputs/server_sync/wear_fm_phase2_20260825")
DEFAULT_SWEEP_ROOT = Path("outputs/server_sync/wear_fm_training_sweep_20260827")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SWEEP_ROOT)
    args = parser.parse_args()

    phase2 = _load_report(args.phase2_root / "wear_only_fm_phase2_report.json")
    sweep_reports = []
    for path in sorted(args.sweep_root.glob("*/wear_only_fm_phase2_report.json")):
        config = path.parent.name
        for row in _load_report(path):
            row = dict(row)
            row["config"] = config
            sweep_reports.append(row)

    rows = _summarize(phase2, sweep_reports)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "wear_fm_training_sweep_summary.csv", rows)
    _write_markdown(args.out_dir / "wear_fm_training_sweep_summary.md", rows)
    print(f"config_protocol_rows={len(rows)}")
    print(f"out_md={args.out_dir / 'wear_fm_training_sweep_summary.md'}")
    return 0


def _load_report(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def _summarize(phase2: list[dict[str, Any]], sweep_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase2_w3fm = [row for row in phase2 if row["route"] == "W3FM_frozen" and row["protocol"] in {"cross_day", "within_subject_day"}]
    baseline_rows = [row for row in phase2 if row["route"] in {"Wphysio", "Wdeep", "Wmoment_frozen"} and row["protocol"] in {"cross_day", "within_subject_day"}]
    all_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in phase2_w3fm:
        all_groups.setdefault(("phase2_W3FM_frozen", row["protocol"]), []).append(row)
    for row in sweep_reports:
        all_groups.setdefault((row["config"], row["protocol"]), []).append(row)

    out = []
    for (config, protocol), rows in sorted(all_groups.items()):
        original = [row for row in phase2_w3fm if row["protocol"] == protocol]
        best = _best_baseline(baseline_rows, protocol)
        summary = {
            "config": config,
            "protocol": protocol,
            "selection_metric": _selection_metric(rows),
            "seed_count": len(rows),
            "rmse_mean": _metric_mean(rows, "rmse"),
            "rmse_std": _metric_std(rows, "rmse"),
            "raw_r_mean": _metric_mean(rows, "raw_r"),
            "raw_r_std": _metric_std(rows, "raw_r"),
            "centered_r_mean": _metric_mean(rows, "within_subject_centered_r"),
            "centered_r_std": _metric_std(rows, "within_subject_centered_r"),
            "best_epoch_mean": _audit_mean(rows, "best_epoch"),
            "epoch_count_mean": _audit_mean(rows, "epoch_count"),
            "train_loss_drop_mean": _train_loss_drop_mean(rows),
            "delta_rmse_vs_original_w3fm": _metric_mean(rows, "rmse") - _metric_mean(original, "rmse") if original else math.nan,
            "delta_raw_r_vs_original_w3fm": _metric_mean(rows, "raw_r") - _metric_mean(original, "raw_r") if original else math.nan,
            "delta_centered_r_vs_original_w3fm": _metric_mean(rows, "within_subject_centered_r") - _metric_mean(original, "within_subject_centered_r") if original else math.nan,
            "best_phase2_baseline_route": best["route"] if best else "",
            "delta_rmse_vs_best_phase2_baseline": _metric_mean(rows, "rmse") - _metric_mean([best], "rmse") if best else math.nan,
            "delta_raw_r_vs_best_phase2_baseline": _metric_mean(rows, "raw_r") - _metric_mean([best], "raw_r") if best else math.nan,
            "delta_centered_r_vs_best_phase2_baseline": _metric_mean(rows, "within_subject_centered_r") - _metric_mean([best], "within_subject_centered_r") if best else math.nan,
        }
        out.append(summary)
    return out


def _best_baseline(rows: list[dict[str, Any]], protocol: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["protocol"] == protocol]
    if not candidates:
        return None
    grouped = {}
    for row in candidates:
        grouped.setdefault(row["route"], []).append(row)
    metric = "within_subject_centered_r" if protocol == "within_subject_day" else "raw_r"
    route, route_rows = max(grouped.items(), key=lambda item: (_metric_mean(item[1], metric), -_metric_mean(item[1], "rmse")))
    return {
        "route": route,
        "protocol": protocol,
        "test": {
            "rmse": _metric_mean(route_rows, "rmse"),
            "raw_r": _metric_mean(route_rows, "raw_r"),
            "within_subject_centered_r": _metric_mean(route_rows, "within_subject_centered_r"),
        },
    }


def _selection_metric(rows: list[dict[str, Any]]) -> str:
    values = {str(row.get("train_audit", {}).get("selection_metric", "rmse")) for row in rows}
    return ",".join(sorted(values))


def _metric_mean(rows: list[dict[str, Any]], metric: str) -> float:
    return float(mean(float(row["test"][metric]) for row in rows)) if rows else math.nan


def _metric_std(rows: list[dict[str, Any]], metric: str) -> float:
    return float(pstdev(float(row["test"][metric]) for row in rows)) if len(rows) > 1 else 0.0


def _audit_mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(mean(float(row["train_audit"][key]) for row in rows)) if rows else math.nan


def _train_loss_drop_mean(rows: list[dict[str, Any]]) -> float:
    drops = []
    for row in rows:
        history = row.get("train_audit", {}).get("history", [])
        if not history:
            continue
        first = float(history[0]["train_loss"])
        last = float(history[-1]["train_loss"])
        if first:
            drops.append((first - last) / first)
    return float(mean(drops)) if drops else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Wear-only FM Training Sweep Summary",
        "",
        "| config | protocol | select | seeds | RMSE | raw r | centered r | d RMSE vs W3FM | d raw r vs W3FM | d centered r vs W3FM | best Phase2 baseline | d RMSE vs best | d raw r vs best | d centered r vs best | best epoch | train loss drop |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['protocol']} | {row['selection_metric']} | {row['seed_count']} | "
            f"{_fmt(row['rmse_mean'])} +/- {_fmt(row['rmse_std'])} | "
            f"{_fmt(row['raw_r_mean'])} +/- {_fmt(row['raw_r_std'])} | "
            f"{_fmt(row['centered_r_mean'])} +/- {_fmt(row['centered_r_std'])} | "
            f"{_fmt(row['delta_rmse_vs_original_w3fm'])} | {_fmt(row['delta_raw_r_vs_original_w3fm'])} | "
            f"{_fmt(row['delta_centered_r_vs_original_w3fm'])} | {row['best_phase2_baseline_route']} | "
            f"{_fmt(row['delta_rmse_vs_best_phase2_baseline'])} | {_fmt(row['delta_raw_r_vs_best_phase2_baseline'])} | "
            f"{_fmt(row['delta_centered_r_vs_best_phase2_baseline'])} | {_fmt(row['best_epoch_mean'])} | "
            f"{_fmt(row['train_loss_drop_mean'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
