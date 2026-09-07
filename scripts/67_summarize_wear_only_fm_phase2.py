#!/usr/bin/env python3
"""Summarize Wear-only FM Phase 2 runs without doing Phase 3 bootstrap."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, default=Path("outputs/wear_fm/phase2/wear_only_fm_phase2_report.json"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/wear_fm/phase2/wear_only_fm_phase2_summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/wear_fm/phase2/wear_only_fm_phase2_summary.md"))
    args = parser.parse_args()
    report = json.loads(args.report_json.read_text(encoding="utf-8"))
    output = summarize_phase2(report, source_report=args.report_json)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, args.out_md)
    print(f"summary_json={args.out_json}")
    print(f"summary_md={args.out_md}")
    for row in output["summary"]:
        print(
            row["protocol"],
            row["route"],
            "rmse",
            _fmt(row["rmse_mean"]),
            "raw_r",
            _fmt(row["raw_r_mean"]),
            "centered",
            _fmt(row["within_subject_centered_r_mean"]),
        )
    return 0


def summarize_phase2(report: dict[str, Any], *, source_report: Path) -> dict[str, Any]:
    rows = list(report["results"])
    summary: list[dict[str, Any]] = []
    for protocol in report["protocols"]:
        for route in report["routes"]:
            group = [row for row in rows if row["protocol"] == protocol and row["route"] == route]
            item: dict[str, Any] = {"protocol": protocol, "route": route, "runs": len(group)}
            for metric in ("rmse", "mae", "raw_r", "within_subject_centered_r", "per_subject_r_mean"):
                values = [row["test"][metric] for row in group if row["test"].get(metric) is not None]
                item[f"{metric}_mean"] = statistics.mean(values) if values else None
                item[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)
            summary.append(item)
    paired = _paired_vs_best_baseline(rows, protocols=report["protocols"], seeds=report["seeds"])
    return {
        "stage": "wear_only_fm_phase2_summary",
        "source_report": str(source_report),
        "run_count": int(report["run_count"]),
        "mask_count": int(report["mask_count"]),
        "summary": summary,
        "paired_vs_best_baseline_by_raw_r": paired,
    }


def _paired_vs_best_baseline(rows: list[dict[str, Any]], *, protocols: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    baselines = {"Wphysio", "Wdeep", "Wmoment_frozen"}
    for protocol in protocols:
        for seed in seeds:
            base = [row for row in rows if row["protocol"] == protocol and row["seed"] == seed and row["route"] in baselines]
            w3fm = next(row for row in rows if row["protocol"] == protocol and row["seed"] == seed and row["route"] == "W3FM_frozen")
            best = max(
                base,
                key=lambda row: (
                    row["test"]["raw_r"] if row["test"].get("raw_r") is not None else -999.0,
                    -(row["test"]["rmse"] or 999.0),
                ),
            )
            paired.append(
                {
                    "protocol": protocol,
                    "seed": int(seed),
                    "best_baseline": best["route"],
                    "best_baseline_raw_r": best["test"]["raw_r"],
                    "w3fm_raw_r": w3fm["test"]["raw_r"],
                    "delta_raw_r": _delta(w3fm["test"]["raw_r"], best["test"]["raw_r"]),
                    "best_baseline_rmse": best["test"]["rmse"],
                    "w3fm_rmse": w3fm["test"]["rmse"],
                    "delta_rmse": _delta(w3fm["test"]["rmse"], best["test"]["rmse"]),
                    "best_baseline_centered_r": best["test"]["within_subject_centered_r"],
                    "w3fm_centered_r": w3fm["test"]["within_subject_centered_r"],
                    "delta_centered_r": _delta(w3fm["test"]["within_subject_centered_r"], best["test"]["within_subject_centered_r"]),
                }
            )
    return paired


def write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Wear-only FM Phase 2 Summary",
        "",
        f"- run_count: `{output['run_count']}`",
        f"- mask_count: `{output['mask_count']}`",
        "",
        "## Mean +/- Std",
        "",
        "| protocol | route | runs | RMSE | raw r | centered r |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in output["summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['runs']} | "
            f"{_mean_std(row, 'rmse')} | {_mean_std(row, 'raw_r')} | {_mean_std(row, 'within_subject_centered_r')} |"
        )
    lines.extend(
        [
            "",
            "## W3FM Vs Best Baseline By Seed",
            "",
            "| protocol | seed | best baseline | delta raw r | delta RMSE | delta centered r |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in output["paired_vs_best_baseline_by_raw_r"]:
        lines.append(
            f"| {row['protocol']} | {row['seed']} | {row['best_baseline']} | "
            f"{_fmt(row['delta_raw_r'])} | {_fmt(row['delta_rmse'])} | {_fmt(row['delta_centered_r'])} |"
        )
    lines.append("")
    lines.append("This file summarizes Phase 2 runs only. It does not perform the Phase 3 subject-day block bootstrap.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "NA"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
