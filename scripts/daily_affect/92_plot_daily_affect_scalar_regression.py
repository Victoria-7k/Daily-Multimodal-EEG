#!/usr/bin/env python3
"""Plot seed-level scalar-regression raw-r and paired window deltas."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.summary_dir.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(args.summary_dir / "run_metrics.csv")
    paired = read_csv(args.summary_dir / "paired_window_deltas.csv")
    plot_seed_metrics(metrics, out_dir / "raw_r_rmse_by_protocol.png")
    plot_paired_deltas(paired, out_dir / "paired_delta_raw_r.png")
    print(f"figures={out_dir}")
    return 0


def plot_seed_metrics(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["protocol"]].append(row)
    figure, axes = plt.subplots(1, max(1, len(grouped)), figsize=(6 * max(1, len(grouped)), 5), squeeze=False)
    for axis, (protocol, values) in zip(axes[0], sorted(grouped.items())):
        labels = sorted({row["condition_id"] for row in values})
        for index, label in enumerate(labels):
            subset = [row for row in values if row["condition_id"] == label]
            raw_r = numeric(subset, "test_raw_r")
            rmse = numeric(subset, "test_rmse")
            axis.scatter([index] * len(raw_r), raw_r, s=25, label="raw r" if index == 0 else None, color="#1f77b4")
            axis.scatter([index] * len(rmse), [-value for value in rmse], s=25, label="-RMSE" if index == 0 else None, color="#d62728", marker="x")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(protocol); axis.set_ylabel("raw r / negative RMSE"); axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=7); axis.legend()
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def plot_paired_deltas(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = parse_float(row.get("delta_raw_r"))
        if value is not None:
            grouped[(row["protocol"], row["condition_id"])].append(value)
    labels, values = zip(*sorted(grouped.items())) if grouped else ([], [])
    figure, axis = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 5))
    means = [float(np.mean(item)) for item in values]
    stds = [float(np.std(item)) for item in values]
    axis.errorbar(range(len(labels)), means, yerr=stds, fmt="o", capsize=3, color="#2ca02c")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("EMA-bag - window raw r"); axis.set_xticks(range(len(labels)))
    axis.set_xticklabels([f"{protocol}\n{condition}" for protocol, condition in labels], rotation=65, ha="right", fontsize=7)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    return [value for row in rows if (value := parse_float(row.get(key))) is not None]


def parse_float(value: str | None) -> float | None:
    if value in {None, "", "None"}:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


if __name__ == "__main__":
    raise SystemExit(main())
