#!/usr/bin/env python3
"""Plot matched event-level predictions for window regression and daily-affect."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_COMPARISON_ROOT = Path("outputs/daily_affect_window_event_bridge_20260906/comparison")
DEFAULT_SEEDS = (240729, 240730, 240731)
METRIC_KEYS = ("qwk", "expected_raw_r", "expected_within_subject_centered_r", "expected_rmse")
EVENT_NUMBER = re.compile(r"(\d+)$")
PANEL_SPECS = (
    ("window_regression", "attention_regression", "Window regression: 23-window full mean", "#009E73"),
    ("daily_affect", "bag_static", "Daily-affect: bag static", "#0072B2"),
    ("daily_affect", "dynamic_kernel_prior_uniform", "Daily-affect: dynamic kernel + prior", "#D55E00"),
)


@dataclass(frozen=True)
class Ensemble:
    system: str
    model_id: str
    event_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    label: np.ndarray
    seed_scores: np.ndarray
    metrics_by_seed: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--eeg-branch", default="eeg_eegpt_partial_ft_v1")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    return parser.parse_args()


def parse_seed_csv(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def load_ensembles(root: Path, seeds: tuple[int, ...]) -> tuple[Ensemble, ...]:
    rows_by_model: dict[tuple[str, str], dict[int, dict[str, dict[str, str]]]] = {}
    with (root / "event_predictions.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["system"], row["model_id"])
            if key not in {(spec[0], spec[1]) for spec in PANEL_SPECS}:
                continue
            seed = int(row["seed"])
            if seed not in seeds:
                continue
            event_rows = rows_by_model.setdefault(key, {}).setdefault(seed, {})
            event_id = row["event_id"]
            if event_id in event_rows:
                raise ValueError(f"duplicate {key} seed {seed} event {event_id}")
            event_rows[event_id] = row
    metrics = load_metric_rows(root / "seed_metrics.csv", seeds)
    return tuple(
        assemble_ensemble(rows_by_model, metrics, system, model_id, seeds)
        for system, model_id, _, _ in PANEL_SPECS
    )


def load_metric_rows(path: Path, seeds: tuple[int, ...]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    rows: dict[tuple[str, str], dict[str, list[float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["system"], row["model_id"])
            if key not in {(spec[0], spec[1]) for spec in PANEL_SPECS} or int(row["seed"]) not in seeds:
                continue
            is_window_native = key[0] == "window_regression" and row["pooling_policy"] == "full_mean" and row["decision_rule"] == "native_round"
            is_daily_native = key[0] == "daily_affect" and row["pooling_policy"] == "joint_23_window" and row["decision_rule"] == "native_argmax"
            if not (is_window_native or is_daily_native):
                continue
            bucket = rows.setdefault(key, {metric: [] for metric in METRIC_KEYS})
            for metric in METRIC_KEYS:
                bucket[metric].append(float(row[metric]))
    return {
        key: {metric: np.asarray(values, dtype=np.float64) for metric, values in by_metric.items()}
        for key, by_metric in rows.items()
    }


def assemble_ensemble(
    rows_by_model: dict[tuple[str, str], dict[int, dict[str, dict[str, str]]]],
    metrics: dict[tuple[str, str], dict[str, np.ndarray]],
    system: str,
    model_id: str,
    seeds: tuple[int, ...],
) -> Ensemble:
    key = (system, model_id)
    per_seed = rows_by_model.get(key, {})
    missing = [seed for seed in seeds if seed not in per_seed]
    if missing:
        raise ValueError(f"{key} missing event predictions for seeds {missing}")
    reference_seed = seeds[0]
    reference = per_seed[reference_seed]
    event_ids = tuple(reference)
    for seed in seeds[1:]:
        if set(per_seed[seed]) != set(event_ids):
            raise ValueError(f"{key} seed {seed} has a different held-out event set")
        for event_id in event_ids:
            for column in ("subject_id", "day_id", "label"):
                if per_seed[seed][event_id][column] != reference[event_id][column]:
                    raise ValueError(f"{key} seed {seed} disagrees on {event_id} {column}")
    if key not in metrics or any(len(values) != len(seeds) for values in metrics[key].values()):
        raise ValueError(f"{key} native metric rows are incomplete")
    return Ensemble(
        system=system,
        model_id=model_id,
        event_id=np.asarray(event_ids, dtype=str),
        subject_id=np.asarray([reference[event_id]["subject_id"] for event_id in event_ids], dtype=str),
        day_id=np.asarray([reference[event_id]["day_id"] for event_id in event_ids], dtype=str),
        label=np.asarray([float(reference[event_id]["label"]) for event_id in event_ids], dtype=np.float64),
        seed_scores=np.asarray(
            [[float(per_seed[seed][event_id]["expected_score"]) for event_id in event_ids] for seed in seeds],
            dtype=np.float64,
        ),
        metrics_by_seed=metrics[key],
    )


def trailing_event_number(event_id: str) -> int:
    match = EVENT_NUMBER.search(event_id)
    return int(match.group(1)) if match else -1


def event_order(ensemble: Ensemble) -> np.ndarray:
    return np.asarray(
        sorted(
            range(len(ensemble.event_id)),
            key=lambda index: (
                str(ensemble.subject_id[index]),
                str(ensemble.day_id[index]),
                trailing_event_number(str(ensemble.event_id[index])),
                str(ensemble.event_id[index]),
            ),
        ),
        dtype=np.int64,
    )


def assert_same_events(ensembles: tuple[Ensemble, ...]) -> None:
    first = ensembles[0]
    for ensemble in ensembles[1:]:
        for column in ("event_id", "subject_id", "day_id", "label"):
            if not np.array_equal(getattr(first, column), getattr(ensemble, column)):
                raise ValueError(f"{first.model_id} and {ensemble.model_id} differ on {column}")


def group_boundaries(ensemble: Ensemble, order: np.ndarray) -> np.ndarray:
    groups = np.asarray(
        [f"{ensemble.subject_id[index]}::{ensemble.day_id[index]}" for index in order], dtype=str
    )
    return np.flatnonzero(groups[1:] != groups[:-1]) + 0.5


def metric_summary(metrics_by_seed: dict[str, np.ndarray]) -> str:
    labels = {
        "qwk": "QWK",
        "expected_raw_r": "raw r",
        "expected_within_subject_centered_r": "centered r",
        "expected_rmse": "RMSE",
    }
    parts: list[str] = []
    for metric in METRIC_KEYS:
        values = metrics_by_seed[metric]
        deviation = values.std(ddof=1) if len(values) > 1 else 0.0
        parts.append(f"{labels[metric]}={values.mean():.3f}+/-{deviation:.3f}")
    return " | ".join(parts)


def plot_bridge(
    *,
    ensembles: tuple[Ensemble, ...],
    order: np.ndarray,
    boundaries: np.ndarray,
    protocol: str,
    route_id: str,
    eeg_branch: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(order), dtype=np.int64)
    figure, axes = plt.subplots(3, 1, figsize=(18, 14), dpi=180, sharex=True, constrained_layout=True)
    figure.suptitle(
        "Matched Event Bridge: Window Mainline vs Daily-Affect\n"
        f"{protocol} | {route_id} | {eeg_branch} | {ensembles[0].seed_scores.shape[0]} matched seeds",
        fontsize=17,
        fontweight="bold",
    )
    for axis, ensemble, (_, _, label, color) in zip(axes, ensembles, PANEL_SPECS):
        seed_scores = ensemble.seed_scores[:, order]
        mean_score = seed_scores.mean(axis=0)
        lower, upper = np.quantile(seed_scores, (0.10, 0.90), axis=0)
        axis.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0, label="Seed 10-90% interval")
        axis.step(x, ensemble.label[order], where="mid", color="#202124", linewidth=1.2, label="True ordinal label")
        axis.plot(x, mean_score, color=color, linewidth=1.7, label="Mean expected score")
        for boundary in boundaries:
            axis.axvline(boundary, color="#9AA0A6", linewidth=0.45, alpha=0.35)
        axis.set_ylim(0.75, 5.25)
        axis.set_yticks((1, 2, 3, 4, 5))
        axis.set_ylabel("Fatigue score")
        axis.grid(axis="y", color="#DADCE0", linewidth=0.65)
        axis.set_title(
            f"{label}\n{metric_summary(ensemble.metrics_by_seed)}",
            fontsize=10,
            linespacing=1.4,
            pad=10,
        )
        axis.legend(loc="upper right", frameon=True, fontsize=8)
    axes[-1].set_xlabel("Test event order, grouped by subject-day; vertical lines mark group boundaries")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    seeds = parse_seed_csv(args.seeds)
    out_dir = args.out_dir or args.comparison_root
    ensembles = load_ensembles(args.comparison_root, seeds)
    assert_same_events(ensembles)
    order = event_order(ensembles[0])
    boundaries = group_boundaries(ensembles[0], order)
    figure_path = out_dir / f"window_daily_affect_matched_event_series_{args.protocol}.png"
    metadata_path = out_dir / f"window_daily_affect_matched_event_series_{args.protocol}.json"
    plot_bridge(
        ensembles=ensembles,
        order=order,
        boundaries=boundaries,
        protocol=args.protocol,
        route_id=args.route_id,
        eeg_branch=args.eeg_branch,
        out_path=figure_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "protocol": args.protocol,
                "route_id": args.route_id,
                "eeg_branch": args.eeg_branch,
                "seeds": list(seeds),
                "event_count": int(len(order)),
                "subject_day_group_count": int(len(boundaries) + 1),
                "event_order": "subject_id, day_id, natural trailing event number",
                "source": "event_predictions.csv and seed_metrics.csv written by script 84",
                "metrics_by_model": {
                    ensemble.model_id: {key: values.tolist() for key, values in ensemble.metrics_by_seed.items()}
                    for ensemble in ensembles
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"figure={figure_path}")
    print(f"metadata={metadata_path}")
    print(f"event_count={len(order)}")
    print(f"subject_day_group_count={len(boundaries) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
