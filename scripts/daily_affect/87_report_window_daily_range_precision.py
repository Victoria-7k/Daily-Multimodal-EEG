#!/usr/bin/env python3
"""Audit prediction range expansion against continuous precision in the matched event bridge."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("outputs/daily_affect_window_event_bridge_20260906/comparison")
DEFAULT_SEEDS = (240729, 240730, 240731)
MODEL_SPECS = (
    ("window_regression", "attention_regression", "Window regression"),
    ("daily_affect", "bag_static", "Daily-affect bag static"),
    ("daily_affect", "dynamic_kernel_prior_uniform", "Daily-affect dynamic kernel + prior"),
)
MODEL_LABELS = {(system, model_id): label for system, model_id, label in MODEL_SPECS}
SUMMARY_KEYS = (
    "prediction_std",
    "p10_p90_span",
    "prediction_iqr",
    "raw_r",
    "covariance_with_label",
    "prediction_slope_per_label",
    "label_mean_span",
    "within_label_std",
    "label_explained_variance_fraction",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    return parser.parse_args()


def parse_seed_csv(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def load_ensembles(path: Path, seeds: tuple[int, ...]) -> tuple[Ensemble, ...]:
    rows_by_model: dict[tuple[str, str], dict[int, dict[str, dict[str, str]]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["system"], row["model_id"])
            if key not in MODEL_LABELS:
                continue
            seed = int(row["seed"])
            if seed not in seeds:
                continue
            events = rows_by_model.setdefault(key, {}).setdefault(seed, {})
            event_id = row["event_id"]
            if event_id in events:
                raise ValueError(f"duplicate prediction for {key}, seed {seed}, event {event_id}")
            events[event_id] = row
    ensembles = tuple(
        build_ensemble(rows_by_model, system, model_id, seeds)
        for system, model_id, _ in MODEL_SPECS
    )
    assert_same_events(ensembles)
    return ensembles


def build_ensemble(
    rows_by_model: dict[tuple[str, str], dict[int, dict[str, dict[str, str]]]],
    system: str,
    model_id: str,
    seeds: tuple[int, ...],
) -> Ensemble:
    key = (system, model_id)
    per_seed = rows_by_model.get(key, {})
    missing = [seed for seed in seeds if seed not in per_seed]
    if missing:
        raise ValueError(f"{key} is missing seeds {missing}")
    reference = per_seed[seeds[0]]
    event_ids = tuple(reference)
    for seed in seeds[1:]:
        if set(per_seed[seed]) != set(event_ids):
            raise ValueError(f"{key} seed {seed} has a different event set")
        for event_id in event_ids:
            for field in ("subject_id", "day_id", "label"):
                if per_seed[seed][event_id][field] != reference[event_id][field]:
                    raise ValueError(f"{key} seed {seed} disagrees on {event_id} {field}")
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
    )


def assert_same_events(ensembles: tuple[Ensemble, ...]) -> None:
    first = ensembles[0]
    for ensemble in ensembles[1:]:
        for field in ("event_id", "subject_id", "day_id", "label"):
            if not np.array_equal(getattr(first, field), getattr(ensemble, field)):
                raise ValueError(f"{first.model_id} and {ensemble.model_id} differ on {field}")


def prediction_statistics(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    label = np.asarray(label, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if label.size != score.size or not label.size:
        raise ValueError("label and score must have the same nonzero length")
    prediction_variance = float(np.var(score))
    label_variance = float(np.var(label))
    covariance = float(np.mean((label - label.mean()) * (score - score.mean())))
    raw_r = covariance / math.sqrt(label_variance * prediction_variance)
    class_means: list[float] = []
    between_variance = 0.0
    for ordinal_label in range(1, 6):
        selected = score[label == ordinal_label]
        if not selected.size:
            continue
        mean = float(selected.mean())
        class_means.append(mean)
        between_variance += selected.size * (mean - score.mean()) ** 2
    between_variance /= score.size
    return {
        "prediction_std": float(math.sqrt(prediction_variance)),
        "p10_p90_span": float(np.quantile(score, 0.90) - np.quantile(score, 0.10)),
        "prediction_iqr": float(np.quantile(score, 0.75) - np.quantile(score, 0.25)),
        "raw_r": float(raw_r),
        "covariance_with_label": covariance,
        "prediction_slope_per_label": float(covariance / label_variance),
        "label_mean_span": float(max(class_means) - min(class_means)),
        "within_label_std": float(math.sqrt(max(prediction_variance - between_variance, 0.0))),
        "label_explained_variance_fraction": float(between_variance / prediction_variance),
    }


def label_statistics(label: np.ndarray) -> dict[str, Any]:
    return {
        "event_count": int(label.size),
        "label_mean": float(np.mean(label)),
        "label_std": float(np.std(label)),
        "label_p10_p90_span": float(np.quantile(label, 0.90) - np.quantile(label, 0.10)),
        "label_iqr": float(np.quantile(label, 0.75) - np.quantile(label, 0.25)),
        "class_count": {str(value): int(np.sum(label == value)) for value in range(1, 6)},
    }


def bootstrap_pair_deltas(
    *,
    label: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    subject_id: np.ndarray,
    day_id: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float | int | str]:
    if iterations <= 0:
        return {"bootstrap_status": "disabled", "bootstrap_iterations": 0}
    blocks = np.asarray([f"{subject}::{day}" for subject, day in zip(subject_id, day_id)], dtype=str)
    unique = np.unique(blocks)
    indices = [np.flatnonzero(blocks == block) for block in unique]
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {key: [] for key in SUMMARY_KEYS}
    for _ in range(iterations):
        selected = np.concatenate([indices[index] for index in rng.integers(0, len(indices), size=len(indices))])
        first_stats = prediction_statistics(label[selected], first[selected])
        second_stats = prediction_statistics(label[selected], second[selected])
        for key in SUMMARY_KEYS:
            deltas[key].append(second_stats[key] - first_stats[key])
    output: dict[str, float | int | str] = {
        "bootstrap_status": "ok",
        "bootstrap_unit": "subject_day",
        "bootstrap_iterations": int(iterations),
        "bootstrap_block_count": int(len(indices)),
    }
    for key, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        output[f"bootstrap_delta_{key}_ci_low"] = float(np.quantile(array, 0.025))
        output[f"bootstrap_delta_{key}_ci_high"] = float(np.quantile(array, 0.975))
    return output


def mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not values.size:
        return None, None
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def format_value(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, output: dict[str, Any]) -> None:
    lines = [
        "# Matched Event Dynamic-Range and Precision Audit",
        "",
        "All rows use the same `cross_day` held-out events, partial-FT EEG token, route and fusion seeds.",
        "Prediction range uses robust P10--P90 and IQR spans. `label_explained_variance_fraction` is the fraction of prediction variance attributable to differences among true 1--5 label groups; remaining variance is within-label variation.",
        "",
        "## Held-Out Label Distribution",
        "",
        f"- events: `{output['label_statistics']['event_count']}`",
        f"- counts for labels 1..5: `{output['label_statistics']['class_count']}`",
        f"- true-label SD: `{format_value(output['label_statistics']['label_std'])}`; P10--P90 span: `{format_value(output['label_statistics']['label_p10_p90_span'])}`",
        "",
        "## Per-System Seed Mean",
        "",
        "| system | prediction SD | P10--P90 | IQR | raw r | label-mean span | within-label SD | label-explained variance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["system_summary"]:
        lines.append(
            f"| {row['label']} | {format_value(row['prediction_std_mean'])}+/-{format_value(row['prediction_std_std'])} | "
            f"{format_value(row['p10_p90_span_mean'])}+/-{format_value(row['p10_p90_span_std'])} | "
            f"{format_value(row['prediction_iqr_mean'])}+/-{format_value(row['prediction_iqr_std'])} | "
            f"{format_value(row['raw_r_mean'])}+/-{format_value(row['raw_r_std'])} | "
            f"{format_value(row['label_mean_span_mean'])}+/-{format_value(row['label_mean_span_std'])} | "
            f"{format_value(row['within_label_std_mean'])}+/-{format_value(row['within_label_std_std'])} | "
            f"{format_value(row['label_explained_variance_fraction_mean'])}+/-{format_value(row['label_explained_variance_fraction_std'])} |"
        )
    lines.extend([
        "",
        "## Paired Daily Minus Window",
        "",
        "Positive range deltas indicate wider daily predictions. Positive raw-r and label-explained-variance deltas indicate stronger continuous label tracking.",
        "",
        "| daily system | delta P10--P90 | range wins | delta raw r | raw-r wins | delta label-explained variance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in output["paired_summary"]:
        lines.append(
            f"| {row['daily_label']} | {format_value(row['mean_delta_p10_p90_span'])}+/-{format_value(row['std_delta_p10_p90_span'])} | "
            f"{row['daily_win_count_p10_p90_span']}/{row['seed_count']} | "
            f"{format_value(row['mean_delta_raw_r'])}+/-{format_value(row['std_delta_raw_r'])} | "
            f"{row['daily_win_count_raw_r']}/{row['seed_count']} | "
            f"{format_value(row['mean_delta_label_explained_variance_fraction'])}+/-{format_value(row['std_delta_label_explained_variance_fraction'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    seeds = parse_seed_csv(args.seeds)
    out_root = args.out_root or args.comparison_root
    ensembles = load_ensembles(args.comparison_root / "event_predictions.csv", seeds)
    per_seed: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for ensemble in ensembles:
        rows: list[dict[str, Any]] = []
        for seed, score in zip(seeds, ensemble.seed_scores):
            row = {
                "seed": seed,
                "system": ensemble.system,
                "model_id": ensemble.model_id,
                "label": MODEL_LABELS[(ensemble.system, ensemble.model_id)],
                **prediction_statistics(ensemble.label, score),
            }
            per_seed.append(row)
            rows.append(row)
        summary = {"system": ensemble.system, "model_id": ensemble.model_id, "label": MODEL_LABELS[(ensemble.system, ensemble.model_id)]}
        for key in SUMMARY_KEYS:
            mean, std = mean_std(rows, key)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        summaries.append(summary)

    window = ensembles[0]
    paired_rows: list[dict[str, Any]] = []
    for daily in ensembles[1:]:
        for offset, (seed, window_score, daily_score) in enumerate(zip(seeds, window.seed_scores, daily.seed_scores)):
            window_stats = prediction_statistics(window.label, window_score)
            daily_stats = prediction_statistics(daily.label, daily_score)
            row = {
                "seed": seed,
                "daily_system": daily.system,
                "daily_model_id": daily.model_id,
                "daily_label": MODEL_LABELS[(daily.system, daily.model_id)],
                **{f"delta_{key}": daily_stats[key] - window_stats[key] for key in SUMMARY_KEYS},
                **bootstrap_pair_deltas(
                    label=window.label,
                    first=window_score,
                    second=daily_score,
                    subject_id=window.subject_id,
                    day_id=window.day_id,
                    iterations=args.bootstrap_iters,
                    seed=args.bootstrap_seed + offset + seed,
                ),
            }
            paired_rows.append(row)

    paired_summary: list[dict[str, Any]] = []
    for daily in ensembles[1:]:
        rows = [row for row in paired_rows if row["daily_model_id"] == daily.model_id]
        summary = {"daily_model_id": daily.model_id, "daily_label": MODEL_LABELS[(daily.system, daily.model_id)], "seed_count": len(rows)}
        for key in SUMMARY_KEYS:
            mean, std = mean_std(rows, f"delta_{key}")
            summary[f"mean_delta_{key}"] = mean
            summary[f"std_delta_{key}"] = std
            summary[f"daily_win_count_{key}"] = int(sum(float(row[f"delta_{key}"]) > 0 for row in rows))
        paired_summary.append(summary)

    output = {
        "script": Path(__file__).name,
        "comparison_status": "diagnostic_on_exploratory_three_seed_matched_event_bridge",
        "contract": {
            "source": "event_predictions.csv written by 84_compare_window_daily_event_level.py",
            "seed_count": len(seeds),
            "bootstrap_unit": "subject_day",
            "bootstrap_iterations": args.bootstrap_iters,
            "daily_minus_window": True,
        },
        "label_statistics": label_statistics(window.label),
        "per_seed": per_seed,
        "system_summary": summaries,
        "paired_per_seed": paired_rows,
        "paired_summary": paired_summary,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "window_daily_range_precision_audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(out_root / "window_daily_range_precision_by_seed.csv", per_seed)
    write_csv(out_root / "window_daily_range_precision_paired.csv", paired_rows)
    write_markdown(out_root / "window_daily_range_precision_audit.md", output)
    print(f"per_seed_row_count={len(per_seed)}")
    print(f"paired_row_count={len(paired_rows)}")
    print(f"out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
