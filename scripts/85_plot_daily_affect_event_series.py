#!/usr/bin/env python3
"""Render a seed-aware held-out event sequence for the current daily-affect route."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEEDS = (240729, 240730, 240731, 240732, 240733, 240734, 240735)
METRIC_KEYS = ("qwk", "expected_raw_r", "expected_within_subject_centered_r", "expected_rmse")
EVENT_NUMBER = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class ModelEnsemble:
    model_id: str
    event_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    label: np.ndarray
    seed_scores: np.ndarray
    metrics_by_seed: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/daily_affect_dynamic_a1_crossday_routefix_20260906/norm_per_modality__adapter_per_modality"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--normalization", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument(
        "--candidate-model",
        choices=("dynamic_kernel_prior_uniform",),
        default="dynamic_kernel_prior_uniform",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    return parser.parse_args()


def parse_seed_csv(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def condition_id(normalization: str, adapter_mode: str) -> str:
    return f"norm_{normalization}__adapter_{adapter_mode}"


def prediction_path(
    run_root: Path,
    model_id: str,
    protocol: str,
    route_id: str,
    condition: str,
    seed: int,
) -> Path:
    if model_id == "bag_static":
        return run_root / "phase0" / protocol / route_id / condition / f"seed_{seed}" / "predictions.npz"
    return run_root / "runs" / protocol / route_id / model_id / condition / f"seed_{seed}" / "predictions.npz"


def load_test_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        test_index = np.asarray(values["test_index"], dtype=np.int64)
        return {
            "event_id": np.asarray(values["event_id"])[test_index],
            "subject_id": np.asarray(values["subject_id"])[test_index],
            "day_id": np.asarray(values["day_id"])[test_index],
            "label": np.asarray(values["label"], dtype=np.float64)[test_index],
            "score": np.asarray(values["test_expected_score"], dtype=np.float64),
        }


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("test", payload)
    return {key: float(test[key]) for key in METRIC_KEYS if key in test and test[key] is not None}


def load_ensemble(
    *,
    run_root: Path,
    model_id: str,
    protocol: str,
    route_id: str,
    condition: str,
    seeds: tuple[int, ...],
) -> ModelEnsemble:
    reference: dict[str, np.ndarray] | None = None
    score_rows: list[np.ndarray] = []
    metric_rows: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for seed in seeds:
        path = prediction_path(run_root, model_id, protocol, route_id, condition, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        current = load_test_prediction(path)
        if reference is None:
            reference = current
        else:
            for key in ("event_id", "subject_id", "day_id", "label"):
                if not np.array_equal(reference[key], current[key]):
                    raise ValueError(f"{model_id} seed {seed} differs in test {key}")
        score_rows.append(current["score"])
        for key, value in load_metrics(path.with_name("metrics.json")).items():
            metric_rows[key].append(value)
    if reference is None:
        raise ValueError(f"no predictions loaded for {model_id}")
    return ModelEnsemble(
        model_id=model_id,
        event_id=reference["event_id"],
        subject_id=reference["subject_id"],
        day_id=reference["day_id"],
        label=reference["label"],
        seed_scores=np.stack(score_rows, axis=0),
        metrics_by_seed={key: np.asarray(values, dtype=np.float64) for key, values in metric_rows.items() if values},
    )


def trailing_event_number(event_id: str) -> int:
    match = EVENT_NUMBER.search(event_id)
    return int(match.group(1)) if match else -1


def event_order(ensemble: ModelEnsemble) -> np.ndarray:
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


def assert_same_events(first: ModelEnsemble, second: ModelEnsemble) -> None:
    for key in ("event_id", "subject_id", "day_id", "label"):
        if not np.array_equal(getattr(first, key), getattr(second, key)):
            raise ValueError(f"model test events disagree on {key}")


def group_boundaries(ensemble: ModelEnsemble, order: np.ndarray) -> np.ndarray:
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
    for key in METRIC_KEYS:
        values = metrics_by_seed.get(key)
        if values is None or not len(values):
            continue
        deviation = values.std(ddof=1) if len(values) > 1 else 0.0
        parts.append(f"{labels[key]}={values.mean():.3f}+/-{deviation:.3f}")
    return " | ".join(parts)


def plot_ensemble(
    *,
    static: ModelEnsemble,
    candidate: ModelEnsemble,
    order: np.ndarray,
    boundaries: np.ndarray,
    protocol: str,
    route_id: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    panels = (
        (static, "Bag static", "#0072B2"),
        (candidate, "Dynamic kernel + prior", "#D55E00"),
    )
    x = np.arange(len(order), dtype=np.int64)
    figure, axes = plt.subplots(2, 1, figsize=(18, 10), dpi=180, sharex=True, constrained_layout=True)
    figure.suptitle(
        "Current Daily-Affect Candidate: True vs Predicted Held-Out Event Sequence\n"
        f"{protocol} | {route_id} | {static.seed_scores.shape[0]} matched seeds",
        fontsize=17,
        fontweight="bold",
    )
    for axis, (ensemble, label, color) in zip(axes, panels):
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
        axis.set_title(f"{label}: {metric_summary(ensemble.metrics_by_seed)}", fontsize=10, pad=9)
        axis.legend(loc="upper right", frameon=True, fontsize=8)
    axes[-1].set_xlabel("Test event order, grouped by subject-day; vertical lines mark group boundaries")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def write_event_csv(path: Path, *, static: ModelEnsemble, candidate: ModelEnsemble, order: np.ndarray) -> None:
    static_scores = static.seed_scores[:, order]
    candidate_scores = candidate.seed_scores[:, order]
    fields = (
        "event_order",
        "event_id",
        "subject_id",
        "day_id",
        "true_label",
        "bag_static_mean_expected_score",
        "bag_static_p10_expected_score",
        "bag_static_p90_expected_score",
        "dynamic_kernel_prior_uniform_mean_expected_score",
        "dynamic_kernel_prior_uniform_p10_expected_score",
        "dynamic_kernel_prior_uniform_p90_expected_score",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, index in enumerate(order):
            writer.writerow(
                {
                    "event_order": int(position),
                    "event_id": str(static.event_id[index]),
                    "subject_id": str(static.subject_id[index]),
                    "day_id": str(static.day_id[index]),
                    "true_label": float(static.label[index]),
                    "bag_static_mean_expected_score": float(static_scores[:, position].mean()),
                    "bag_static_p10_expected_score": float(np.quantile(static_scores[:, position], 0.10)),
                    "bag_static_p90_expected_score": float(np.quantile(static_scores[:, position], 0.90)),
                    "dynamic_kernel_prior_uniform_mean_expected_score": float(candidate_scores[:, position].mean()),
                    "dynamic_kernel_prior_uniform_p10_expected_score": float(np.quantile(candidate_scores[:, position], 0.10)),
                    "dynamic_kernel_prior_uniform_p90_expected_score": float(np.quantile(candidate_scores[:, position], 0.90)),
                }
            )


def main() -> int:
    args = parse_args()
    seeds = parse_seed_csv(args.seeds)
    condition = condition_id(args.normalization, args.adapter_mode)
    out_dir = args.out_dir or args.run_root / "figures_7seed"
    static = load_ensemble(
        run_root=args.run_root,
        model_id="bag_static",
        protocol=args.protocol,
        route_id=args.route_id,
        condition=condition,
        seeds=seeds,
    )
    candidate = load_ensemble(
        run_root=args.run_root,
        model_id=args.candidate_model,
        protocol=args.protocol,
        route_id=args.route_id,
        condition=condition,
        seeds=seeds,
    )
    assert_same_events(static, candidate)
    order = event_order(static)
    boundaries = group_boundaries(static, order)
    figure_path = out_dir / f"daily_affect_held_out_event_series_{args.protocol}.png"
    csv_path = out_dir / f"daily_affect_held_out_event_series_{args.protocol}.csv"
    metadata_path = out_dir / f"daily_affect_held_out_event_series_{args.protocol}.json"
    plot_ensemble(
        static=static,
        candidate=candidate,
        order=order,
        boundaries=boundaries,
        protocol=args.protocol,
        route_id=args.route_id,
        out_path=figure_path,
    )
    write_event_csv(csv_path, static=static, candidate=candidate, order=order)
    metadata_path.write_text(
        json.dumps(
            {
                "protocol": args.protocol,
                "route_id": args.route_id,
                "normalization": args.normalization,
                "adapter_mode": args.adapter_mode,
                "candidate_model": args.candidate_model,
                "seeds": list(seeds),
                "event_count": int(len(order)),
                "subject_day_group_count": int(len(boundaries) + 1),
                "event_order": "subject_id, day_id, natural trailing event number",
                "metric_summary": {
                    "bag_static": {key: values.tolist() for key, values in static.metrics_by_seed.items()},
                    args.candidate_model: {key: values.tolist() for key, values in candidate.metrics_by_seed.items()},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"figure={figure_path}")
    print(f"event_csv={csv_path}")
    print(f"metadata={metadata_path}")
    print(f"event_count={len(order)}")
    print(f"subject_day_group_count={len(boundaries) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
