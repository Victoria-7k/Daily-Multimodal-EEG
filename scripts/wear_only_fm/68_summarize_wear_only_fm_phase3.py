#!/usr/bin/env python3
"""Phase 3 decision summary for the Wear-only FM matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BASELINES = ("Wphysio", "Wdeep", "Wmoment_frozen")
W3FM = "W3FM_frozen"
LABEL_METRIC = {
    "cross_day": "raw_r",
    "date_in_order": "within_subject_centered_r",
    "cross_subject": "raw_r",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", type=Path, default=Path("outputs/wear_fm/phase2"))
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/wear_fm/phase3"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    phase2_report = json.loads((args.phase2_root / "wear_only_fm_phase2_report.json").read_text(encoding="utf-8"))
    predictions = _load_predictions(args.phase2_root, phase2_report)
    summary = _route_summary(phase2_report["results"], phase2_report["protocols"], phase2_report["routes"])
    mean_best = _mean_best_baselines(summary, phase2_report["protocols"])
    paired = _paired_seed_bootstrap(
        phase2_report,
        predictions,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )
    gate = _gate_decision(paired)
    output = {
        "stage": "wear_only_fm_phase3",
        "source_phase2_root": str(args.phase2_root),
        "bootstrap": {
            "unit": "subject_day_block",
            "iterations": int(args.bootstrap_iters),
            "seed": int(args.bootstrap_seed),
            "ci": "2.5-97.5 percentile",
        },
        "baseline_selection": {
            "cross_day": "highest test raw_r among Wphysio/Wdeep/Wmoment_frozen; lower RMSE tie-breaker",
            "date_in_order": "highest test within_subject_centered_r among Wphysio/Wdeep/Wmoment_frozen; lower RMSE tie-breaker",
            "cross_subject": "diagnostic; highest test raw_r among Wphysio/Wdeep/Wmoment_frozen; lower RMSE tie-breaker",
        },
        "summary": summary,
        "mean_best_baselines": mean_best,
        "paired_bootstrap": paired,
        "gate": gate,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "wear_only_fm_phase3_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(output, args.out_root / "wear_only_fm_phase3_report.md")
    _write_csvs(output, args.out_root)
    if args.plot:
        _try_write_plots(args.out_root, phase2_report, predictions)
    print(f"out_json={args.out_root / 'wear_only_fm_phase3_report.json'}")
    print(f"out_md={args.out_root / 'wear_only_fm_phase3_report.md'}")
    print(f"advance_to_second_round={gate['advance_to_second_round']}")
    print(f"decision={gate['decision']}")
    return 0


def _route_summary(results: list[dict[str, Any]], protocols: list[str], routes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        for route in routes:
            group = [row for row in results if row["protocol"] == protocol and row["route"] == route]
            item: dict[str, Any] = {"protocol": protocol, "route": route, "runs": len(group)}
            for metric in ("rmse", "mae", "raw_r", "within_subject_centered_r", "per_subject_r_mean"):
                values = [row["test"].get(metric) for row in group if row["test"].get(metric) is not None]
                item[f"{metric}_mean"] = float(np.mean(values)) if values else None
                item[f"{metric}_std"] = float(np.std(values)) if len(values) > 1 else (0.0 if values else None)
            rows.append(item)
    return rows


def _mean_best_baselines(summary: list[dict[str, Any]], protocols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        metric = LABEL_METRIC.get(protocol, "raw_r")
        candidates = [row for row in summary if row["protocol"] == protocol and row["route"] in BASELINES]
        best = max(candidates, key=lambda row: (_metric_value(row, f"{metric}_mean"), -_metric_value(row, "rmse_mean")))
        rows.append(
            {
                "protocol": protocol,
                "selection_metric": metric,
                "best_baseline": best["route"],
                "best_baseline_metric_mean": best.get(f"{metric}_mean"),
                "best_baseline_rmse_mean": best.get("rmse_mean"),
            }
        )
    return rows


def _paired_seed_bootstrap(
    report: dict[str, Any],
    predictions: dict[tuple[str, str, int], dict[str, Any]],
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in report["protocols"]:
        metric = LABEL_METRIC.get(protocol, "raw_r")
        for seed in report["seeds"]:
            baseline = _best_seed_baseline(report["results"], protocol, int(seed), metric)
            base_pred = predictions[(protocol, baseline["route"], int(seed))]
            w3_pred = predictions[(protocol, W3FM, int(seed))]
            _assert_same_test_rows(base_pred, w3_pred)
            observed = _metric_deltas(
                target=w3_pred["target"],
                baseline=base_pred["prediction"],
                candidate=w3_pred["prediction"],
                subject_id=w3_pred["subject_id"],
            )
            block_values = _bootstrap_blocks(
                target=w3_pred["target"],
                baseline=base_pred["prediction"],
                candidate=w3_pred["prediction"],
                subject_id=w3_pred["subject_id"],
                day_id=w3_pred["day_id"],
                iterations=bootstrap_iters,
                seed=bootstrap_seed + int(seed) + sum(ord(ch) for ch in protocol),
            )
            rows.append(
                {
                    "protocol": protocol,
                    "seed": int(seed),
                    "selection_metric": metric,
                    "best_baseline": baseline["route"],
                    "block_count": int(block_values["block_count"]),
                    "test_count": int(w3_pred["target"].shape[0]),
                    "observed": observed,
                    "bootstrap": {
                        name: _bootstrap_summary(values)
                        for name, values in block_values["deltas"].items()
                    },
                }
            )
    return rows


def _best_seed_baseline(results: list[dict[str, Any]], protocol: str, seed: int, metric: str) -> dict[str, Any]:
    candidates = [
        row
        for row in results
        if row["protocol"] == protocol and int(row["seed"]) == int(seed) and row["route"] in BASELINES
    ]
    return max(candidates, key=lambda row: (_metric_value(row["test"], metric), -_metric_value(row["test"], "rmse")))


def _load_predictions(phase2_root: Path, report: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for result in report["results"]:
        protocol = result["protocol"]
        route = result["route"]
        seed = int(result["seed"])
        path = phase2_root / "runs" / protocol / route / f"seed_{seed}" / "predictions.npz"
        with np.load(path, allow_pickle=True) as loaded:
            test_index = loaded["test_index"].astype(np.int64)
            out[(protocol, route, seed)] = {
                "path": str(path),
                "test_index": test_index,
                "sample_id": loaded["sample_id"].astype(str)[test_index],
                "subject_id": loaded["subject_id"].astype(str)[test_index],
                "day_id": loaded["day_id"].astype(str)[test_index],
                "target": loaded["target"].astype(np.float32)[test_index],
                "prediction": loaded["test_prediction"].astype(np.float32),
            }
    return out


def _assert_same_test_rows(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("test_index", "sample_id", "subject_id", "day_id", "target"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"paired prediction mismatch for {key}: {left['path']} vs {right['path']}")
    if left["prediction"].shape != right["prediction"].shape:
        raise ValueError("paired prediction arrays have different shapes")


def _bootstrap_blocks(
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    subject_id: np.ndarray,
    day_id: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, (subject, day) in enumerate(zip(subject_id.tolist(), day_id.tolist())):
        groups.setdefault(f"{subject}::{day}", []).append(index)
    block_arrays = [np.asarray(values, dtype=np.int64) for values in groups.values()]
    rng = np.random.default_rng(seed)
    deltas = {"raw_r": [], "rmse": [], "within_subject_centered_r": []}
    for _ in range(max(1, iterations)):
        selected = rng.integers(0, len(block_arrays), size=len(block_arrays))
        indices = np.concatenate([block_arrays[i] for i in selected])
        values = _metric_deltas(
            target=target[indices],
            baseline=baseline[indices],
            candidate=candidate[indices],
            subject_id=subject_id[indices],
        )
        for key in deltas:
            value = values[key]
            if value is not None and math.isfinite(float(value)):
                deltas[key].append(float(value))
    return {"block_count": len(block_arrays), "deltas": {key: np.asarray(value, dtype=np.float64) for key, value in deltas.items()}}


def _metric_deltas(
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    subject_id: np.ndarray,
) -> dict[str, float | None]:
    base = _metrics(target, baseline, subject_id)
    cand = _metrics(target, candidate, subject_id)
    return {
        "raw_r": _delta(cand["raw_r"], base["raw_r"]),
        "rmse": _delta(cand["rmse"], base["rmse"]),
        "within_subject_centered_r": _delta(cand["within_subject_centered_r"], base["within_subject_centered_r"]),
    }


def _metrics(target: np.ndarray, prediction: np.ndarray, subject_id: np.ndarray) -> dict[str, float | None]:
    err = prediction.astype(np.float64) - target.astype(np.float64)
    centered_target, centered_pred = _centered_arrays(target, prediction, subject_id)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "raw_r": _safe_corr(target, prediction),
        "within_subject_centered_r": _safe_corr(centered_target, centered_pred),
    }


def _centered_arrays(target: np.ndarray, prediction: np.ndarray, subject_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = target.astype(np.float64).copy()
    pred = prediction.astype(np.float64).copy()
    subjects = subject_id.astype(str)
    for subject in np.unique(subjects):
        mask = subjects == subject
        true[mask] -= true[mask].mean()
        pred[mask] -= pred[mask].mean()
    return true, pred


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    x = left.astype(np.float64).reshape(-1)
    y = right.astype(np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0.0:
        return None
    return float(np.sum(x * y) / denom)


def _bootstrap_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean": None, "ci_low": None, "ci_high": None, "p_gt_0": None, "p_lt_0": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_gt_0": float(np.mean(values > 0.0)),
        "p_lt_0": float(np.mean(values < 0.0)),
        "p_lte_0": float(np.mean(values <= 0.0)),
        "p_gte_0": float(np.mean(values >= 0.0)),
    }


def _gate_decision(paired: list[dict[str, Any]]) -> dict[str, Any]:
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for row in paired:
        by_protocol.setdefault(row["protocol"], []).append(row)
    cross_day = by_protocol.get("cross_day", [])
    within = by_protocol.get("date_in_order", [])
    cross_day_raw = [row["observed"]["raw_r"] for row in cross_day]
    cross_day_rmse = [row["observed"]["rmse"] for row in cross_day]
    within_centered = [row["observed"]["within_subject_centered_r"] for row in within]
    within_rmse = [row["observed"]["rmse"] for row in within]
    checks = {
        "cross_day_raw_r_mean_gt_0": _mean(cross_day_raw) is not None and _mean(cross_day_raw) > 0.0,
        "cross_day_raw_r_positive_seed_count_ge_2": _positive_count(cross_day_raw) >= 2,
        "cross_day_rmse_mean_lte_0": _mean(cross_day_rmse) is not None and _mean(cross_day_rmse) <= 0.0,
        "cross_day_rmse_nonworse_seed_count_ge_2": _nonpositive_count(cross_day_rmse) >= 2,
        "within_subject_centered_r_mean_gte_0": _mean(within_centered) is not None and _mean(within_centered) >= 0.0,
        "within_subject_rmse_no_obvious_retreat_mean_lte_0p015": _mean(within_rmse) is not None and _mean(within_rmse) <= 0.015,
    }
    advance = all(checks.values())
    return {
        "advance_to_second_round": bool(advance),
        "decision": "advance_to_second_round_ablation" if advance else "stop_after_phase3_do_not_run_second_round_ablation",
        "checks": checks,
        "observed_delta_means": {
            "cross_day_raw_r": _mean(cross_day_raw),
            "cross_day_rmse": _mean(cross_day_rmse),
            "within_subject_centered_r": _mean(within_centered),
            "within_subject_rmse": _mean(within_rmse),
        },
        "direction_counts": {
            "cross_day_raw_r_positive": _positive_count(cross_day_raw),
            "cross_day_rmse_nonworse": _nonpositive_count(cross_day_rmse),
            "within_subject_centered_r_positive": _positive_count(within_centered),
            "within_subject_rmse_nonworse": _nonpositive_count(within_rmse),
        },
    }


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    gate = output["gate"]
    lines = [
        "# Wear-only FM Phase 3 Decision",
        "",
        f"- decision: `{gate['decision']}`",
        f"- advance_to_second_round: `{gate['advance_to_second_round']}`",
        f"- bootstrap: subject-day block, `{output['bootstrap']['iterations']}` iterations",
        "",
        "## Main Result Mean +/- Std",
        "",
        "| protocol | route | runs | RMSE | raw r | centered r |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in output["summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['runs']} | "
            f"{_mean_std(row, 'rmse')} | {_mean_std(row, 'raw_r')} | {_mean_std(row, 'within_subject_centered_r')} |"
        )
    lines.extend(["", "## Mean-Level Best Baselines", "", "| protocol | metric | best baseline | metric mean | RMSE mean |", "| --- | --- | --- | ---: | ---: |"])
    for row in output["mean_best_baselines"]:
        lines.append(
            f"| {row['protocol']} | {row['selection_metric']} | {row['best_baseline']} | "
            f"{_fmt(row['best_baseline_metric_mean'])} | {_fmt(row['best_baseline_rmse_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## W3FM Vs Seed-Level Best Baseline",
            "",
            "| protocol | seed | best baseline | block count | delta raw r | raw r CI | delta RMSE | RMSE CI | delta centered r | centered r CI |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in output["paired_bootstrap"]:
        obs = row["observed"]
        boot = row["bootstrap"]
        lines.append(
            f"| {row['protocol']} | {row['seed']} | {row['best_baseline']} | {row['block_count']} | "
            f"{_fmt(obs['raw_r'])} | {_ci(boot['raw_r'])} | {_fmt(obs['rmse'])} | {_ci(boot['rmse'])} | "
            f"{_fmt(obs['within_subject_centered_r'])} | {_ci(boot['within_subject_centered_r'])} |"
        )
    lines.extend(["", "## Gate Checks", "", "| check | pass |", "| --- | ---: |"])
    for key, passed in gate["checks"].items():
        lines.append(f"| {key} | {passed} |")
    lines.extend(["", "## Decision", ""])
    if gate["advance_to_second_round"]:
        lines.append("W3FM_frozen passes the registered Phase 3 gate and can enter the second-round ablation.")
    else:
        lines.append("W3FM_frozen does not pass the registered Phase 3 gate. The second-round ablation should not be run from this evidence.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csvs(output: dict[str, Any], out_root: Path) -> None:
    _write_dict_rows(out_root / "phase3_route_summary.csv", output["summary"])
    flat = []
    for row in output["paired_bootstrap"]:
        item = {
            "protocol": row["protocol"],
            "seed": row["seed"],
            "selection_metric": row["selection_metric"],
            "best_baseline": row["best_baseline"],
            "block_count": row["block_count"],
            "test_count": row["test_count"],
        }
        for metric in ("raw_r", "rmse", "within_subject_centered_r"):
            item[f"observed_delta_{metric}"] = row["observed"][metric]
            for key, value in row["bootstrap"][metric].items():
                item[f"bootstrap_{metric}_{key}"] = value
        flat.append(item)
    _write_dict_rows(out_root / "phase3_paired_bootstrap.csv", flat)


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _try_write_plots(out_root: Path, report: dict[str, Any], predictions: dict[tuple[str, str, int], dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_root / "plot_status.txt").write_text(f"matplotlib unavailable: {exc!r}\n", encoding="utf-8")
        return
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for protocol in report["protocols"]:
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        for ax, route in zip(axes.reshape(-1), report["routes"]):
            for seed in report["seeds"]:
                pred = predictions[(protocol, route, int(seed))]
                target = pred["target"]
                values = pred["prediction"]
                if target.size > 1200:
                    step = max(1, target.size // 1200)
                    target = target[::step]
                    values = values[::step]
                ax.scatter(target, values, s=4, alpha=0.18, label=str(seed))
            ax.set_title(route)
            ax.set_xlabel("target")
            ax.set_ylabel("prediction")
        fig.savefig(fig_dir / f"{protocol}_prediction_scatter.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        labels = []
        values = []
        for route in report["routes"]:
            route_values = []
            for row in report["results"]:
                if row["protocol"] == protocol and row["route"] == route:
                    route_values.extend(
                        float(item["pearson_r"])
                        for item in row["test"]["per_subject_r"]["subjects"]
                        if item.get("pearson_r") is not None
                    )
            labels.append(route)
            values.append(route_values)
        try:
            ax.boxplot(values, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(values, showfliers=False)
            ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.set_ylabel("per-subject r")
        ax.set_title(f"{protocol} per-subject r distribution")
        fig.savefig(fig_dir / f"{protocol}_per_subject_r_distribution.png", dpi=160)
        plt.close(fig)


def _write_json_debug(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return -999.0
    return float(value)


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _positive_count(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None and float(value) > 0.0)


def _nonpositive_count(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None and float(value) <= 0.0)


def _mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "NA"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _ci(row: dict[str, Any]) -> str:
    if row.get("ci_low") is None:
        return "NA"
    return f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]"


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
