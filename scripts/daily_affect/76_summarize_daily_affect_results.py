#!/usr/bin/env python3
"""Summarize daily-affect phase0 and state-matrix metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from daily_multimodal.daily_affect.metrics import classification_metrics, regression_bridge_metrics


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_RUN_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_routefix_20260906"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "reports")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    args = parser.parse_args()

    rows = load_metric_rows(args.run_root)
    summary_rows = summarize(rows)
    paired_rows = paired_deltas(rows, bootstrap_iters=args.bootstrap_iters, bootstrap_seed=args.bootstrap_seed)
    gate_rows = gate_summary(paired_rows)
    output = {
        "script": Path(__file__).name,
        "run_root": str(args.run_root),
        "run_count": len(rows),
        "summary": summary_rows,
        "paired_delta_summary": paired_rows,
        "gate_summary": gate_rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_json = args.out_root / "daily_affect_ordinal_summary.json"
    out_md = args.out_root / "daily_affect_ordinal_summary.md"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_root / "protocol_route_summary.csv", summary_rows)
    write_csv(args.out_root / "paired_delta_summary.csv", paired_rows)
    write_csv(args.out_root / "gate_summary.csv", gate_rows)
    write_markdown(output, out_md)
    print(f"run_count={len(rows)}")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    return 0


def load_metric_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "protocol" not in item or "test" not in item:
            continue
        item = dict(item)
        item["metrics_path"] = str(path)
        augment_bridge_metrics(item, path)
        rows.append(item)
    return rows


def augment_bridge_metrics(item: dict[str, Any], metrics_path: Path) -> None:
    """Backfill bridge metrics from saved event-level predictions when needed."""

    prediction_path = Path(item.get("prediction_path", ""))
    if not prediction_path.is_file():
        prediction_path = metrics_path.parent / "predictions.npz"
    if not prediction_path.is_file():
        return
    try:
        with np.load(prediction_path, allow_pickle=True) as data:
            labels = data["label"]
            subjects = data["subject_id"]
            for split_name in ("train", "val", "test"):
                split_metrics = item.get(split_name)
                if not isinstance(split_metrics, dict):
                    continue
                required = {"expected_rmse", "expected_raw_r", "expected_within_subject_centered_r"}
                if required.issubset(split_metrics):
                    continue
                indices = data[f"{split_name}_index"]
                score = data[f"{split_name}_expected_score"]
                split_metrics.update(regression_bridge_metrics(labels[indices], score, subject_ids=subjects[indices]))
    except (KeyError, OSError, ValueError):
        return


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = experiment_key(row, include_model=True)
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (protocol, route_id, model_id, experiment, normalization, adapter_mode, objective, routing), values in sorted(grouped.items()):
        out.append(
            {
                "protocol": protocol,
                "route_id": route_id,
                "model_id": model_id,
                "experiment_id": experiment,
                "normalization": normalization,
                "adapter_mode": adapter_mode,
                "objective_id": objective,
                "routing_id": routing,
                "seed_count": len(values),
                "seeds": ",".join(str(row.get("seed")) for row in values),
                "test_qwk_mean": stat(values, "qwk", "mean"),
                "test_qwk_std": stat(values, "qwk", "std"),
                "test_macro_f1_mean": stat(values, "macro_f1", "mean"),
                "test_macro_f1_std": stat(values, "macro_f1", "std"),
                "test_accuracy_mean": stat(values, "accuracy", "mean"),
                "test_accuracy_std": stat(values, "accuracy", "std"),
                "test_ordinal_mae_mean": stat(values, "ordinal_mae", "mean"),
                "test_ordinal_mae_std": stat(values, "ordinal_mae", "std"),
                "test_expected_rmse_mean": stat(values, "expected_rmse", "mean"),
                "test_expected_rmse_std": stat(values, "expected_rmse", "std"),
                "test_expected_raw_r_mean": stat(values, "expected_raw_r", "mean"),
                "test_expected_raw_r_std": stat(values, "expected_raw_r", "std"),
                "test_expected_within_subject_centered_r_mean": stat(values, "expected_within_subject_centered_r", "mean"),
                "test_expected_within_subject_centered_r_std": stat(values, "expected_within_subject_centered_r", "std"),
            }
        )
    return out


def paired_deltas(
    rows: list[dict[str, Any]],
    *,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = paired_context_key(row) + (int(row.get("seed", 0)),)
        model_id = row.get("model_id", "bag_static")
        if model_id == "bag_static" and key not in by_key:
            by_key[key] = row
    out: list[dict[str, Any]] = []
    for row in rows:
        model_id = row.get("model_id", "bag_static")
        if model_id == "bag_static":
            continue
        key = paired_context_key(row) + (int(row.get("seed", 0)),)
        base = by_key.get(key)
        if base is None:
            continue
        out.append(
            delta_row(
                base,
                row,
                bootstrap=paired_subject_day_bootstrap(
                    base,
                    row,
                    iterations=bootstrap_iters,
                    seed=bootstrap_seed + int(row.get("seed", 0)),
                ),
            )
        )
    return out


def gate_summary(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[
            (
                row["protocol"],
                row["route_id"],
                row["model_id"],
                row["experiment_id"],
                row["normalization"],
                row["adapter_mode"],
                row["objective_id"],
                row["routing_id"],
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (protocol, route_id, model_id, experiment, normalization, adapter_mode, objective, routing), values in sorted(grouped.items()):
        qwk = [float(row["delta_qwk"]) for row in values if row.get("delta_qwk") != ""]
        f1 = [float(row["delta_macro_f1"]) for row in values if row.get("delta_macro_f1") != ""]
        mae = [float(row["delta_ordinal_mae"]) for row in values if row.get("delta_ordinal_mae") != ""]
        rmse = [float(row["delta_expected_rmse"]) for row in values if row.get("delta_expected_rmse") != ""]
        raw_r = [float(row["delta_expected_raw_r"]) for row in values if row.get("delta_expected_raw_r") != ""]
        centered_r = [float(row["delta_expected_within_subject_centered_r"]) for row in values if row.get("delta_expected_within_subject_centered_r") != ""]
        qwk_wins = sum(value > 0 for value in qwk)
        mae_wins = sum(value < 0 for value in mae)
        pass_gate = bool(qwk and mean(qwk) > 0.0 and qwk_wins >= max(1, len(qwk) // 2 + 1) and (not mae or mean(mae) <= 0.0))
        out.append(
            {
                "protocol": protocol,
                "route_id": route_id,
                "model_id": model_id,
                "experiment_id": experiment,
                "normalization": normalization,
                "adapter_mode": adapter_mode,
                "objective_id": objective,
                "routing_id": routing,
                "paired_seed_count": len(values),
                "mean_delta_qwk": safe_mean(qwk),
                "std_delta_qwk": safe_std(qwk),
                "mean_delta_macro_f1": safe_mean(f1),
                "std_delta_macro_f1": safe_std(f1),
                "mean_delta_ordinal_mae": safe_mean(mae),
                "std_delta_ordinal_mae": safe_std(mae),
                "mean_delta_expected_rmse": safe_mean(rmse),
                "std_delta_expected_rmse": safe_std(rmse),
                "mean_delta_expected_raw_r": safe_mean(raw_r),
                "std_delta_expected_raw_r": safe_std(raw_r),
                "mean_delta_expected_within_subject_centered_r": safe_mean(centered_r),
                "std_delta_expected_within_subject_centered_r": safe_std(centered_r),
                "qwk_win_count": qwk_wins,
                "mae_win_count": mae_wins,
                "preliminary_gate": "pass" if pass_gate else "stop_or_continue_diagnostic",
            }
        )
    return out


def delta_row(base: dict[str, Any], row: dict[str, Any], *, bootstrap: dict[str, Any]) -> dict[str, Any]:
    result = {
        "protocol": row["protocol"],
        "route_id": row.get("route_id", row.get("route", "")),
        "model_id": row.get("model_id", ""),
        "experiment_id": experiment_id_label(row),
        "normalization": row.get("normalization", "unknown"),
        "adapter_mode": adapter_mode_label(row),
        "objective_id": str(row.get("objective_id", "legacy_unversioned")),
        "routing_id": str(row.get("routing_id", "legacy_unversioned")),
        "seed": int(row.get("seed", 0)),
        "baseline_model_id": base.get("model_id", "bag_static"),
        "baseline_metrics_path": base.get("metrics_path", ""),
        "candidate_metrics_path": row.get("metrics_path", ""),
        "delta_qwk": delta_metric(base, row, "qwk"),
        "delta_macro_f1": delta_metric(base, row, "macro_f1"),
        "delta_accuracy": delta_metric(base, row, "accuracy"),
        "delta_ordinal_mae": delta_metric(base, row, "ordinal_mae"),
        "delta_expected_rmse": delta_metric(base, row, "expected_rmse"),
        "delta_expected_raw_r": delta_metric(base, row, "expected_raw_r"),
        "delta_expected_within_subject_centered_r": delta_metric(base, row, "expected_within_subject_centered_r"),
    }
    result.update(bootstrap)
    return result


def delta_metric(base: dict[str, Any], row: dict[str, Any], metric: str) -> float | str:
    left = base.get("test", {}).get(metric)
    right = row.get("test", {}).get(metric)
    if left is None or right is None:
        return ""
    return float(right) - float(left)


def paired_subject_day_bootstrap(
    base: dict[str, Any],
    candidate: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired candidate-minus-baseline deltas by subject-day blocks within a seed."""
    if iterations <= 0:
        return {"bootstrap_status": "disabled", "bootstrap_iterations": 0}
    base_data = load_test_predictions(base)
    candidate_data = load_test_predictions(candidate)
    if base_data is None or candidate_data is None:
        return {"bootstrap_status": "unavailable_predictions", "bootstrap_iterations": int(iterations)}
    base_aligned, candidate_aligned = align_test_predictions(base_data, candidate_data)
    if base_aligned is None or candidate_aligned is None:
        return {"bootstrap_status": "unavailable_unmatched_events", "bootstrap_iterations": int(iterations)}
    truth = base_aligned["truth"]
    subject = base_aligned["subject_id"]
    day = base_aligned["day_id"]
    blocks = np.asarray([f"{item_subject}::{item_day}" for item_subject, item_day in zip(subject, day)], dtype=str)
    unique_blocks = np.unique(blocks)
    if truth.size == 0 or unique_blocks.size == 0:
        return {"bootstrap_status": "unavailable_empty_test", "bootstrap_iterations": int(iterations)}
    block_indices = [np.flatnonzero(blocks == block) for block in unique_blocks]
    rng = np.random.default_rng(seed)
    qwk_deltas: list[float] = []
    mae_deltas: list[float] = []
    for _ in range(int(iterations)):
        sampled = rng.integers(0, len(block_indices), size=len(block_indices))
        selected = np.concatenate([block_indices[index] for index in sampled])
        base_metrics = classification_metrics(truth[selected], base_aligned["predicted"][selected])
        candidate_metrics = classification_metrics(truth[selected], candidate_aligned["predicted"][selected])
        base_qwk = base_metrics.get("qwk")
        candidate_qwk = candidate_metrics.get("qwk")
        if base_qwk is not None and candidate_qwk is not None:
            qwk_deltas.append(float(candidate_qwk) - float(base_qwk))
        base_mae = base_metrics.get("ordinal_mae")
        candidate_mae = candidate_metrics.get("ordinal_mae")
        if base_mae is not None and candidate_mae is not None:
            mae_deltas.append(float(candidate_mae) - float(base_mae))
    result: dict[str, Any] = {
        "bootstrap_status": "ok",
        "bootstrap_unit": "subject_day",
        "bootstrap_iterations": int(iterations),
        "bootstrap_event_count": int(truth.size),
        "bootstrap_block_count": int(len(block_indices)),
    }
    result.update(bootstrap_interval(qwk_deltas, "bootstrap_delta_qwk"))
    result.update(bootstrap_interval(mae_deltas, "bootstrap_delta_ordinal_mae"))
    return result


def load_test_predictions(row: dict[str, Any]) -> dict[str, np.ndarray] | None:
    metrics_path = Path(str(row.get("metrics_path", "")))
    path = Path(str(row.get("prediction_path", "")))
    if not path.is_file():
        path = metrics_path.parent / "predictions.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            test_index = data["test_index"].astype(np.int64)
            return {
                "event_id": data["event_id"][test_index].astype(str),
                "subject_id": data["subject_id"][test_index].astype(str),
                "day_id": data["day_id"][test_index].astype(str),
                "truth": data["label_zero_based"][test_index].astype(np.int64),
                "predicted": data["test_predicted_class"].astype(np.int64),
            }
    except (KeyError, OSError, ValueError):
        return None


def align_test_predictions(
    base: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    candidate_lookup = {event_id: index for index, event_id in enumerate(candidate["event_id"].tolist())}
    order = [candidate_lookup.get(event_id) for event_id in base["event_id"].tolist()]
    if any(index is None for index in order):
        return None, None
    candidate_order = np.asarray(order, dtype=np.int64)
    if not np.array_equal(base["truth"], candidate["truth"][candidate_order]):
        return None, None
    return base, {key: values[candidate_order] for key, values in candidate.items()}


def bootstrap_interval(values: list[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {f"{prefix}_replicate_count": 0, f"{prefix}_ci_low": None, f"{prefix}_ci_high": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_replicate_count": int(array.size),
        f"{prefix}_ci_low": float(np.percentile(array, 2.5)),
        f"{prefix}_ci_high": float(np.percentile(array, 97.5)),
    }


def adapter_mode_label(row: dict[str, Any]) -> str:
    value = row.get("adapter_mode")
    if value in {"shared", "per_modality"}:
        return str(value)
    return "legacy_coupled_to_normalization"


def experiment_id_label(row: dict[str, Any]) -> str:
    return str(row.get("experiment_id", "legacy_unversioned"))


def experiment_key(row: dict[str, Any], *, include_model: bool) -> tuple[str, ...]:
    values = [
        str(row["protocol"]),
        str(row.get("route_id", row.get("route", ""))),
    ]
    if include_model:
        values.append(str(row.get("model_id", "bag_static")))
    values.extend(
        [
            experiment_id_label(row),
            str(row.get("normalization", "unknown")),
            adapter_mode_label(row),
            str(row.get("objective_id", "legacy_unversioned")),
            str(row.get("routing_id", "legacy_unversioned")),
        ]
    )
    return tuple(values)


def paired_context_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Fields that must match before a static baseline and a routed candidate are paired."""
    return (
        str(row["protocol"]),
        str(row.get("route_id", row.get("route", ""))),
        str(row.get("normalization", "unknown")),
        adapter_mode_label(row),
        str(row.get("objective_id", "legacy_unversioned")),
    )


def stat(rows: list[dict[str, Any]], metric: str, kind: str) -> float | str:
    values = [float(row["test"][metric]) for row in rows if row.get("test", {}).get(metric) is not None]
    if not values:
        return ""
    if kind == "mean":
        return float(mean(values))
    if kind == "std":
        return float(pstdev(values))
    raise ValueError(kind)


def safe_mean(values: list[float]) -> float | str:
    return "" if not values else float(mean(values))


def safe_std(values: list[float]) -> float | str:
    return "" if not values else float(pstdev(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Daily-Affect Ordinal Summary",
        "",
        f"- run_count: `{output['run_count']}`",
        f"- run_root: `{output['run_root']}`",
        "",
        "## Protocol Route Summary",
        "",
        "| protocol | route | model | experiment | normalization | adapter | objective | routing | seeds | QWK mean+/-std | Macro-F1 | Ordinal MAE mean+/-std |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in output["summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route_id']} | {row['model_id']} | {row['experiment_id']} | {row['normalization']} | {row['adapter_mode']} | {row['objective_id']} | {row['routing_id']} | {row['seed_count']} | "
            f"{fmt(row['test_qwk_mean'])}+/-{fmt(row['test_qwk_std'])} | {fmt(row['test_macro_f1_mean'])} | "
            f"{fmt(row['test_ordinal_mae_mean'])}+/-{fmt(row['test_ordinal_mae_std'])} |"
        )
    lines.extend(
        [
            "",
        "## Preliminary Gates",
            "",
            "| protocol | route | model | experiment | normalization | adapter | objective | routing | paired seeds | delta QWK mean+/-std | QWK wins | delta Macro-F1 | delta ordinal MAE mean+/-std | gate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in output["gate_summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route_id']} | {row['model_id']} | {row['experiment_id']} | {row['normalization']} | {row['adapter_mode']} | {row['objective_id']} | {row['routing_id']} | {row['paired_seed_count']} | "
            f"{fmt(row['mean_delta_qwk'])}+/-{fmt(row['std_delta_qwk'])} | {row['qwk_win_count']}/{row['paired_seed_count']} | "
            f"{fmt(row['mean_delta_macro_f1'])} | {fmt(row['mean_delta_ordinal_mae'])}+/-{fmt(row['std_delta_ordinal_mae'])} | {row['preliminary_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Regression Bridge",
            "",
            "Expected ordinal scores are evaluated on the common 1--5 fatigue scale. These columns enable descriptive comparison with the window-level route; the EMA-bag ordinal gate above remains the decision criterion.",
            "",
            "| protocol | route | model | experiment | normalization | adapter | objective | routing | Expected RMSE | Expected raw r | Expected centered r |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in output["summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route_id']} | {row['model_id']} | {row['experiment_id']} | {row['normalization']} | {row['adapter_mode']} | {row['objective_id']} | {row['routing_id']} | "
            f"{fmt(row['test_expected_rmse_mean'])} | {fmt(row['test_expected_raw_r_mean'])} | {fmt(row['test_expected_within_subject_centered_r_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Regression Bridge Deltas",
            "",
            "| protocol | route | model | experiment | normalization | adapter | objective | routing | paired seeds | delta Expected RMSE | delta Expected raw r | delta Expected centered r |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["gate_summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route_id']} | {row['model_id']} | {row['experiment_id']} | {row['normalization']} | {row['adapter_mode']} | {row['objective_id']} | {row['routing_id']} | {row['paired_seed_count']} | "
            f"{fmt(row['mean_delta_expected_rmse'])} | {fmt(row['mean_delta_expected_raw_r'])} | {fmt(row['mean_delta_expected_within_subject_centered_r'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
