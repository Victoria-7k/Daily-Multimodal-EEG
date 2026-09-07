#!/usr/bin/env python3
"""Compare frozen window-regression predictions with EMA-bag daily-affect models.

The comparison unit is one EMA event. Window predictions are pooled over the
canonical 23 aligned windows; daily-affect models contribute their native
five-class decision and expected ordinal score. Calibration uses validation
events only. Test events must consist entirely of window-test rows, preventing
a boundary event from borrowing in-sample window predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.daily_affect.metrics import classification_metrics
from daily_multimodal.daily_affect.training import DailyAffectBagDataset, load_bag_dataset


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_BRIDGE_ROOT = DEFAULT_ROOT / "outputs/daily_affect_window_event_bridge_20260906"
DEFAULT_DAILY_ROUTE = "A1_Wphysio_full__eeg_eegpt_partial_ft_v1"
METRIC_KEYS = (
    "qwk",
    "macro_f1",
    "ordinal_mae",
    "expected_rmse",
    "expected_raw_r",
    "expected_within_subject_centered_r",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BRIDGE_ROOT / "bags")
    parser.add_argument("--window-root", type=Path, default=DEFAULT_BRIDGE_ROOT / "window_runs")
    parser.add_argument("--daily-root", type=Path, default=DEFAULT_BRIDGE_ROOT / "daily_runs")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_BRIDGE_ROOT / "comparison")
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--daily-route-id", default=DEFAULT_DAILY_ROUTE)
    parser.add_argument("--window-route-id", default="A1_Wphysio_full")
    parser.add_argument("--eeg-branch", default="eeg_eegpt_partial_ft_v1")
    parser.add_argument("--normalization", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--daily-model-ids", default="bag_static,dynamic_kernel_prior_uniform")
    parser.add_argument("--seeds", default="240729,240730,240731")
    parser.add_argument("--pooling-policies", default="full_mean,last_30s,kernel_short,kernel_medium,kernel_long")
    parser.add_argument("--primary-pooling", default="full_mean")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    args = parser.parse_args()

    seeds = parse_int_csv(args.seeds)
    model_ids = parse_csv(args.daily_model_ids)
    policies = parse_csv(args.pooling_policies)
    if args.primary_pooling not in policies:
        raise ValueError("--primary-pooling must appear in --pooling-policies")
    for policy in policies:
        pooling_weights(policy, 23)

    condition_id = f"norm_{args.normalization}__adapter_{args.adapter_mode}"
    seed_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    event_prediction_rows: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    for seed in seeds:
        bag_path = args.bags_root / args.protocol / args.daily_route_id / f"seed_{seed}" / "ema_bags.npz"
        dataset = load_bag_dataset(bag_path)
        window_path = window_prediction_path(
            args.window_root, seed, args.protocol, args.eeg_branch, args.window_route_id
        )
        window = load_window_predictions(window_path)
        window_test = event_records_from_window_predictions(dataset, window, dataset.test_index, "test", policies)
        window_val = event_records_from_window_predictions(dataset, window, dataset.val_index, "val", policies)
        preflight.append(
            {
                "seed": seed,
                "bag_path": str(bag_path),
                "window_prediction_path": str(window_path),
                "daily_test_event_count": len(dataset.test_index),
                "window_test_event_count": len(window_test[args.primary_pooling]["event_id"]),
                "window_val_event_count": len(window_val[args.primary_pooling]["event_id"]),
                "test_events_all_pure_window_test": True,
                "val_events_all_pure_window_val": True,
            }
        )

        window_results: dict[tuple[str, str], dict[str, Any]] = {}
        for policy in policies:
            test_record = window_test[policy]
            val_record = window_val[policy]
            native_classes = rounded_classes(test_record["score"])
            calibrated_classes, calibrator = calibrated_classes_from_validation(
                val_record["score"], val_record["label"], test_record["score"]
            )
            for decision, classes, extra in (
                ("native_round", native_classes, {}),
                ("validation_isotonic", calibrated_classes, {"isotonic": calibrator}),
            ):
                record = {**test_record, "predicted_class": classes}
                record["metrics"] = score_record(record, classes)
                window_results[(policy, decision)] = record
                seed_rows.append(
                    metric_row(
                        seed=seed,
                        system="window_regression",
                        model_id="attention_regression",
                        pooling_policy=policy,
                        decision_rule=decision,
                        record=record,
                        extra=extra,
                    )
                )

        event_prediction_rows.extend(
            event_prediction_rows_from_record(
                seed=seed,
                system="window_regression",
                model_id="attention_regression",
                record=window_results[(args.primary_pooling, "native_round")],
            )
        )

        for model_id in model_ids:
            prediction_path = daily_prediction_path(
                args.daily_root, model_id, args.protocol, args.daily_route_id, condition_id, seed
            )
            daily_test = load_daily_record(prediction_path, dataset, "test")
            daily_val = load_daily_record(prediction_path, dataset, "val")
            event_prediction_rows.extend(
                event_prediction_rows_from_record(
                    seed=seed,
                    system="daily_affect",
                    model_id=model_id,
                    record=daily_test,
                )
            )
            calibrated_classes, calibrator = calibrated_classes_from_validation(
                daily_val["score"], daily_val["label"], daily_test["score"]
            )
            for decision, classes, extra in (
                ("native_argmax", daily_test["predicted_class"], {}),
                ("validation_isotonic", calibrated_classes, {"isotonic": calibrator}),
            ):
                record = {**daily_test, "predicted_class": classes}
                record["metrics"] = score_record(record, classes)
                seed_rows.append(
                    metric_row(
                        seed=seed,
                        system="daily_affect",
                        model_id=model_id,
                        pooling_policy="joint_23_window",
                        decision_rule=decision,
                        record=record,
                        extra=extra,
                    )
                )
                for policy in policies:
                    window_decision = "native_round" if decision == "native_argmax" else "validation_isotonic"
                    bootstrap_iterations = args.bootstrap_iters if policy == args.primary_pooling else 0
                    delta = paired_delta_with_bootstrap(
                        window_results[(policy, window_decision)],
                        record,
                        iterations=bootstrap_iterations,
                        seed=args.bootstrap_seed + seed,
                    )
                    if policy != args.primary_pooling:
                        delta["bootstrap_status"] = "not_run_diagnostic_pooling"
                    pair_rows.append(
                        {
                            "seed": seed,
                            "protocol": args.protocol,
                            "daily_route_id": args.daily_route_id,
                            "window_route_id": args.window_route_id,
                            "daily_model_id": model_id,
                            "window_model_id": "attention_regression",
                            "window_pooling_policy": policy,
                            "decision_family": "native" if decision == "native_argmax" else "validation_isotonic",
                            **delta,
                        }
                    )

    summary_rows = summarize_pair_rows(pair_rows)
    output = {
        "script": Path(__file__).name,
        "comparison_status": "exploratory_three_seed_matched_event_bridge",
        "promotion_statement": (
            "This bridge aligns event units and upstream tokens. It has three matched fusion seeds, "
            "so it is descriptive evidence and cannot replace either the five-to-seven-seed promotion gate."
        ),
        "contract": {
            "protocol": args.protocol,
            "daily_route_id": args.daily_route_id,
            "window_route_id": args.window_route_id,
            "eeg_branch": args.eeg_branch,
            "normalization": args.normalization,
            "adapter_mode": args.adapter_mode,
            "window_event_rule": "all 23 windows must originate from the matching window split",
            "calibration_rule": "native decision plus validation-only isotonic expected-score calibration",
            "bootstrap_scope": (
                f"{args.bootstrap_iters} subject-day resamples for the pre-specified "
                f"primary pooling ({args.primary_pooling}); diagnostic pooling uses point estimates"
            ),
            "daily_minus_window_delta": True,
        },
        "preflight": preflight,
        "seed_metrics": seed_rows,
        "paired_deltas": pair_rows,
        "event_prediction_row_count": len(event_prediction_rows),
        "summary": summary_rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "window_daily_event_comparison.json", output)
    write_csv(args.out_root / "seed_metrics.csv", seed_rows)
    write_csv(args.out_root / "paired_deltas.csv", pair_rows)
    write_csv(args.out_root / "paired_summary.csv", summary_rows)
    write_csv(args.out_root / "event_predictions.csv", event_prediction_rows)
    write_markdown(args.out_root / "window_daily_event_comparison.md", output, args.primary_pooling)
    print(f"preflight_seed_count={len(preflight)}")
    print(f"metric_row_count={len(seed_rows)}")
    print(f"paired_delta_count={len(pair_rows)}")
    print(f"event_prediction_row_count={len(event_prediction_rows)}")
    print(f"out_root={args.out_root}")
    return 0


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in parse_csv(value))


def window_prediction_path(root: Path, seed: int, protocol: str, eeg_branch: str, route_id: str) -> Path:
    path = root / f"seed_{seed}" / "predictions" / protocol / eeg_branch / route_id / "regression_raw_lambda_0_headlambda_0.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing window prediction artifact: {path}")
    return path


def daily_prediction_path(root: Path, model_id: str, protocol: str, route_id: str, condition_id: str, seed: int) -> Path:
    if model_id == "bag_static":
        path = root / "phase0" / protocol / route_id / condition_id / f"seed_{seed}" / "predictions.npz"
    else:
        path = root / "runs" / protocol / route_id / model_id / condition_id / f"seed_{seed}" / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing daily-affect prediction artifact: {path}")
    return path


def load_window_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "train_index", "val_index", "test_index", "train_prediction", "val_prediction", "test_prediction",
            "target", "sample_id", "subject_id", "event_id",
        }
        missing = sorted(required.difference(loaded.files))
        if missing:
            raise ValueError(f"{path} missing {missing}")
        target = loaded["target"].astype(np.float64)
        prediction = np.full(target.shape, np.nan, dtype=np.float64)
        split = np.full(target.shape, "", dtype="U8")
        for name in ("train", "val", "test"):
            indices = loaded[f"{name}_index"].astype(np.int64)
            values = loaded[f"{name}_prediction"].astype(np.float64)
            if indices.shape != values.shape:
                raise ValueError(f"{path} has mismatched {name} prediction length")
            if np.any(np.isfinite(prediction[indices])):
                raise ValueError(f"{path} assigns a window prediction more than once")
            prediction[indices] = values
            split[indices] = name
        if not np.all(np.isfinite(prediction)) or np.any(split == ""):
            raise ValueError(f"{path} does not reconstruct all window predictions")
        return {
            "prediction": prediction,
            "split": split,
            "target": target,
            "sample_id": loaded["sample_id"].astype(str),
            "subject_id": loaded["subject_id"].astype(str),
            "event_id": loaded["event_id"].astype(str),
        }


def event_records_from_window_predictions(
    dataset: DailyAffectBagDataset,
    window: dict[str, np.ndarray],
    bag_indices: np.ndarray,
    required_split: str,
    policies: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    id_to_index = {sample_id: index for index, sample_id in enumerate(window["sample_id"].tolist())}
    rows: dict[str, dict[str, list[Any]]] = {
        policy: {"event_id": [], "subject_id": [], "day_id": [], "label": [], "score": []} for policy in policies
    }
    for bag_index in bag_indices.tolist():
        sample_ids = dataset.sample_id_matrix[bag_index].astype(str)
        try:
            indices = np.asarray([id_to_index[sample_id] for sample_id in sample_ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"event {dataset.event_id[bag_index]} has a sample_id absent from the window artifact") from exc
        if not np.all(window["split"][indices] == required_split):
            found = sorted(np.unique(window["split"][indices]).tolist())
            raise ValueError(
                f"event {dataset.event_id[bag_index]} is a mixed window split {found}; "
                f"strict event comparison requires all 23 rows from {required_split}"
            )
        if not np.allclose(window["target"][indices], dataset.label[bag_index], atol=1e-6):
            raise ValueError(f"event {dataset.event_id[bag_index]} has inconsistent window and bag labels")
        for policy in policies:
            rows[policy]["event_id"].append(str(dataset.event_id[bag_index]))
            rows[policy]["subject_id"].append(str(dataset.subject_id[bag_index]))
            rows[policy]["day_id"].append(resolved_day_id(str(dataset.event_id[bag_index]), str(dataset.day_id[bag_index])))
            rows[policy]["label"].append(float(dataset.label[bag_index]))
            rows[policy]["score"].append(float(np.dot(window["prediction"][indices], pooling_weights(policy, len(indices)))))
    return {policy: {key: np.asarray(values) for key, values in row.items()} for policy, row in rows.items()}


def pooling_weights(policy: str, sequence_len: int) -> np.ndarray:
    if sequence_len != 23:
        raise ValueError(f"the event bridge requires 23 windows, got {sequence_len}")
    if policy == "full_mean":
        raw = np.ones(sequence_len, dtype=np.float64)
    elif policy == "last_30s":
        raw = np.zeros(sequence_len, dtype=np.float64)
        raw[max(0, sequence_len - 5) :] = 1.0
    elif policy in {"kernel_short", "kernel_medium", "kernel_long"}:
        tau_seconds = {"kernel_short": 15.0, "kernel_medium": 45.0, "kernel_long": 120.0}[policy]
        distance = (sequence_len - 1) - np.arange(sequence_len, dtype=np.float64)
        raw = np.exp(-distance / (tau_seconds / 5.0))
    else:
        raise ValueError(f"unsupported pooling policy: {policy}")
    return raw / raw.sum()


def load_daily_record(path: Path, dataset: DailyAffectBagDataset, split_name: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        index_key = f"{split_name}_index"
        score_key = f"{split_name}_expected_score"
        class_key = f"{split_name}_predicted_class"
        required = {index_key, score_key, class_key}
        missing = sorted(required.difference(loaded.files))
        if missing:
            raise ValueError(f"{path} missing {missing}")
        indices = loaded[index_key].astype(np.int64)
        expected = loaded[score_key].astype(np.float64)
        predicted = loaded[class_key].astype(np.int64)
    expected_indices = getattr(dataset, f"{split_name}_index")
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"{path} {index_key} does not match its source EMA bag")
    if not (indices.size == expected.size == predicted.size):
        raise ValueError(f"{path} has inconsistent {split_name} prediction lengths")
    events = dataset.event_id[indices].astype(str)
    return {
        "event_id": events,
        "subject_id": dataset.subject_id[indices].astype(str),
        "day_id": np.asarray([resolved_day_id(event, day) for event, day in zip(events, dataset.day_id[indices])]),
        "label": dataset.label[indices].astype(np.float64),
        "score": expected,
        "predicted_class": np.clip(predicted, 0, 4),
    }


def resolved_day_id(event_id: str, fallback: str) -> str:
    match = re.search(r"_day-([^_]+)_event-", event_id)
    return match.group(1) if match else fallback


def rounded_classes(score: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(score, dtype=np.float64)), 1, 5).astype(np.int64) - 1


def fit_isotonic(score: np.ndarray, label: np.ndarray) -> dict[str, list[float]]:
    x = np.asarray(score, dtype=np.float64).reshape(-1)
    y = np.asarray(label, dtype=np.float64).reshape(-1)
    if x.size == 0 or x.size != y.size:
        raise ValueError("isotonic calibration needs equally sized nonempty validation arrays")
    unique, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y, minlength=unique.size).astype(np.float64)
    counts = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    blocks: list[dict[str, float | int]] = []
    for index in range(unique.size):
        blocks.append({"start": index, "end": index, "sum": float(sums[index]), "count": float(counts[index])})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if float(left["sum"]) / float(left["count"]) <= float(right["sum"]) / float(right["count"]):
                break
            blocks[-2:] = [{
                "start": int(left["start"]), "end": int(right["end"]),
                "sum": float(left["sum"]) + float(right["sum"]),
                "count": float(left["count"]) + float(right["count"]),
            }]
    return {
        "upper_bounds": [float(unique[int(block["end"])]) for block in blocks],
        "values": [float(block["sum"]) / float(block["count"]) for block in blocks],
    }


def apply_isotonic(calibrator: dict[str, list[float]], score: np.ndarray) -> np.ndarray:
    upper = np.asarray(calibrator["upper_bounds"], dtype=np.float64)
    values = np.asarray(calibrator["values"], dtype=np.float64)
    positions = np.searchsorted(upper, np.asarray(score, dtype=np.float64), side="left")
    return values[np.clip(positions, 0, len(values) - 1)]


def calibrated_classes_from_validation(val_score: np.ndarray, val_label: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    calibrator = fit_isotonic(val_score, val_label)
    return rounded_classes(apply_isotonic(calibrator, test_score)), calibrator


def score_record(record: dict[str, np.ndarray], predicted_class: np.ndarray) -> dict[str, Any]:
    label_zero = np.clip(np.rint(record["label"]), 1, 5).astype(np.int64) - 1
    return classification_metrics(
        label_zero,
        np.asarray(predicted_class, dtype=np.int64),
        expected_score=record["score"],
        subject_ids=record["subject_id"],
    )


def metric_row(
    *, seed: int, system: str, model_id: str, pooling_policy: str, decision_rule: str,
    record: dict[str, Any], extra: dict[str, Any],
) -> dict[str, Any]:
    metrics = record["metrics"]
    return {
        "seed": seed,
        "system": system,
        "model_id": model_id,
        "pooling_policy": pooling_policy,
        "decision_rule": decision_rule,
        "event_count": int(len(record["event_id"])),
        **{key: metrics.get(key) for key in METRIC_KEYS},
        "calibrator_block_count": len(extra.get("isotonic", {}).get("values", [])),
    }


def event_prediction_rows_from_record(
    *, seed: int, system: str, model_id: str, record: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "system": system,
            "model_id": model_id,
            "event_id": str(event_id),
            "subject_id": str(subject_id),
            "day_id": str(day_id),
            "label": float(label),
            "expected_score": float(score),
        }
        for event_id, subject_id, day_id, label, score in zip(
            record["event_id"],
            record["subject_id"],
            record["day_id"],
            record["label"],
            record["score"],
        )
    ]


def aligned_records(left: dict[str, Any], right: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_order = np.argsort(left["event_id"])
    right_order = np.argsort(right["event_id"])
    if not np.array_equal(left["event_id"][left_order], right["event_id"][right_order]):
        raise ValueError("daily and window records do not contain the same event ids")
    keys = ("event_id", "subject_id", "day_id", "label", "score", "predicted_class")
    first = {key: left[key][left_order] for key in keys}
    second = {key: right[key][right_order] for key in keys}
    if not np.array_equal(first["label"], second["label"]):
        raise ValueError("daily and window event labels disagree")
    if not np.array_equal(first["subject_id"], second["subject_id"]):
        raise ValueError("daily and window subject ids disagree")
    return first, second


def paired_delta_with_bootstrap(window: dict[str, Any], daily: dict[str, Any], *, iterations: int, seed: int) -> dict[str, Any]:
    window, daily = aligned_records(window, daily)
    window_metrics = score_record(window, window["predicted_class"])
    daily_metrics = score_record(daily, daily["predicted_class"])
    result = {
        "event_count": int(len(window["event_id"])),
        **{f"window_{key}": window_metrics.get(key) for key in METRIC_KEYS},
        **{f"daily_{key}": daily_metrics.get(key) for key in METRIC_KEYS},
        **{f"delta_daily_minus_window_{key}": metric_delta(window_metrics, daily_metrics, key) for key in METRIC_KEYS},
    }
    result.update(bootstrap_deltas(window, daily, iterations=iterations, seed=seed))
    return result


def metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    first, second = left.get(key), right.get(key)
    return None if first is None or second is None else float(second) - float(first)


def bootstrap_deltas(window: dict[str, Any], daily: dict[str, Any], *, iterations: int, seed: int) -> dict[str, Any]:
    if iterations <= 0:
        return {"bootstrap_status": "disabled", "bootstrap_iterations": 0}
    blocks = np.asarray([f"{subject}::{day}" for subject, day in zip(window["subject_id"], window["day_id"])], dtype=str)
    unique = np.unique(blocks)
    if unique.size == 0:
        return {"bootstrap_status": "unavailable_empty_test", "bootstrap_iterations": int(iterations)}
    block_indices = [np.flatnonzero(blocks == block) for block in unique]
    values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        sampled = np.concatenate([block_indices[index] for index in rng.integers(0, len(block_indices), size=len(block_indices))])
        window_metrics = score_record(subset_record(window, sampled), window["predicted_class"][sampled])
        daily_metrics = score_record(subset_record(daily, sampled), daily["predicted_class"][sampled])
        for key in METRIC_KEYS:
            delta = metric_delta(window_metrics, daily_metrics, key)
            if delta is not None and math.isfinite(delta):
                values[key].append(delta)
    result: dict[str, Any] = {
        "bootstrap_status": "ok",
        "bootstrap_unit": "subject_day",
        "bootstrap_iterations": int(iterations),
        "bootstrap_block_count": int(len(block_indices)),
    }
    for key, series in values.items():
        array = np.asarray(series, dtype=np.float64)
        result[f"bootstrap_delta_daily_minus_window_{key}_ci_low"] = float(np.quantile(array, 0.025)) if array.size else None
        result[f"bootstrap_delta_daily_minus_window_{key}_ci_high"] = float(np.quantile(array, 0.975)) if array.size else None
    return result


def subset_record(record: dict[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: record[key][indices] for key in ("event_id", "subject_id", "day_id", "label", "score", "predicted_class")}


def summarize_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["daily_model_id"]), str(row["window_pooling_policy"]), str(row["decision_family"]))].append(row)
    output: list[dict[str, Any]] = []
    positive_metrics = {"qwk", "macro_f1", "expected_raw_r", "expected_within_subject_centered_r"}
    for (daily_model, policy, decision), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "daily_model_id": daily_model,
            "window_pooling_policy": policy,
            "decision_family": decision,
            "paired_seed_count": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda row: int(row["seed"]))),
        }
        for key in METRIC_KEYS:
            series = [row.get(f"delta_daily_minus_window_{key}") for row in group]
            valid = np.asarray([value for value in series if value is not None and math.isfinite(float(value))], dtype=np.float64)
            summary[f"mean_delta_daily_minus_window_{key}"] = float(valid.mean()) if valid.size else None
            summary[f"std_delta_daily_minus_window_{key}"] = float(valid.std(ddof=1)) if valid.size > 1 else 0.0 if valid.size else None
            summary[f"daily_win_count_{key}"] = int(np.sum(valid > 0.0)) if key in positive_metrics else int(np.sum(valid < 0.0))
        output.append(summary)
    return output


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def write_markdown(path: Path, output: dict[str, Any], primary_pooling: str) -> None:
    lines = [
        "# Window Mainline vs Daily-Affect Event-Level Comparison",
        "",
        f"- status: `{output['comparison_status']}`",
        f"- protocol: `{output['contract']['protocol']}`",
        f"- daily route: `{output['contract']['daily_route_id']}`",
        f"- window route: `{output['contract']['window_route_id']}`",
        f"- EEG branch: `{output['contract']['eeg_branch']}`",
        f"- primary window pooling: `{primary_pooling}`",
        f"- bootstrap: {output['contract']['bootstrap_scope']}",
        f"- {output['promotion_statement']}",
        "",
        "## Preflight",
        "",
        "| seed | daily test events | window test events | window val events | strict split check |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in output["preflight"]:
        lines.append(f"| {row['seed']} | {row['daily_test_event_count']} | {row['window_test_event_count']} | {row['window_val_event_count']} | pass |")
    lines.extend([
        "",
        "## Paired Deltas",
        "",
        "Each delta is daily-affect minus pooled window regression. Positive is favorable for QWK/F1/r; negative is favorable for MAE/RMSE.",
        "",
        "| daily model | window pooling | decision | seeds | delta QWK | daily QWK wins | delta raw r | daily raw-r wins | delta RMSE | daily RMSE wins |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in output["summary"]:
        if row["window_pooling_policy"] != primary_pooling:
            continue
        lines.append(
            f"| {row['daily_model_id']} | {row['window_pooling_policy']} | {row['decision_family']} | {row['paired_seed_count']} | "
            f"{fmt(row['mean_delta_daily_minus_window_qwk'])}+/-{fmt(row['std_delta_daily_minus_window_qwk'])} | {row['daily_win_count_qwk']}/{row['paired_seed_count']} | "
            f"{fmt(row['mean_delta_daily_minus_window_expected_raw_r'])}+/-{fmt(row['std_delta_daily_minus_window_expected_raw_r'])} | {row['daily_win_count_expected_raw_r']}/{row['paired_seed_count']} | "
            f"{fmt(row['mean_delta_daily_minus_window_expected_rmse'])}+/-{fmt(row['std_delta_daily_minus_window_expected_rmse'])} | {row['daily_win_count_expected_rmse']}/{row['paired_seed_count']} |"
        )
    lines.extend([
        "",
        "## Temporal-Pooling Diagnostic",
        "",
        "| daily model | window pooling | decision | delta QWK | daily QWK wins |",
        "| --- | --- | --- | ---: | ---: |",
    ])
    for row in output["summary"]:
        if row["decision_family"] == "native":
            lines.append(
                f"| {row['daily_model_id']} | {row['window_pooling_policy']} | {row['decision_family']} | "
                f"{fmt(row['mean_delta_daily_minus_window_qwk'])}+/-{fmt(row['std_delta_daily_minus_window_qwk'])} | {row['daily_win_count_qwk']}/{row['paired_seed_count']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
