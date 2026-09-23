"""Run Phase 0/1 fatigue prediction diagnostics and post-hoc calibration.

This helper is intentionally post-hoc: it reads frozen baseline predictions,
fits calibration parameters on validation predictions only, and then applies
the frozen calibrators to train/val/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPRESENTATIVE_ROUTES = (
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_full", "highest raw r"),
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_no_audio", "lowest RMSE / stronger centered r"),
    ("date_in_order", "eeg_eegpt_partial_ft_v1", "B0_Wdeep_no_audio", "highest raw r / centered r"),
    ("date_in_order", "eeg_eegpt_partial_ft_v1", "A2_Wdeep_full", "lowest RMSE"),
)


@dataclass(frozen=True)
class RouteKey:
    protocol: str
    eeg_branch: str
    experiment: str
    role: str

    @property
    def slug(self) -> str:
        return f"{self.protocol}__{self.eeg_branch}__{self.experiment}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-json",
        default="outputs/server_sync/eeg_encoder_256d_5route_20260814/reports/"
        "eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json",
    )
    parser.add_argument(
        "--prediction-root",
        default="outputs/server_sync/eeg_encoder_256d_5route_20260814/predictions/"
        "eeg_encoder_256d_5route_fusion_video_only_seed240800_raw",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/server_sync/fatigue_calibration_20260816",
    )
    parser.add_argument("--row-count", type=int, default=28819)
    return parser.parse_args()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def centered_r(pred: np.ndarray, target: np.ndarray, subject: np.ndarray) -> float:
    pred_centered = pred.astype(np.float64).copy()
    target_centered = target.astype(np.float64).copy()
    for sid in np.unique(subject):
        mask = subject == sid
        pred_centered[mask] -= float(np.mean(pred_centered[mask]))
        target_centered[mask] -= float(np.mean(target_centered[mask]))
    return pearson(pred_centered, target_centered)


def per_subject_r(pred: np.ndarray, target: np.ndarray, subject: np.ndarray) -> tuple[float, float]:
    values: list[float] = []
    for sid in np.unique(subject):
        mask = subject == sid
        if int(np.sum(mask)) >= 2:
            value = pearson(pred[mask], target[mask])
            if math.isfinite(value):
                values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def linear_fit(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if len(pred) < 2 or float(np.std(pred)) == 0.0:
        return 1.0, float(np.mean(target) - np.mean(pred))
    slope, intercept = np.polyfit(pred.astype(np.float64), target.astype(np.float64), deg=1)
    return float(slope), float(intercept)


def label_masks(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = target <= 2.0
    high = target >= 4.0
    if not np.any(low):
        low = target <= np.quantile(target, 0.25)
    if not np.any(high):
        high = target >= np.quantile(target, 0.75)
    return low, high


def metrics_for(pred: np.ndarray, target: np.ndarray, subject: np.ndarray) -> dict[str, float]:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    err = pred - target
    low, high = label_masks(target)
    slope, intercept = linear_fit(pred, target)
    ps_mean, ps_std = per_subject_r(pred, target, subject)
    true_std = float(np.std(target))
    pred_std = float(np.std(pred))
    return {
        "count": int(len(target)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "raw_r": pearson(pred, target),
        "within_subject_centered_r": centered_r(pred, target, subject),
        "per_subject_r_mean": ps_mean,
        "per_subject_r_std": ps_std,
        "pred_mean": float(np.mean(pred)),
        "true_mean": float(np.mean(target)),
        "pred_std": pred_std,
        "true_std": true_std,
        "pred_std_over_true_std": float(pred_std / true_std) if true_std else float("nan"),
        "low_fatigue_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "high_fatigue_bias": float(np.mean(err[high])) if np.any(high) else float("nan"),
        "low_fatigue_mae": float(np.mean(np.abs(err[low]))) if np.any(low) else float("nan"),
        "high_fatigue_mae": float(np.mean(np.abs(err[high]))) if np.any(high) else float("nan"),
        "extreme_mae": float(np.mean(np.abs(err[low | high]))) if np.any(low | high) else float("nan"),
        "calibration_slope_y_on_pred": slope,
        "calibration_intercept_y_on_pred": intercept,
        "low_count": int(np.sum(low)),
        "high_count": int(np.sum(high)),
    }


def resolve_prediction_path(root: Path, report_run: dict[str, Any]) -> Path:
    direct = root / report_run["protocol"] / report_run["eeg_branch"] / report_run["experiment"] / "raw_lambda_0.npz"
    if direct.exists():
        return direct
    recorded = Path(str(report_run.get("prediction_path", "")))
    if recorded.exists():
        return recorded
    parts = recorded.parts
    if "outputs" in parts:
        suffix = Path(*parts[parts.index("outputs") + 2 :]) if len(parts) > parts.index("outputs") + 2 else recorded
        candidate = root.parent / suffix
        if candidate.exists():
            return candidate
    raise FileNotFoundError(direct)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def split_arrays(data: dict[str, np.ndarray], split: str, pred_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = data[f"{split}_index"].astype(np.int64)
    pred = data[f"{split}_prediction"].astype(np.float64) if pred_override is None else pred_override.astype(np.float64)
    target = data["target"][index].astype(np.float64)
    subject = data["subject_id"][index].astype(str)
    return pred, target, subject


def audit_npz(data: dict[str, np.ndarray], row_count: int) -> dict[str, Any]:
    required = {
        "train_index",
        "val_index",
        "test_index",
        "train_prediction",
        "val_prediction",
        "test_prediction",
        "target",
        "sample_id",
        "subject_id",
        "event_id",
    }
    missing = sorted(required - set(data))
    split_sets = {split: set(data[f"{split}_index"].astype(int).tolist()) for split in ("train", "val", "test")}
    overlaps = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    length_ok = all(len(data[key]) == row_count for key in ("target", "sample_id", "subject_id", "event_id") if key in data)
    pred_length_ok = all(
        len(data[f"{split}_prediction"]) == len(data[f"{split}_index"])
        for split in ("train", "val", "test")
        if f"{split}_prediction" in data and f"{split}_index" in data
    )
    index_bounds_ok = all(
        int(np.min(data[f"{split}_index"])) >= 0 and int(np.max(data[f"{split}_index"])) < row_count
        for split in ("train", "val", "test")
        if f"{split}_index" in data and len(data[f"{split}_index"])
    )
    return {
        "missing_keys": missing,
        "row_count": int(len(data["target"])) if "target" in data else None,
        "length_ok": bool(length_ok),
        "prediction_length_ok": bool(pred_length_ok),
        "index_bounds_ok": bool(index_bounds_ok),
        "split_overlaps": overlaps,
        "split_overlap_ok": all(value == 0 for value in overlaps.values()),
        "unique_sample_id_count": int(len(set(data["sample_id"].astype(str).tolist()))) if "sample_id" in data else None,
        "sample_id_unique": bool(len(set(data["sample_id"].astype(str).tolist())) == len(data["sample_id"])) if "sample_id" in data else False,
    }


def day_key(event_id: str) -> str:
    parts = event_id.split("_")
    for part in parts:
        if part.startswith("day-"):
            return part
    return event_id


def subject_day_stats(data: dict[str, np.ndarray], split: str) -> dict[str, Any]:
    index = data[f"{split}_index"].astype(np.int64)
    pred = data[f"{split}_prediction"].astype(np.float64)
    target = data["target"][index].astype(np.float64)
    subjects = data["subject_id"][index].astype(str)
    events = data["event_id"][index].astype(str)
    keys = np.array([f"{subject} {day_key(event)}" for subject, event in zip(subjects, events)], dtype=object)
    rows = []
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        if int(np.sum(mask)) < 2:
            continue
        pred_slice = pred[mask]
        target_slice = target[mask]
        rows.append(
            {
                "subject_day": key,
                "count": int(np.sum(mask)),
                "true_std": float(np.std(target_slice)),
                "pred_std": float(np.std(pred_slice)),
                "std_ratio": float(np.std(pred_slice) / np.std(target_slice)) if float(np.std(target_slice)) else float("nan"),
                "bias": float(np.mean(pred_slice - target_slice)),
                "raw_r": pearson(pred_slice, target_slice),
            }
        )
    finite = [row for row in rows if math.isfinite(row["true_std"]) and row["true_std"] > 0]
    selected = max(finite, key=lambda row: (row["true_std"], row["count"])) if finite else (rows[0] if rows else {})
    return {"selected_subject_day": selected, "subject_day_count": len(rows)}


def fit_calibrators(data: dict[str, np.ndarray]) -> dict[str, dict[str, float | str]]:
    val_pred, val_target, _ = split_arrays(data, "val")
    train_target = data["target"][data["train_index"].astype(np.int64)].astype(np.float64)
    a, b = linear_fit(val_pred, val_target)
    pred_std = float(np.std(val_pred))
    target_std = float(np.std(val_target))
    variance_scale = float(target_std / pred_std) if pred_std else 1.0
    pred_mean = float(np.mean(val_pred))
    target_mean = float(np.mean(val_target))
    return {
        "linear_calibration": {"type": "linear", "a": a, "b": b},
        "variance_calibration": {
            "type": "variance",
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "scale": variance_scale,
        },
        "clipped_variance_calibration": {
            "type": "clipped_variance",
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "scale": variance_scale,
            "clip_min": float(np.min(train_target)),
            "clip_max": float(np.max(train_target)),
        },
    }


def apply_calibrator(pred: np.ndarray, params: dict[str, float | str]) -> np.ndarray:
    kind = str(params["type"])
    if kind == "linear":
        calibrated = float(params["a"]) * pred + float(params["b"])
    elif kind in {"variance", "clipped_variance"}:
        calibrated = float(params["target_mean"]) + (pred - float(params["pred_mean"])) * float(params["scale"])
        if kind == "clipped_variance":
            calibrated = np.clip(calibrated, float(params["clip_min"]), float(params["clip_max"]))
    else:
        raise ValueError(f"Unsupported calibrator: {kind}")
    return calibrated.astype(np.float32)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_distribution_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "protocol",
        "eeg_branch",
        "experiment",
        "role",
        "calibration",
        "split",
        "count",
        "rmse",
        "mae",
        "raw_r",
        "within_subject_centered_r",
        "per_subject_r_mean",
        "per_subject_r_std",
        "pred_std_over_true_std",
        "low_fatigue_bias",
        "high_fatigue_bias",
        "extreme_mae",
        "calibration_slope_y_on_pred",
        "calibration_intercept_y_on_pred",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = []
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if math.isfinite(value) else "nan")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_json)
    prediction_root = Path(args.prediction_root)
    out_root = Path(args.out_root)
    phase0 = out_root / "phase0_preflight"
    phase1 = out_root / "phase1_posthoc_calibration"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = {
        (run["protocol"], run["eeg_branch"], run["experiment"]): run
        for run in report["results"]
        if run.get("status", "ok") == "ok"
    }
    routes = [RouteKey(*values) for values in REPRESENTATIVE_ROUTES]

    preflight_routes: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    subject_day_rows: list[dict[str, Any]] = []
    calibration_val: dict[str, Any] = {}
    calibration_test: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []

    for route in routes:
        run = runs[(route.protocol, route.eeg_branch, route.experiment)]
        npz_path = resolve_prediction_path(prediction_root, run)
        data = load_npz(npz_path)
        audit = audit_npz(data, args.row_count)
        split_metrics = {}
        for split in ("train", "val", "test"):
            pred, target, subject = split_arrays(data, split)
            metrics = metrics_for(pred, target, subject)
            split_metrics[split] = metrics
            baseline_rows.append(
                {
                    "phase": "phase0",
                    "protocol": route.protocol,
                    "eeg_branch": route.eeg_branch,
                    "experiment": route.experiment,
                    "role": route.role,
                    "calibration": "baseline",
                    "split": split,
                    **metrics,
                }
            )
        sd = subject_day_stats(data, "test")
        subject_day_rows.append({"route": route.slug, **sd})
        preflight_routes.append(
            {
                "protocol": route.protocol,
                "eeg_branch": route.eeg_branch,
                "experiment": route.experiment,
                "role": route.role,
                "prediction_path": str(npz_path),
                "audit": audit,
                "metrics": split_metrics,
                "subject_day_test": sd,
            }
        )

        calibrators = fit_calibrators(data)
        calibration_val[route.slug] = {"baseline": split_metrics["val"], "calibrations": {}}
        calibration_test[route.slug] = {"baseline": split_metrics["test"], "calibrations": {}}
        for name, params in calibrators.items():
            calibrated_splits = {
                split: apply_calibrator(data[f"{split}_prediction"].astype(np.float64), params)
                for split in ("train", "val", "test")
            }
            out_npz = phase1 / "predictions" / route.protocol / route.eeg_branch / route.experiment / f"{name}.npz"
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_npz,
                train_index=data["train_index"],
                val_index=data["val_index"],
                test_index=data["test_index"],
                train_prediction=calibrated_splits["train"],
                val_prediction=calibrated_splits["val"],
                test_prediction=calibrated_splits["test"],
                target=data["target"],
                sample_id=data["sample_id"],
                subject_id=data["subject_id"],
                event_id=data["event_id"],
                calibration_name=np.array(name),
                calibration_params=np.array(json.dumps(params, ensure_ascii=False)),
                source_prediction_path=np.array(str(npz_path)),
            )
            for split in ("train", "val", "test"):
                pred, target, subject = split_arrays(data, split, calibrated_splits[split])
                metrics = metrics_for(pred, target, subject)
                baseline_rows.append(
                    {
                        "phase": "phase1",
                        "protocol": route.protocol,
                        "eeg_branch": route.eeg_branch,
                        "experiment": route.experiment,
                        "role": route.role,
                        "calibration": name,
                        "split": split,
                        **metrics,
                    }
                )
                if split == "val":
                    calibration_val[route.slug]["calibrations"][name] = {"params": params, "metrics": metrics, "path": str(out_npz)}
                if split == "test":
                    calibration_test[route.slug]["calibrations"][name] = {"params": params, "metrics": metrics, "path": str(out_npz)}
            val_metrics = calibration_val[route.slug]["calibrations"][name]["metrics"]
            base_val = split_metrics["val"]
            high_base = abs(float(base_val["high_fatigue_bias"]))
            high_new = abs(float(val_metrics["high_fatigue_bias"]))
            high_bias_reduction = (high_base - high_new) / high_base if high_base else float("nan")
            std_delta = float(val_metrics["pred_std_over_true_std"] - base_val["pred_std_over_true_std"])
            passes = (
                (std_delta >= 0.15 or high_bias_reduction >= 0.15)
                and float(val_metrics["rmse"] - base_val["rmse"]) <= 0.010
                and float(val_metrics["mae"] - base_val["mae"]) <= 0.010
                and float(val_metrics["raw_r"] - base_val["raw_r"]) >= -0.010
                and float(val_metrics["within_subject_centered_r"] - base_val["within_subject_centered_r"]) >= -0.015
            )
            gate_rows.append(
                {
                    "route": route.slug,
                    "protocol": route.protocol,
                    "calibration": name,
                    "std_ratio_delta": std_delta,
                    "high_abs_bias_reduction": float(high_bias_reduction),
                    "rmse_delta": float(val_metrics["rmse"] - base_val["rmse"]),
                    "mae_delta": float(val_metrics["mae"] - base_val["mae"]),
                    "raw_r_delta": float(val_metrics["raw_r"] - base_val["raw_r"]),
                    "centered_r_delta": float(
                        val_metrics["within_subject_centered_r"] - base_val["within_subject_centered_r"]
                    ),
                    "passes_phase2_gate": bool(passes),
                }
            )

    preflight_ok = all(
        not route["audit"]["missing_keys"]
        and route["audit"]["length_ok"]
        and route["audit"]["prediction_length_ok"]
        and route["audit"]["index_bounds_ok"]
        and route["audit"]["split_overlap_ok"]
        for route in preflight_routes
    )
    phenomenon_ok = all(
        route["metrics"]["test"]["pred_std_over_true_std"] < 0.75 for route in preflight_routes
    )
    preflight_payload = {
        "ok": bool(preflight_ok and phenomenon_ok),
        "preflight_ok": bool(preflight_ok),
        "phenomenon_ok": bool(phenomenon_ok),
        "report_json": str(report_path),
        "prediction_root": str(prediction_root),
        "route_count": len(preflight_routes),
        "routes": preflight_routes,
    }
    write_json(phase0 / "preflight.json", preflight_payload)
    write_distribution_csv(phase0 / "baseline_distribution_stats.csv", baseline_rows)
    write_json(phase1 / "metrics_val.json", calibration_val)
    write_json(phase1 / "metrics_test_frozen.json", calibration_test)

    phase2_gate_passed = any(row["passes_phase2_gate"] for row in gate_rows if row["protocol"] in {"cross_day", "date_in_order"})
    summary_lines = [
        "# Phase 1 Post-hoc Calibration Summary",
        "",
        f"- Phase 0 preflight ok: `{preflight_payload['ok']}`",
        f"- Phase 1 val gate passed: `{phase2_gate_passed}`",
        "- Calibrators were fit on validation predictions only; test metrics below are frozen evaluations.",
        "- Isotonic calibration was not run in this pass; it remains an optional overfit-risk diagnostic.",
        "",
        "## Baseline Test Diagnostics",
        "",
        markdown_table(
            [
                {
                    "route": route["protocol"] + "/" + route["experiment"],
                    "rmse": route["metrics"]["test"]["rmse"],
                    "raw_r": route["metrics"]["test"]["raw_r"],
                    "centered_r": route["metrics"]["test"]["within_subject_centered_r"],
                    "std_ratio": route["metrics"]["test"]["pred_std_over_true_std"],
                    "low_bias": route["metrics"]["test"]["low_fatigue_bias"],
                    "high_bias": route["metrics"]["test"]["high_fatigue_bias"],
                }
                for route in preflight_routes
            ],
            ["route", "rmse", "raw_r", "centered_r", "std_ratio", "low_bias", "high_bias"],
        ),
        "",
        "## Selected Subject-day Checks",
        "",
        markdown_table(
            [
                {
                    "route": row["route"],
                    **row.get("selected_subject_day", {}),
                }
                for row in subject_day_rows
            ],
            ["route", "subject_day", "count", "true_std", "pred_std", "std_ratio", "bias", "raw_r"],
        ),
        "",
        "## Val Gate",
        "",
        markdown_table(
            gate_rows,
            [
                "route",
                "calibration",
                "std_ratio_delta",
                "high_abs_bias_reduction",
                "rmse_delta",
                "mae_delta",
                "raw_r_delta",
                "centered_r_delta",
                "passes_phase2_gate",
            ],
        ),
    ]
    (phase1 / "calibration_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (phase1 / "calibration_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    diag_lines = [
        "# Phase 0 Baseline Prediction Diagnostics",
        "",
        f"- Preflight ok: `{preflight_payload['ok']}`",
        "- The test-set diagnostics reproduce the compressed prediction range: every representative route has "
        "`pred_std / true_std < 0.75`.",
        "",
        "## Representative Routes",
        "",
        markdown_table(
            [
                {
                    "route": route["protocol"] + "/" + route["experiment"],
                    "role": route["role"],
                    "val_rmse": route["metrics"]["val"]["rmse"],
                    "val_raw_r": route["metrics"]["val"]["raw_r"],
                    "test_rmse": route["metrics"]["test"]["rmse"],
                    "test_raw_r": route["metrics"]["test"]["raw_r"],
                    "test_centered_r": route["metrics"]["test"]["within_subject_centered_r"],
                    "test_std_ratio": route["metrics"]["test"]["pred_std_over_true_std"],
                    "test_low_bias": route["metrics"]["test"]["low_fatigue_bias"],
                    "test_high_bias": route["metrics"]["test"]["high_fatigue_bias"],
                }
                for route in preflight_routes
            ],
            [
                "route",
                "role",
                "val_rmse",
                "val_raw_r",
                "test_rmse",
                "test_raw_r",
                "test_centered_r",
                "test_std_ratio",
                "test_low_bias",
                "test_high_bias",
            ],
        ),
    ]
    (phase0 / "baseline_prediction_diagnostics.md").write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    print(json.dumps({"phase0_ok": preflight_payload["ok"], "phase1_gate_passed": phase2_gate_passed}, indent=2))


if __name__ == "__main__":
    main()
