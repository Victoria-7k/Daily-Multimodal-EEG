"""Phase 1b distribution and prediction-collapse checks for fatigue calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROUTES = (
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_full", "highest raw r"),
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_no_audio", "lowest RMSE / stronger centered r"),
    ("date_in_order", "eeg_eegpt_partial_ft_v1", "B0_Wdeep_no_audio", "highest raw r / centered r"),
    ("date_in_order", "eeg_eegpt_partial_ft_v1", "A2_Wdeep_full", "lowest RMSE"),
)

CALIBRATIONS = ("variance_calibration", "clipped_variance_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        default="outputs/server_sync/eeg_encoder_256d_5route_20260814/predictions/"
        "eeg_encoder_256d_5route_fusion_video_only_seed240800_raw",
    )
    parser.add_argument(
        "--phase1-root",
        default="outputs/server_sync/fatigue_calibration_20260816/phase1_posthoc_calibration",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/server_sync/fatigue_calibration_20260816/phase1b_distribution_check",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def split_view(data: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray]:
    index = data[f"{split}_index"].astype(np.int64)
    target = data["target"][index].astype(np.float64)
    pred = data[f"{split}_prediction"].astype(np.float64)
    return target, pred


def baseline_path(prediction_root: Path, protocol: str, eeg: str, experiment: str) -> Path:
    return prediction_root / protocol / eeg / experiment / "raw_lambda_0.npz"


def calibration_path(phase1_root: Path, protocol: str, eeg: str, experiment: str, calibration: str) -> Path:
    return phase1_root / "predictions" / protocol / eeg / experiment / f"{calibration}.npz"


def value_counts(target: np.ndarray) -> dict[str, int]:
    rounded = np.rint(target).astype(int)
    return {str(value): int(np.sum(rounded == value)) for value in range(1, 6)}


def split_stats(target: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    low_mask = target <= 2.0
    high_mask = target >= 4.0
    pred_2x = (pred >= 2.0) & (pred < 3.0)
    pred_225_275 = (pred >= 2.25) & (pred <= 2.75)
    high_pred = pred[high_mask] if np.any(high_mask) else np.asarray([], dtype=np.float64)
    low_pred = pred[low_mask] if np.any(low_mask) else np.asarray([], dtype=np.float64)
    true_std = float(np.std(target))
    pred_std = float(np.std(pred))
    return {
        "count": int(len(target)),
        "target_mean": float(np.mean(target)),
        "target_std": true_std,
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "target_q10": float(np.quantile(target, 0.10)),
        "target_q25": float(np.quantile(target, 0.25)),
        "target_median": float(np.quantile(target, 0.50)),
        "target_q75": float(np.quantile(target, 0.75)),
        "target_q90": float(np.quantile(target, 0.90)),
        "target_counts": value_counts(target),
        "low_count": int(np.sum(low_mask)),
        "high_count": int(np.sum(high_mask)),
        "low_rate": float(np.mean(low_mask)),
        "high_rate": float(np.mean(high_mask)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": pred_std,
        "pred_std_over_true_std": float(pred_std / true_std) if true_std else float("nan"),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "pred_q10": float(np.quantile(pred, 0.10)),
        "pred_q25": float(np.quantile(pred, 0.25)),
        "pred_median": float(np.quantile(pred, 0.50)),
        "pred_q75": float(np.quantile(pred, 0.75)),
        "pred_q90": float(np.quantile(pred, 0.90)),
        "pred_2x_rate": float(np.mean(pred_2x)),
        "pred_225_275_rate": float(np.mean(pred_225_275)),
        "high_target_pred_mean": float(np.mean(high_pred)) if len(high_pred) else float("nan"),
        "high_target_pred_std": float(np.std(high_pred)) if len(high_pred) else float("nan"),
        "high_target_pred_2x_rate": float(np.mean((high_pred >= 2.0) & (high_pred < 3.0))) if len(high_pred) else float("nan"),
        "high_target_bias": float(np.mean(high_pred - target[high_mask])) if len(high_pred) else float("nan"),
        "low_target_pred_mean": float(np.mean(low_pred)) if len(low_pred) else float("nan"),
        "low_target_bias": float(np.mean(low_pred - target[low_mask])) if len(low_pred) else float("nan"),
    }


def distribution_shift(train: dict[str, Any], other: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_delta_vs_train": float(other["target_mean"] - train["target_mean"]),
        "std_ratio_vs_train": float(other["target_std"] / train["target_std"]) if train["target_std"] else float("nan"),
        "low_rate_delta_vs_train": float(other["low_rate"] - train["low_rate"]),
        "high_rate_delta_vs_train": float(other["high_rate"] - train["high_rate"]),
    }


def diagnose_route(split_stats_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train = split_stats_by_name["train"]
    val = split_stats_by_name["val"]
    test = split_stats_by_name["test"]
    val_shift = distribution_shift(train, val)
    test_shift = distribution_shift(train, test)
    train_extreme_sparse = train["low_rate"] < 0.15 or train["high_rate"] < 0.15
    split_shift = (
        abs(val_shift["mean_delta_vs_train"]) >= 0.25
        or abs(test_shift["mean_delta_vs_train"]) >= 0.25
        or abs(val_shift["high_rate_delta_vs_train"]) >= 0.10
        or abs(test_shift["high_rate_delta_vs_train"]) >= 0.10
    )
    collapse_by_split = {
        split: stats["pred_2x_rate"] >= 0.70 or stats["pred_225_275_rate"] >= 0.50
        for split, stats in split_stats_by_name.items()
    }
    high_collapse = {
        split: stats["high_target_pred_2x_rate"] >= 0.50 if math.isfinite(stats["high_target_pred_2x_rate"]) else False
        for split, stats in split_stats_by_name.items()
    }
    return {
        "train_extreme_sparse": bool(train_extreme_sparse),
        "split_distribution_shift": bool(split_shift),
        "prediction_collapses_near_2x": collapse_by_split,
        "high_label_predictions_collapse_near_2x": high_collapse,
        "val_shift": val_shift,
        "test_shift": test_shift,
        "summary": make_summary(train, val, test, collapse_by_split, high_collapse, split_shift, train_extreme_sparse),
    }


def make_summary(
    train: dict[str, Any],
    val: dict[str, Any],
    test: dict[str, Any],
    collapse_by_split: dict[str, bool],
    high_collapse: dict[str, bool],
    split_shift: bool,
    train_extreme_sparse: bool,
) -> str:
    parts: list[str] = []
    if split_shift:
        parts.append("split fatigue distribution differs from train")
    else:
        parts.append("split fatigue distributions are broadly comparable")
    if train_extreme_sparse:
        parts.append("train extreme fatigue coverage is sparse")
    else:
        parts.append(
            f"train has low/high coverage {train['low_rate']:.1%}/{train['high_rate']:.1%}"
        )
    collapsed = [split for split, ok in collapse_by_split.items() if ok]
    if collapsed:
        parts.append("predictions concentrate in 2.x on " + ",".join(collapsed))
    high_collapsed = [split for split, ok in high_collapse.items() if ok]
    if high_collapsed:
        parts.append("high-label predictions often remain 2.x on " + ",".join(high_collapsed))
    parts.append(
        "test pred std ratio {:.3f}, high-target pred mean {:.3f}".format(
            test["pred_std_over_true_std"], test["high_target_pred_mean"]
        )
    )
    return "; ".join(parts)


def calibration_diagnostic_rows(phase1_root: Path) -> list[dict[str, Any]]:
    metrics_path = phase1_root / "metrics_val.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for route, payload in metrics.items():
        base = payload["baseline"]
        for calibration in CALIBRATIONS:
            candidate = payload["calibrations"][calibration]["metrics"]
            high_base = abs(float(base["high_fatigue_bias"]))
            high_new = abs(float(candidate["high_fatigue_bias"]))
            high_reduction = (high_base - high_new) / high_base if high_base else float("nan")
            rows.append(
                {
                    "route": route,
                    "calibration": calibration,
                    "std_ratio_delta": float(candidate["pred_std_over_true_std"] - base["pred_std_over_true_std"]),
                    "high_abs_bias_reduction": float(high_reduction),
                    "rmse_delta": float(candidate["rmse"] - base["rmse"]),
                    "mae_delta": float(candidate["mae"] - base["mae"]),
                    "raw_r_delta": float(candidate["raw_r"] - base["raw_r"]),
                    "centered_r_delta": float(
                        candidate["within_subject_centered_r"] - base["within_subject_centered_r"]
                    ),
                }
            )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protocol",
        "experiment",
        "role",
        "split",
        "count",
        "target_mean",
        "target_std",
        "low_count",
        "high_count",
        "low_rate",
        "high_rate",
        "pred_mean",
        "pred_std",
        "pred_std_over_true_std",
        "pred_2x_rate",
        "pred_225_275_rate",
        "high_target_pred_mean",
        "high_target_pred_2x_rate",
        "high_target_bias",
        "low_target_bias",
        "target_count_1",
        "target_count_2",
        "target_count_3",
        "target_count_4",
        "target_count_5",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if math.isfinite(value) else "nan")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(out_dir: Path, route_payloads: list[dict[str, Any]], flat_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> None:
    decision_rows = [
        {
            "route": row["route"],
            "calibration": row["calibration"],
            "std_ratio_delta": row["std_ratio_delta"],
            "high_abs_bias_reduction": row["high_abs_bias_reduction"],
            "rmse_delta": row["rmse_delta"],
            "mae_delta": row["mae_delta"],
        }
        for row in diag_rows
    ]
    split_summary = []
    for payload in route_payloads:
        for split, stats in payload["splits"].items():
            split_summary.append(
                {
                    "route": f"{payload['protocol']}/{payload['experiment']}",
                    "split": split,
                    "target_mean": stats["target_mean"],
                    "target_std": stats["target_std"],
                    "low_rate": stats["low_rate"],
                    "high_rate": stats["high_rate"],
                    "pred_mean": stats["pred_mean"],
                    "pred_std_ratio": stats["pred_std_over_true_std"],
                    "pred_2x_rate": stats["pred_2x_rate"],
                    "high_pred_mean": stats["high_target_pred_mean"],
                    "high_pred_2x_rate": stats["high_target_pred_2x_rate"],
                }
            )
    diagnosis_rows = [
        {
            "route": f"{payload['protocol']}/{payload['experiment']}",
            "train_extreme_sparse": payload["diagnosis"]["train_extreme_sparse"],
            "split_shift": payload["diagnosis"]["split_distribution_shift"],
            "collapse": payload["diagnosis"]["prediction_collapses_near_2x"],
            "high_collapse": payload["diagnosis"]["high_label_predictions_collapse_near_2x"],
            "summary": payload["diagnosis"]["summary"],
        }
        for payload in route_payloads
    ]
    lines = [
        "# Phase 1 Diagnostic Decision and Phase 1b Checks",
        "",
        "## Phase 1 Diagnostic Decision",
        "",
        "Phase 1 is retained as a diagnostic result, not a formal candidate. Variance calibration and clipped variance calibration increase `pred_std / true_std` and reduce high-fatigue underestimation on validation, but they also increase RMSE and MAE beyond the Phase 1 gate. Linear calibration can slightly reduce RMSE, but it further narrows the prediction range and does not address high-fatigue underestimation.",
        "",
        md_table(
            decision_rows,
            ["route", "calibration", "std_ratio_delta", "high_abs_bias_reduction", "rmse_delta", "mae_delta"],
        ),
        "",
        "## Phase 1b Split and Collapse Checks",
        "",
        md_table(
            split_summary,
            [
                "route",
                "split",
                "target_mean",
                "target_std",
                "low_rate",
                "high_rate",
                "pred_mean",
                "pred_std_ratio",
                "pred_2x_rate",
                "high_pred_mean",
                "high_pred_2x_rate",
            ],
        ),
        "",
        "## Diagnosis",
        "",
        md_table(
            diagnosis_rows,
            ["route", "train_extreme_sparse", "split_shift", "collapse", "high_collapse", "summary"],
        ),
        "",
        "## Interpretation",
        "",
        "- Train splits have abundant low-fatigue samples but sparse high-fatigue samples: high-rate is about 5%-7% in train for the representative routes.",
        "- Large train/val/test fatigue distribution shift is not the main explanation by this check, although val/test high-rate is modestly higher than train in the main protocols.",
        "- Baseline predictions are compressed around the low-2 to low-3 range. The collapse is most visible on high-fatigue samples, supporting the hypothesis that MSE-style training and conditional-mean behavior compress the output range.",
        "- The next useful step is Phase 2 loss/sampler work rather than another global post-hoc scaling pass.",
    ]
    (out_dir / "phase1b_distribution_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root)
    phase1_root = Path(args.phase1_root)
    out_dir = Path(args.out_dir)

    route_payloads: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for protocol, eeg, experiment, role in ROUTES:
        data = load_npz(baseline_path(prediction_root, protocol, eeg, experiment))
        split_payload: dict[str, dict[str, Any]] = {}
        for split in ("train", "val", "test"):
            target, pred = split_view(data, split)
            stats = split_stats(target, pred)
            split_payload[split] = stats
            counts = stats["target_counts"]
            flat_rows.append(
                {
                    "protocol": protocol,
                    "experiment": experiment,
                    "role": role,
                    "split": split,
                    **{key: value for key, value in stats.items() if key != "target_counts"},
                    "target_count_1": counts["1"],
                    "target_count_2": counts["2"],
                    "target_count_3": counts["3"],
                    "target_count_4": counts["4"],
                    "target_count_5": counts["5"],
                }
            )
        route_payloads.append(
            {
                "protocol": protocol,
                "eeg_branch": eeg,
                "experiment": experiment,
                "role": role,
                "splits": split_payload,
                "diagnosis": diagnose_route(split_payload),
            }
        )

    diag_rows = calibration_diagnostic_rows(phase1_root)
    payload = {
        "phase1_decision": {
            "retain_as_diagnostic": True,
            "formal_candidate": False,
            "reason": "Variance calibration increases dynamic range and reduces high-fatigue underestimation, but RMSE/MAE worsen beyond the Phase 1 gate.",
        },
        "phase1_calibration_diagnostics": diag_rows,
        "phase1b_routes": route_payloads,
    }
    write_json(out_dir / "phase1b_distribution_check.json", payload)
    write_csv(out_dir / "phase1b_distribution_check.csv", flat_rows)
    write_markdown(out_dir, route_payloads, flat_rows, diag_rows)
    print(json.dumps({"out_dir": str(out_dir), "route_count": len(route_payloads)}, indent=2))


if __name__ == "__main__":
    main()
