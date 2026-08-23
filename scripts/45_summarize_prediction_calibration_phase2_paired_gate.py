"""Summarize the Phase 2 three-seed paired gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", default="outputs/server_sync/fatigue_calibration_20260816/phase2_paired_gate")
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
    pred_std = float(np.std(pred))
    true_std = float(np.std(target))
    return {
        "count": int(len(target)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "raw_r": pearson(pred, target),
        "within_subject_centered_r": centered_r(pred, target, subject),
        "pred_std": pred_std,
        "true_std": true_std,
        "pred_std_over_true_std": float(pred_std / true_std) if true_std else float("nan"),
        "low_fatigue_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "high_fatigue_bias": float(np.mean(err[high])) if np.any(high) else float("nan"),
        "low_fatigue_mae": float(np.mean(np.abs(err[low]))) if np.any(low) else float("nan"),
        "high_fatigue_mae": float(np.mean(np.abs(err[high]))) if np.any(high) else float("nan"),
        "extreme_mae": float(np.mean(np.abs(err[low | high]))) if np.any(low | high) else float("nan"),
        "low_count": int(np.sum(low)),
        "high_count": int(np.sum(high)),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def split_arrays(data: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = data[f"{split}_index"].astype(np.int64)
    pred = data[f"{split}_prediction"].astype(np.float64)
    target = data["target"][index].astype(np.float64)
    subject = data["subject_id"][index].astype(str)
    return pred, target, subject


def prediction_path(phase2_root: Path, record: dict[str, Any]) -> Path:
    report_path = Path(record["report_json"])
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not payload.get("results"):
        raise ValueError(f"no results in {report_path}")
    result = payload["results"][0]
    recorded = Path(str(result.get("prediction_path", "")))
    if recorded.exists():
        return recorded
    candidate = (
        phase2_root
        / "predictions"
        / record["run_id"]
        / record["protocol"]
        / record["eeg_branch"]
        / record["experiment"]
        / f"{record['loss_mode']}_lambda_{float(record['lambda']):g}.npz"
    )
    if candidate.exists():
        return candidate
    raise FileNotFoundError(candidate)


def delta_metrics(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    high_base = abs(float(baseline["high_fatigue_bias"]))
    high_new = abs(float(candidate["high_fatigue_bias"]))
    extreme_base = float(baseline["extreme_mae"])
    extreme_new = float(candidate["extreme_mae"])
    return {
        "rmse_delta": float(candidate["rmse"] - baseline["rmse"]),
        "mae_delta": float(candidate["mae"] - baseline["mae"]),
        "raw_r_delta": float(candidate["raw_r"] - baseline["raw_r"]),
        "centered_r_delta": float(candidate["within_subject_centered_r"] - baseline["within_subject_centered_r"]),
        "std_ratio_delta": float(candidate["pred_std_over_true_std"] - baseline["pred_std_over_true_std"]),
        "high_abs_bias_reduction": float((high_base - high_new) / high_base) if high_base else float("nan"),
        "extreme_mae_reduction": float((extreme_base - extreme_new) / extreme_base) if extreme_base else float("nan"),
    }


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_candidate(candidate_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    val_rows = rows
    primary_metric = "raw_r_delta" if mean([row["val_raw_r_delta"] for row in val_rows]) >= mean(
        [row["val_centered_r_delta"] for row in val_rows]
    ) else "centered_r_delta"
    positive_count = sum(1 for row in val_rows if float(row[f"val_{primary_metric}"]) > 0.0)
    high_or_extreme = max(
        mean([row["val_high_abs_bias_reduction"] for row in val_rows]),
        mean([row["val_extreme_mae_reduction"] for row in val_rows]),
    )
    summary = {
        "candidate_id": candidate_id,
        "role": rows[0]["candidate_role"],
        "protocol": rows[0]["protocol"],
        "experiment": rows[0]["experiment"],
        "loss_mode": rows[0]["loss_mode"],
        "lambda": rows[0]["lambda"],
        "sampler": rows[0]["sampler"],
        "seed_count": len(rows),
        "mean_val_rmse_delta": mean([row["val_rmse_delta"] for row in val_rows]),
        "mean_val_mae_delta": mean([row["val_mae_delta"] for row in val_rows]),
        "mean_val_raw_r_delta": mean([row["val_raw_r_delta"] for row in val_rows]),
        "mean_val_centered_r_delta": mean([row["val_centered_r_delta"] for row in val_rows]),
        "mean_val_std_ratio_delta": mean([row["val_std_ratio_delta"] for row in val_rows]),
        "mean_val_high_abs_bias_reduction": mean([row["val_high_abs_bias_reduction"] for row in val_rows]),
        "mean_val_extreme_mae_reduction": mean([row["val_extreme_mae_reduction"] for row in val_rows]),
        "mean_test_rmse_delta": mean([row["test_rmse_delta"] for row in val_rows]),
        "mean_test_raw_r_delta": mean([row["test_raw_r_delta"] for row in val_rows]),
        "mean_test_centered_r_delta": mean([row["test_centered_r_delta"] for row in val_rows]),
        "primary_metric": primary_metric,
        "positive_seed_count": int(positive_count),
    }
    passes = (
        (
            float(summary["mean_val_raw_r_delta"]) >= 0.015
            or float(summary["mean_val_centered_r_delta"]) >= 0.020
        )
        and float(summary["mean_val_rmse_delta"]) <= 0.010
        and float(summary["mean_val_mae_delta"]) <= 0.010
        and float(summary["mean_val_std_ratio_delta"]) >= 0.100
        and float(high_or_extreme) >= 0.150
        and positive_count >= 2
    )
    summary["passes_phase3_gate"] = bool(passes)
    return summary


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "nan"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(keys) + " |",
            "| " + " | ".join("---" for _ in keys) + " |",
            *["| " + " | ".join(fmt(row.get(key, "")) for key in keys) + " |" for row in rows],
        ]
    )


def write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def main() -> None:
    args = parse_args()
    phase2_root = Path(args.phase2_root)
    manifest_path = phase2_root / "configs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baselines = {
        (row["pair_id"], int(row["seed"])): row
        for row in manifest
        if row["candidate_role"] == "baseline"
    }
    pair_rows: list[dict[str, Any]] = []
    for row in manifest:
        if row["candidate_role"] == "baseline":
            continue
        baseline = baselines[(row["pair_id"], int(row["seed"]))]
        cand_npz = load_npz(prediction_path(phase2_root, row))
        base_npz = load_npz(prediction_path(phase2_root, baseline))
        record: dict[str, Any] = {
            "candidate_id": row["candidate_id"],
            "candidate_role": row["candidate_role"],
            "protocol": row["protocol"],
            "experiment": row["experiment"],
            "loss_mode": row["loss_mode"],
            "lambda": float(row["lambda"]),
            "sampler": row["sampler"],
            "seed": int(row["seed"]),
            "candidate_prediction_path": str(prediction_path(phase2_root, row)),
            "baseline_prediction_path": str(prediction_path(phase2_root, baseline)),
        }
        for split in ("val", "test"):
            cand_metrics = metrics_for(*split_arrays(cand_npz, split))
            base_metrics = metrics_for(*split_arrays(base_npz, split))
            deltas = delta_metrics(cand_metrics, base_metrics)
            for key, value in cand_metrics.items():
                record[f"{split}_{key}"] = value
            for key, value in base_metrics.items():
                record[f"baseline_{split}_{key}"] = value
            for key, value in deltas.items():
                record[f"{split}_{key}"] = value
        pair_rows.append(record)

    summaries = [
        summarize_candidate(candidate_id, rows)
        for candidate_id, rows in sorted(group_by(pair_rows, "candidate_id").items())
    ]
    summary_keys = [
        "candidate_id",
        "role",
        "protocol",
        "experiment",
        "loss_mode",
        "lambda",
        "seed_count",
        "mean_val_rmse_delta",
        "mean_val_mae_delta",
        "mean_val_raw_r_delta",
        "mean_val_centered_r_delta",
        "mean_val_std_ratio_delta",
        "mean_val_high_abs_bias_reduction",
        "mean_val_extreme_mae_reduction",
        "positive_seed_count",
        "passes_phase3_gate",
        "mean_test_rmse_delta",
        "mean_test_raw_r_delta",
        "mean_test_centered_r_delta",
    ]
    pair_keys = [
        "candidate_id",
        "candidate_role",
        "protocol",
        "experiment",
        "loss_mode",
        "lambda",
        "sampler",
        "seed",
        "val_rmse_delta",
        "val_mae_delta",
        "val_raw_r_delta",
        "val_centered_r_delta",
        "val_std_ratio_delta",
        "val_high_abs_bias_reduction",
        "val_extreme_mae_reduction",
        "test_rmse_delta",
        "test_raw_r_delta",
        "test_centered_r_delta",
        "candidate_prediction_path",
        "baseline_prediction_path",
    ]
    write_csv(phase2_root / "paired_seed_deltas.csv", pair_rows, pair_keys)
    write_csv(phase2_root / "paired_gate_summary.csv", summaries, summary_keys)
    (phase2_root / "paired_gate_summary.json").write_text(
        json.dumps({"summaries": summaries, "paired_rows": pair_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase 2 Three-Seed Paired Gate",
        "",
        "Gate follows `prediction_calibration_handoff_20260816.md`: 3-seed val paired delta is used to decide whether Phase 3 ordinal/distributional heads should run.",
        "",
        "## Candidate Gate Summary",
        "",
        markdown_table(summaries, summary_keys),
        "",
        "## Per-Seed Val/Test Deltas",
        "",
        markdown_table(pair_rows, pair_keys[:-2]),
    ]
    (phase2_root / "paired_gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(summaries), "passed": sum(1 for row in summaries if row["passes_phase3_gate"])}, indent=2))


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


if __name__ == "__main__":
    main()
