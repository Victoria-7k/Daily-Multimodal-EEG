"""Summarize the Phase 3 ordinal/distributional-head paired screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", default="outputs/server_sync/fatigue_calibration_20260816/phase3_ordinal_head")
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
    with np.load(_openable_path(path), allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def split_arrays(data: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = data[f"{split}_index"].astype(np.int64)
    pred = data[f"{split}_prediction"].astype(np.float64)
    target = data["target"][index].astype(np.float64)
    subject = data["subject_id"][index].astype(str)
    return pred, target, subject


def prediction_path(phase3_root: Path, record: dict[str, Any]) -> Path:
    payload = json.loads(Path(record["report_json"]).read_text(encoding="utf-8"))
    if not payload.get("results"):
        raise ValueError(f"no results in {record['report_json']}")
    recorded = Path(str(payload["results"][0].get("prediction_path", "")))
    if recorded.exists():
        return recorded
    fallback = sorted((phase3_root / "predictions" / record["run_id"]).rglob("*.npz"))
    if fallback:
        return fallback[0]
    raise FileNotFoundError(recorded)


def delta_metrics(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    high_base = abs(float(baseline["high_fatigue_bias"]))
    high_new = abs(float(candidate["high_fatigue_bias"]))
    high_mae_base = float(baseline["high_fatigue_mae"])
    high_mae_new = float(candidate["high_fatigue_mae"])
    extreme_base = float(baseline["extreme_mae"])
    extreme_new = float(candidate["extreme_mae"])
    return {
        "rmse_delta": float(candidate["rmse"] - baseline["rmse"]),
        "mae_delta": float(candidate["mae"] - baseline["mae"]),
        "raw_r_delta": float(candidate["raw_r"] - baseline["raw_r"]),
        "centered_r_delta": float(candidate["within_subject_centered_r"] - baseline["within_subject_centered_r"]),
        "std_ratio_delta": float(candidate["pred_std_over_true_std"] - baseline["pred_std_over_true_std"]),
        "high_abs_bias_reduction": float((high_base - high_new) / high_base) if high_base else float("nan"),
        "high_mae_reduction": float((high_mae_base - high_mae_new) / high_mae_base) if high_mae_base else float("nan"),
        "extreme_mae_reduction": float((extreme_base - extreme_new) / extreme_base) if extreme_base else float("nan"),
    }


def confusion_rows(data: dict[str, np.ndarray], split: str) -> list[dict[str, Any]]:
    pred, target, _ = split_arrays(data, split)
    pred_label = np.clip(np.rint(pred), 1, 5).astype(int)
    true_label = np.clip(np.rint(target), 1, 5).astype(int)
    rows = []
    for truth in range(1, 6):
        mask = true_label == truth
        total = int(np.sum(mask))
        row: dict[str, Any] = {"true_label": truth, "count": total}
        for pred_value in range(1, 6):
            row[f"pred_{pred_value}"] = int(np.sum(pred_label[mask] == pred_value)) if total else 0
        rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    high_or_mae = max(
        mean([row["val_high_abs_bias_reduction"] for row in rows]),
        mean([row["val_high_mae_reduction"] for row in rows]),
    )
    raw_delta = mean([row["val_raw_r_delta"] for row in rows])
    centered_delta = mean([row["val_centered_r_delta"] for row in rows])
    summary = {
        "candidate_id": rows[0]["candidate_id"],
        "protocol": rows[0]["protocol"],
        "experiment": rows[0]["experiment"],
        "head": rows[0]["head"],
        "head_lambda": rows[0]["head_lambda"],
        "seed_count": len(rows),
        "mean_val_rmse_delta": mean([row["val_rmse_delta"] for row in rows]),
        "mean_val_mae_delta": mean([row["val_mae_delta"] for row in rows]),
        "mean_val_raw_r_delta": raw_delta,
        "mean_val_centered_r_delta": centered_delta,
        "mean_val_std_ratio_delta": mean([row["val_std_ratio_delta"] for row in rows]),
        "mean_val_high_abs_bias_reduction": mean([row["val_high_abs_bias_reduction"] for row in rows]),
        "mean_val_high_mae_reduction": mean([row["val_high_mae_reduction"] for row in rows]),
        "mean_val_extreme_mae_reduction": mean([row["val_extreme_mae_reduction"] for row in rows]),
        "mean_test_rmse_delta": mean([row["test_rmse_delta"] for row in rows]),
        "mean_test_raw_r_delta": mean([row["test_raw_r_delta"] for row in rows]),
        "mean_test_centered_r_delta": mean([row["test_centered_r_delta"] for row in rows]),
        "positive_seed_count": sum(1 for row in rows if float(row["val_std_ratio_delta"]) >= 0.0 and float(row["val_high_abs_bias_reduction"]) > 0.0),
    }
    passes = (
        float(summary["mean_val_std_ratio_delta"]) >= 0.100
        and high_or_mae >= 0.100
        and raw_delta >= -0.010
        and centered_delta >= -0.015
        and float(summary["mean_val_rmse_delta"]) <= 0.015
        and int(summary["positive_seed_count"]) >= 2
    )
    summary["passes_phase4_gate"] = bool(passes)
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
    phase3_root = Path(args.phase3_root)
    manifest = json.loads((phase3_root / "configs" / "manifest.json").read_text(encoding="utf-8"))
    baselines = {(row["pair_id"], int(row["seed"])): row for row in manifest if row["candidate_role"] == "baseline"}
    pair_rows: list[dict[str, Any]] = []
    confusion_blocks: list[str] = ["# Phase 3 Confusion / Bin Summary", ""]
    for row in manifest:
        if row["candidate_role"] == "baseline":
            continue
        baseline = baselines[(row["pair_id"], int(row["seed"]))]
        cand_npz = load_npz(prediction_path(phase3_root, row))
        base_npz = load_npz(prediction_path(phase3_root, baseline))
        record: dict[str, Any] = {
            "candidate_id": row["candidate_id"],
            "protocol": row["protocol"],
            "experiment": row["experiment"],
            "head": row["head"],
            "head_lambda": float(row["head_lambda"]),
            "sampler": row["sampler"],
            "seed": int(row["seed"]),
            "candidate_prediction_path": str(prediction_path(phase3_root, row)),
            "baseline_prediction_path": str(prediction_path(phase3_root, baseline)),
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
        if int(row["seed"]) == 240800:
            confusion_blocks.extend(
                [
                    f"## {row['candidate_id']} / {row['protocol']} / {row['experiment']} / seed {row['seed']}",
                    "",
                    "Validation rounded prediction bins:",
                    "",
                    markdown_table(confusion_rows(cand_npz, "val"), ["true_label", "count", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"]),
                    "",
                ]
            )

    summaries = [summarize_group(rows) for _, rows in sorted(group_by(pair_rows, ["candidate_id", "protocol", "experiment"]).items())]
    summary_keys = [
        "candidate_id",
        "protocol",
        "experiment",
        "head",
        "head_lambda",
        "seed_count",
        "mean_val_rmse_delta",
        "mean_val_mae_delta",
        "mean_val_raw_r_delta",
        "mean_val_centered_r_delta",
        "mean_val_std_ratio_delta",
        "mean_val_high_abs_bias_reduction",
        "mean_val_high_mae_reduction",
        "positive_seed_count",
        "passes_phase4_gate",
        "mean_test_rmse_delta",
        "mean_test_raw_r_delta",
        "mean_test_centered_r_delta",
    ]
    pair_keys = [
        "candidate_id",
        "protocol",
        "experiment",
        "head",
        "head_lambda",
        "sampler",
        "seed",
        "val_rmse_delta",
        "val_mae_delta",
        "val_raw_r_delta",
        "val_centered_r_delta",
        "val_std_ratio_delta",
        "val_high_abs_bias_reduction",
        "val_high_mae_reduction",
        "test_rmse_delta",
        "test_raw_r_delta",
        "test_centered_r_delta",
        "candidate_prediction_path",
        "baseline_prediction_path",
    ]
    write_csv(phase3_root / "paired_seed_deltas.csv", pair_rows, pair_keys)
    write_csv(phase3_root / "paired_gate_summary.csv", summaries, summary_keys)
    (phase3_root / "metrics_val.json").write_text(
        json.dumps({"summaries": summaries, "paired_rows": pair_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase 3 Ordinal/Distributional Head Paired Gate",
        "",
        "Phase 2 did not pass its paired gate, so this screen uses raw MSE baseline settings with alternative heads.",
        "",
        "## Candidate Gate Summary",
        "",
        markdown_table(summaries, summary_keys),
        "",
        "## Per-Seed Val/Test Deltas",
        "",
        markdown_table(pair_rows, pair_keys[:-2]),
    ]
    (phase3_root / "paired_delta_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (phase3_root / "confusion_or_bin_summary.md").write_text("\n".join(confusion_blocks) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_route_count": len(summaries), "passed": sum(1 for row in summaries if row["passes_phase4_gate"])}, indent=2))


def group_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


if __name__ == "__main__":
    main()
