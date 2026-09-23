"""Summarize Phase 2 fatigue loss/sampler runs against the raw baseline."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-prediction-root",
        default="outputs/server_sync/eeg_encoder_256d_5route_20260814/predictions/"
        "eeg_encoder_256d_5route_fusion_video_only_seed240800_raw",
    )
    parser.add_argument(
        "--phase2-root",
        default="outputs/server_sync/fatigue_calibration_20260816/phase2_loss_sampler",
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
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


def baseline_path(root: Path, route: RouteKey) -> Path:
    return root / route.protocol / route.eeg_branch / route.experiment / "raw_lambda_0.npz"


def phase2_path(phase2_root: Path, config: str, row: dict[str, Any]) -> Path:
    recorded = Path(str(row.get("prediction_path", "")))
    if recorded.exists():
        return recorded
    parts = recorded.parts
    if "outputs" in parts:
        suffix = Path(*parts[parts.index("outputs") + 1 :])
        candidate = phase2_root.parents[2] / suffix if len(phase2_root.parents) >= 3 else suffix
        if candidate.exists():
            return candidate
    lam = float(row["centered_lambda"])
    name = f"{row['loss_mode']}_lambda_{lam:g}.npz"
    candidate = phase2_root / "predictions" / config / row["protocol"] / row["eeg_branch"] / row["experiment"] / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(candidate)


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
    length_ok = all(len(data[key]) == row_count for key in ("target", "sample_id", "subject_id", "event_id") if key in data)
    pred_length_ok = all(
        len(data[f"{split}_prediction"]) == len(data[f"{split}_index"])
        for split in ("train", "val", "test")
        if f"{split}_prediction" in data and f"{split}_index" in data
    )
    return {"missing_keys": missing, "length_ok": bool(length_ok), "prediction_length_ok": bool(pred_length_ok)}


def delta_metrics(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float | bool]:
    high_base = abs(float(baseline["high_fatigue_bias"]))
    high_new = abs(float(metrics["high_fatigue_bias"]))
    low_base = abs(float(baseline["low_fatigue_bias"]))
    low_new = abs(float(metrics["low_fatigue_bias"]))
    high_reduction = (high_base - high_new) / high_base if high_base else float("nan")
    low_reduction = (low_base - low_new) / low_base if low_base else float("nan")
    deltas: dict[str, float | bool] = {
        "rmse_delta": float(metrics["rmse"] - baseline["rmse"]),
        "mae_delta": float(metrics["mae"] - baseline["mae"]),
        "raw_r_delta": float(metrics["raw_r"] - baseline["raw_r"]),
        "centered_r_delta": float(metrics["within_subject_centered_r"] - baseline["within_subject_centered_r"]),
        "std_ratio_delta": float(metrics["pred_std_over_true_std"] - baseline["pred_std_over_true_std"]),
        "high_abs_bias_reduction": float(high_reduction),
        "low_abs_bias_reduction": float(low_reduction),
    }
    no_metric_regression = (
        float(deltas["rmse_delta"]) <= 0.020
        and float(deltas["mae_delta"]) <= 0.020
        and float(deltas["raw_r_delta"]) >= -0.010
        and float(deltas["centered_r_delta"]) >= -0.015
    )
    useful_shift = (
        float(deltas["std_ratio_delta"]) >= 0.050
        or float(deltas["high_abs_bias_reduction"]) >= 0.050
        or float(deltas["rmse_delta"]) <= -0.010
        or float(deltas["raw_r_delta"]) >= 0.015
        or float(deltas["centered_r_delta"]) >= 0.015
    )
    deltas["passes_phase2_gate"] = bool(no_metric_regression and useful_shift)
    return deltas


def read_phase2_rows(phase2_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted((phase2_root / "reports").glob("*.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        config = report_path.stem
        for row in payload.get("results", []):
            rows.append({"config": config, **row})
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "nan"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> str:
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = ["| " + " | ".join(fmt(row.get(key, "")) for key in keys) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def event_day(event_id: str) -> str:
    for part in event_id.split("_"):
        if part.startswith("day-"):
            return part
    return event_id


def selected_subject_day(data: dict[str, np.ndarray], split: str) -> np.ndarray:
    index = data[f"{split}_index"].astype(np.int64)
    target = data["target"][index].astype(np.float64)
    subjects = data["subject_id"][index].astype(str)
    events = data["event_id"][index].astype(str)
    keys = np.array([f"{sid} {event_day(event)}" for sid, event in zip(subjects, events)], dtype=object)
    best_key = ""
    best_score = (-1.0, -1)
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        score = (float(np.std(target[mask])), int(np.sum(mask)))
        if score > best_score:
            best_key = key
            best_score = score
    return keys == best_key


def best_rows_for_visual(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = str(row["route"])
        current = chosen.get(slug)
        key = (
            1 if row.get("passes_phase2_gate") else 0,
            -float(row.get("val_rmse", float("inf"))),
            float(row.get("val_raw_r", float("-inf"))),
            float(row.get("val_centered_r", float("-inf"))),
        )
        if current is None or key > current["_visual_key"]:
            copied = dict(row)
            copied["_visual_key"] = key
            chosen[slug] = copied
    return chosen


def draw_phase2_visual(path: Path, rows: list[dict[str, Any]], baseline_root: Path, phase2_root: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return

    selected = best_rows_for_visual(rows)
    routes = [RouteKey(*values) for values in REPRESENTATIVE_ROUTES]
    panel_w, panel_h = 760, 330
    margin = 36
    image = Image.new("RGB", (panel_w * 2 + margin * 3, panel_h * 2 + margin * 3 + 70), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = "Phase 2 best val-gated route trends on selected test subject-days"
    draw.text((margin, 18), title, fill=(10, 16, 28), font=font)
    colors = {"true": (16, 24, 40), "baseline": (80, 100, 130), "phase2": (220, 38, 38)}

    def line(points: list[tuple[float, float]], color: tuple[int, int, int], width: int) -> None:
        if len(points) >= 2:
            draw.line(points, fill=color, width=width)

    for i, route in enumerate(routes):
        left = margin + (i % 2) * (panel_w + margin)
        top = margin + 50 + (i // 2) * (panel_h + margin)
        right = left + panel_w
        bottom = top + panel_h
        draw.rectangle((left, top, right, bottom), outline=(210, 218, 230), width=1)
        base_data = load_npz(baseline_path(baseline_root, route))
        chosen = selected.get(route.slug)
        if not chosen:
            continue
        phase_data = load_npz(Path(chosen["prediction_path"]))
        mask = selected_subject_day(base_data, "test")
        x = np.arange(int(np.sum(mask)))
        test_index = base_data["test_index"].astype(np.int64)
        target = base_data["target"][test_index][mask].astype(np.float64)
        base_pred = base_data["test_prediction"].astype(np.float64)[mask]
        phase_pred = phase_data["test_prediction"].astype(np.float64)[mask]
        ymin = float(min(np.min(target), np.min(base_pred), np.min(phase_pred))) - 0.15
        ymax = float(max(np.max(target), np.max(base_pred), np.max(phase_pred))) + 0.15
        if ymax <= ymin:
            ymax = ymin + 1.0
        plot_l, plot_t = left + 56, top + 46
        plot_r, plot_b = right - 24, bottom - 42
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = plot_b - frac * (plot_b - plot_t)
            draw.line((plot_l, yy, plot_r, yy), fill=(230, 235, 242), width=1)
        draw.text((left + 14, top + 12), f"{route.protocol} / {route.experiment}", fill=(10, 16, 28), font=font)
        draw.text(
            (left + 14, top + 26),
            f"{chosen['config']} | gate={chosen['passes_phase2_gate']} | test r {chosen['test_raw_r']:.3f}",
            fill=(70, 82, 100),
            font=font,
        )

        def to_points(values: np.ndarray) -> list[tuple[float, float]]:
            if len(values) == 1:
                xs = np.array([0.0])
            else:
                xs = x / float(max(1, len(values) - 1))
            ys = (values - ymin) / (ymax - ymin)
            return [(plot_l + float(xx) * (plot_r - plot_l), plot_b - float(yy) * (plot_b - plot_t)) for xx, yy in zip(xs, ys)]

        line(to_points(target), colors["true"], 4)
        line(to_points(base_pred), colors["baseline"], 3)
        line(to_points(phase_pred), colors["phase2"], 3)
        draw.text((plot_l, plot_b + 12), "true", fill=colors["true"], font=font)
        draw.text((plot_l + 70, plot_b + 12), "baseline", fill=colors["baseline"], font=font)
        draw.text((plot_l + 170, plot_b + 12), "phase2", fill=colors["phase2"], font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_prediction_root)
    phase2_root = Path(args.phase2_root)
    routes = {RouteKey(*values).slug: RouteKey(*values) for values in REPRESENTATIVE_ROUTES}
    baseline_metrics: dict[str, dict[str, dict[str, float]]] = {}
    baseline_audit: dict[str, Any] = {}

    for route in routes.values():
        data = load_npz(baseline_path(baseline_root, route))
        baseline_audit[route.slug] = audit_npz(data, args.row_count)
        baseline_metrics[route.slug] = {}
        for split in ("train", "val", "test"):
            pred, target, subject = split_arrays(data, split)
            baseline_metrics[route.slug][split] = metrics_for(pred, target, subject)

    rows: list[dict[str, Any]] = []
    for row in read_phase2_rows(phase2_root):
        route = RouteKey(row["protocol"], row["eeg_branch"], row["experiment"], "")
        if route.slug not in routes:
            continue
        pred_path = phase2_path(phase2_root, row["config"], row)
        data = load_npz(pred_path)
        record: dict[str, Any] = {
            "config": row["config"],
            "route": route.slug,
            "protocol": route.protocol,
            "eeg_branch": route.eeg_branch,
            "experiment": route.experiment,
            "loss_mode": row["loss_mode"],
            "lambda": float(row["centered_lambda"]),
            "sampler": row.get("train_audit", {}).get("sampler", ""),
            "seed": row.get("seed", ""),
            "prediction_path": str(pred_path),
        }
        for split in ("val", "test"):
            pred, target, subject = split_arrays(data, split)
            metrics = metrics_for(pred, target, subject)
            for key, value in metrics.items():
                record[f"{split}_{key}"] = value
            if split == "val":
                for key, value in delta_metrics(metrics, baseline_metrics[route.slug][split]).items():
                    record[key] = value
        rows.append(record)

    rows.sort(
        key=lambda row: (
            row["protocol"],
            row["experiment"],
            0 if row.get("passes_phase2_gate") else 1,
            float(row["val_rmse"]),
            -float(row["val_raw_r"]),
        )
    )
    fieldnames = [
        "config",
        "protocol",
        "eeg_branch",
        "experiment",
        "loss_mode",
        "lambda",
        "sampler",
        "seed",
        "val_rmse",
        "val_mae",
        "val_raw_r",
        "val_within_subject_centered_r",
        "val_pred_std_over_true_std",
        "val_low_fatigue_bias",
        "val_high_fatigue_bias",
        "rmse_delta",
        "mae_delta",
        "raw_r_delta",
        "centered_r_delta",
        "std_ratio_delta",
        "high_abs_bias_reduction",
        "low_abs_bias_reduction",
        "passes_phase2_gate",
        "test_rmse",
        "test_mae",
        "test_raw_r",
        "test_within_subject_centered_r",
        "test_pred_std_over_true_std",
        "test_low_fatigue_bias",
        "test_high_fatigue_bias",
        "prediction_path",
    ]
    write_csv(phase2_root / "metrics_val_test_with_deltas.csv", rows, fieldnames)
    (phase2_root / "metrics_val_test_with_deltas.json").write_text(
        json.dumps({"baseline": baseline_metrics, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pass_rows = [row for row in rows if row.get("passes_phase2_gate")]
    best_by_route = list(best_rows_for_visual(rows).values())
    lines = [
        "# Phase 2 Loss/Sampler Summary",
        "",
        "Phase 2 changes training loss/sampling and is compared against the same raw baseline predictions.",
        "The gate is validation-only: RMSE/MAE, raw r, and centered r must stay within a small tolerance, while either prediction spread, high-fatigue bias, RMSE, raw r, or centered r must improve.",
        "",
        f"- Completed Phase 2 rows: `{len(rows)}`",
        f"- Val-gated rows: `{len(pass_rows)}`",
        "- Phase 1 remains a diagnostic result: variance calibration can increase `pred_std / true_std` and reduce high-fatigue underestimation, but it worsens RMSE/MAE enough that it is not a formal candidate.",
        "",
        "## Val-Gated Rows",
        "",
        markdown_table(
            pass_rows,
            [
                "config",
                "protocol",
                "experiment",
                "loss_mode",
                "lambda",
                "val_rmse",
                "val_raw_r",
                "val_within_subject_centered_r",
                "val_pred_std_over_true_std",
                "rmse_delta",
                "raw_r_delta",
                "centered_r_delta",
                "std_ratio_delta",
                "high_abs_bias_reduction",
                "passes_phase2_gate",
            ],
        )
        if pass_rows
        else "No Phase 2 row passed the validation gate.",
        "",
        "## Best Row Per Route",
        "",
        markdown_table(
            best_by_route,
            [
                "config",
                "protocol",
                "experiment",
                "loss_mode",
                "lambda",
                "passes_phase2_gate",
                "val_rmse",
                "val_raw_r",
                "val_within_subject_centered_r",
                "test_rmse",
                "test_raw_r",
                "test_within_subject_centered_r",
                "test_pred_std_over_true_std",
                "test_high_fatigue_bias",
            ],
        ),
        "",
        "## Baseline Val/Test Reference",
        "",
        markdown_table(
            [
                {
                    "protocol": route.protocol,
                    "experiment": route.experiment,
                    "val_rmse": baseline_metrics[route.slug]["val"]["rmse"],
                    "val_raw_r": baseline_metrics[route.slug]["val"]["raw_r"],
                    "val_centered_r": baseline_metrics[route.slug]["val"]["within_subject_centered_r"],
                    "val_std_ratio": baseline_metrics[route.slug]["val"]["pred_std_over_true_std"],
                    "test_rmse": baseline_metrics[route.slug]["test"]["rmse"],
                    "test_raw_r": baseline_metrics[route.slug]["test"]["raw_r"],
                    "test_centered_r": baseline_metrics[route.slug]["test"]["within_subject_centered_r"],
                    "test_std_ratio": baseline_metrics[route.slug]["test"]["pred_std_over_true_std"],
                }
                for route in routes.values()
            ],
            [
                "protocol",
                "experiment",
                "val_rmse",
                "val_raw_r",
                "val_centered_r",
                "val_std_ratio",
                "test_rmse",
                "test_raw_r",
                "test_centered_r",
                "test_std_ratio",
            ],
        ),
    ]
    (phase2_root / "paired_delta_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    draw_phase2_visual(
        phase2_root / "visualizations" / "phase2_best_route_trends_test.png",
        rows,
        baseline_root,
        phase2_root,
    )
    print(json.dumps({"phase2_rows": len(rows), "val_gated_rows": len(pass_rows)}, indent=2))


if __name__ == "__main__":
    main()
