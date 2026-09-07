"""Visualize Phase 1 post-hoc calibration against baseline predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROUTES = (
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_full", "highest raw r"),
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_no_audio", "lowest RMSE"),
    ("within_subject_day", "eeg_eegpt_partial_ft_v1", "B0_Wdeep_no_audio", "highest raw/centered r"),
    ("within_subject_day", "eeg_eegpt_partial_ft_v1", "A2_Wdeep_full", "lowest RMSE"),
)

CALIBRATIONS = (
    ("baseline", "Baseline", "#2563eb", "-"),
    ("linear_calibration", "Linear", "#16a34a", "-"),
    ("variance_calibration", "Variance", "#dc2626", "-"),
    ("clipped_variance_calibration", "Clipped variance", "#9333ea", "-"),
)


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
        default="outputs/server_sync/fatigue_calibration_20260816/phase1_posthoc_calibration/visualizations",
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
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


def day_key(event_id: str) -> str:
    for part in event_id.split("_"):
        if part.startswith("day-"):
            return part
    return event_id


def split_view(data: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    index = data[f"{split}_index"].astype(np.int64)
    return {
        "index": index,
        "prediction": data[f"{split}_prediction"].astype(np.float64),
        "target": data["target"][index].astype(np.float64),
        "subject": data["subject_id"][index].astype(str),
        "event": data["event_id"][index].astype(str),
    }


def selected_subject_day(view: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    keys = np.array(
        [f"{subject} {day_key(event)}" for subject, event in zip(view["subject"], view["event"])],
        dtype=object,
    )
    best_key = ""
    best_mask = np.zeros(len(keys), dtype=bool)
    best_score = (-1.0, -1)
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        if int(np.sum(mask)) < 2:
            continue
        true_std = float(np.std(view["target"][mask]))
        score = (true_std, int(np.sum(mask)))
        if score > best_score:
            best_key = key
            best_mask = mask
            best_score = score
    return best_key, best_mask


def metric_pair(data: dict[str, np.ndarray], split: str) -> dict[str, float]:
    view = split_view(data, split)
    return {
        "raw_r": pearson(view["prediction"], view["target"]),
        "centered_r": centered_r(view["prediction"], view["target"], view["subject"]),
        "rmse": float(np.sqrt(np.mean((view["prediction"] - view["target"]) ** 2))),
        "mae": float(np.mean(np.abs(view["prediction"] - view["target"]))),
        "std_ratio": float(np.std(view["prediction"]) / np.std(view["target"])),
    }


def calibration_path(phase1_root: Path, protocol: str, eeg: str, experiment: str, calibration: str) -> Path:
    return phase1_root / "predictions" / protocol / eeg / experiment / f"{calibration}.npz"


def baseline_path(prediction_root: Path, protocol: str, eeg: str, experiment: str) -> Path:
    return prediction_root / protocol / eeg / experiment / "raw_lambda_0.npz"


def write_comparison(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase1_raw_centered_r_comparison.csv"
    fieldnames = [
        "split",
        "protocol",
        "experiment",
        "calibration",
        "old_raw_r",
        "new_raw_r",
        "delta_raw_r",
        "old_centered_r",
        "new_centered_r",
        "delta_centered_r",
        "old_rmse",
        "new_rmse",
        "delta_rmse",
        "old_std_ratio",
        "new_std_ratio",
        "delta_std_ratio",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    md_path = out_dir / "phase1_raw_centered_r_comparison.md"
    lines = [
        "# Phase 1 Raw R and Centered R Comparison",
        "",
        "Calibration parameters were fitted on val predictions. Test rows are frozen evaluation, not selection.",
        "",
        "| split | route | calibration | old raw r | new raw r | delta | old centered r | new centered r | delta | delta RMSE | delta std ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        route = f"{row['protocol']}/{row['experiment']}"
        lines.append(
            "| {split} | {route} | {calibration} | {old_raw_r:.4f} | {new_raw_r:.4f} | {delta_raw_r:.4f} | "
            "{old_centered_r:.4f} | {new_centered_r:.4f} | {delta_centered_r:.4f} | {delta_rmse:.4f} | "
            "{delta_std_ratio:.4f} |".format(route=route, **row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trends(prediction_root: Path, phase1_root: Path, out_dir: Path, split: str) -> Path:
    width, height = 3200, 1800
    margin_x, margin_top, margin_bottom = 130, 185, 180
    gap_x, gap_y = 90, 130
    panel_w = (width - margin_x * 2 - gap_x) // 2
    panel_h = (height - margin_top - margin_bottom - gap_y) // 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = load_font(56)
    font_panel = load_font(30)
    font_text = load_font(24)
    font_small = load_font(21)
    draw.text(
        (margin_x, 35),
        f"Phase 1 post-hoc calibration: true fatigue vs predictions ({split})",
        fill="#111827",
        font=font_title,
    )

    legend_items = [("True fatigue", "#111827"), *[(label, color) for _cal, label, color, _style in CALIBRATIONS]]
    legend_x = margin_x
    legend_y = height - 105
    for label, color in legend_items:
        draw.line((legend_x, legend_y, legend_x + 65, legend_y), fill=color, width=7)
        draw.text((legend_x + 80, legend_y - 16), label, fill="#111827", font=font_text)
        legend_x += 420 if label == "Clipped variance" else 310

    for idx, (protocol, eeg, experiment, role) in enumerate(ROUTES):
        row, col = divmod(idx, 2)
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_top + row * (panel_h + gap_y)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        baseline = load_npz(baseline_path(prediction_root, protocol, eeg, experiment))
        base_view = split_view(baseline, split)
        key, mask = selected_subject_day(base_view)
        draw_panel_axes(draw, x0, y0, x1, y1, font_small)
        title = f"{protocol} / {experiment}"
        subtitle = f"{role}; {key}; n={int(np.sum(mask))}"
        draw.text((x0, y0 - 74), title, fill="#111827", font=font_panel)
        draw.text((x0, y0 - 36), subtitle, fill="#4b5563", font=font_text)
        draw_series(draw, base_view["target"][mask], x0, y0, x1, y1, "#111827", width=7)
        for calibration, label, color, linestyle in CALIBRATIONS:
            data = baseline if calibration == "baseline" else load_npz(calibration_path(phase1_root, protocol, eeg, experiment, calibration))
            view = split_view(data, split)
            _ = label, linestyle
            draw_series(draw, view["prediction"][mask], x0, y0, x1, y1, color, width=4)
    out_path = out_dir / f"phase1_calibration_trends_{split}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_panel_axes(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    font: ImageFont.ImageFont,
) -> None:
    plot_left = x0 + 70
    plot_top = y0 + 20
    plot_right = x1 - 20
    plot_bottom = y1 - 70
    for value in range(1, 6):
        y = y_value(value, plot_top, plot_bottom)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=3)
        draw.text((x0 + 18, y - 15), str(value), fill="#111827", font=font)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#cbd5e1", width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#cbd5e1", width=3)
    draw.text((plot_left + 10, plot_bottom + 26), "Window order within selected subject-day", fill="#111827", font=font)
    draw.text((x0 + 12, plot_top - 2), "fatigue", fill="#111827", font=font)


def y_value(value: float, plot_top: int, plot_bottom: int) -> int:
    clipped = max(0.5, min(5.5, float(value)))
    return int(plot_bottom - (clipped - 0.5) / 5.0 * (plot_bottom - plot_top))


def draw_series(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: str,
    *,
    width: int,
) -> None:
    plot_left = x0 + 70
    plot_top = y0 + 20
    plot_right = x1 - 20
    plot_bottom = y1 - 70
    if len(values) == 0:
        return
    if len(values) == 1:
        x = plot_left
        y = y_value(float(values[0]), plot_top, plot_bottom)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        return
    points = []
    for idx, value in enumerate(values):
        x = int(plot_left + idx / (len(values) - 1) * (plot_right - plot_left))
        y = y_value(float(value), plot_top, plot_bottom)
        points.append((x, y))
    draw.line(points, fill=color, width=width, joint="curve")


def main() -> None:
    args = parse_args()
    prediction_root = Path(args.prediction_root)
    phase1_root = Path(args.phase1_root)
    out_dir = Path(args.out_dir)

    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        for protocol, eeg, experiment, _role in ROUTES:
            baseline = load_npz(baseline_path(prediction_root, protocol, eeg, experiment))
            old = metric_pair(baseline, split)
            for calibration, _label, _color, _linestyle in CALIBRATIONS[1:]:
                calibrated = load_npz(calibration_path(phase1_root, protocol, eeg, experiment, calibration))
                new = metric_pair(calibrated, split)
                rows.append(
                    {
                        "split": split,
                        "protocol": protocol,
                        "experiment": experiment,
                        "calibration": calibration,
                        "old_raw_r": old["raw_r"],
                        "new_raw_r": new["raw_r"],
                        "delta_raw_r": new["raw_r"] - old["raw_r"],
                        "old_centered_r": old["centered_r"],
                        "new_centered_r": new["centered_r"],
                        "delta_centered_r": new["centered_r"] - old["centered_r"],
                        "old_rmse": old["rmse"],
                        "new_rmse": new["rmse"],
                        "delta_rmse": new["rmse"] - old["rmse"],
                        "old_std_ratio": old["std_ratio"],
                        "new_std_ratio": new["std_ratio"],
                        "delta_std_ratio": new["std_ratio"] - old["std_ratio"],
                    }
                )
    write_comparison(rows, out_dir)
    trend_path = plot_trends(prediction_root, phase1_root, out_dir, args.split)
    print(json.dumps({"trend_path": str(trend_path), "comparison_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
