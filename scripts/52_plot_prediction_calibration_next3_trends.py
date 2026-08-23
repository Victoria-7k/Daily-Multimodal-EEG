"""Plot Phase 4 next-three fatigue calibration screen trends and metric deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROUTES = (
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_no_audio"),
    ("within_subject_day", "eeg_eegpt_partial_ft_v1", "A2_Wdeep_full"),
)

VARIANT_LABELS = {
    "temporal_gru": "Temporal GRU",
    "subject_day_mean_residual": "Day mean+residual",
    "high_fatigue_day_oversample": "High fatigue oversample",
}

COLORS = {
    "truth": "#111827",
    "phase0": "#64748b",
    "temporal_gru": "#2563eb",
    "subject_day_mean_residual": "#dc2626",
    "high_fatigue_day_oversample": "#16a34a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(
            "outputs/server_sync/eeg_encoder_256d_5route_20260814/predictions/"
            "eeg_encoder_256d_5route_fusion_video_only_seed240800_raw"
        ),
    )
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=Path("outputs/server_sync/fatigue_calibration_20260816/phase4_next3_screen"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/server_sync/fatigue_calibration_20260816/phase4_next3_trend_visualizations"),
    )
    parser.add_argument("--split", default="test", choices=("val", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.screen_root / "metrics_val_test.json").read_text(encoding="utf-8"))
    records = build_records(args, metrics)
    image_path = args.out_dir / f"phase4_next3_route_trends_{args.split}.png"
    comparison_rows = build_metric_rows(metrics)
    draw_trends(image_path, records, split=args.split)
    write_csv(args.out_dir / "phase4_next3_metric_comparison.csv", comparison_rows)
    write_markdown(args.out_dir / "phase4_next3_metric_comparison.md", comparison_rows)
    print(json.dumps({"image_path": str(image_path), "route_count": len(records)}, indent=2))


def build_records(args: argparse.Namespace, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    runs = metrics["runs"]
    records = []
    for protocol, eeg, experiment in ROUTES:
        variants = {}
        for variant in VARIANT_LABELS:
            match = [
                run
                for run in runs
                if run["protocol"] == protocol
                and run["experiment"] == experiment
                and run["variant"] == variant
            ]
            if not match:
                raise ValueError(f"missing {variant} for {protocol}/{experiment}")
            variants[variant] = match[0]
        records.append(
            {
                "protocol": protocol,
                "eeg": eeg,
                "experiment": experiment,
                "baseline_path": args.baseline_root / protocol / eeg / experiment / "raw_lambda_0.npz",
                "variants": variants,
            }
        )
    return records


def build_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in metrics["rows"]:
        split = row["split"]
        prefix = f"{split}_"
        candidate = {
            "rmse": float(row[f"{prefix}rmse"]),
            "mae": float(row[f"{prefix}mae"]),
            "raw_r": float(row[f"{prefix}raw_r"]),
            "centered_r": float(row[f"{prefix}within_subject_centered_r"]),
            "std_ratio": float(row[f"{prefix}pred_std_over_true_std"]),
        }
        deltas = {
            "rmse": float(row[f"{prefix}rmse_delta"]),
            "mae": float(row[f"{prefix}mae_delta"]),
            "raw_r": float(row[f"{prefix}raw_r_delta"]),
            "centered_r": float(row[f"{prefix}centered_r_delta"]),
            "std_ratio": float(row[f"{prefix}std_ratio_delta"]),
        }
        baseline = {key: candidate[key] - deltas[key] for key in candidate}
        rows.append(
            {
                "split": split,
                "protocol": row["protocol"],
                "experiment": row["experiment"],
                "variant": row["variant"],
                "baseline_raw_r": baseline["raw_r"],
                "candidate_raw_r": candidate["raw_r"],
                "raw_r_delta": deltas["raw_r"],
                "baseline_centered_r": baseline["centered_r"],
                "candidate_centered_r": candidate["centered_r"],
                "centered_r_delta": deltas["centered_r"],
                "baseline_std_ratio": baseline["std_ratio"],
                "candidate_std_ratio": candidate["std_ratio"],
                "std_ratio_delta": deltas["std_ratio"],
                "rmse_delta": deltas["rmse"],
                "mae_delta": deltas["mae"],
                "high_abs_bias_reduction": float(row[f"{prefix}high_abs_bias_reduction"]),
            }
        )
    return rows


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(openable_path(Path(path)), allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def day_key(event_id: str) -> str:
    for part in str(event_id).split("_"):
        if part.startswith("day-"):
            return part
    return str(event_id)


def split_view(data: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    index = data[f"{split}_index"].astype(np.int64)
    return {
        "index": index,
        "prediction": data[f"{split}_prediction"].astype(np.float64),
        "target": data["target"][index].astype(np.float64),
        "sample_id": data["sample_id"][index].astype(str),
        "subject": data["subject_id"][index].astype(str),
        "event": data["event_id"][index].astype(str),
    }


def selected_subject_day(view: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    keys = np.array([f"{subject} {day_key(event)}" for subject, event in zip(view["subject"], view["event"])], dtype=object)
    best_key = ""
    best_mask = np.zeros(len(keys), dtype=bool)
    best_score = (-1.0, -1)
    for key in sorted(set(keys.tolist())):
        mask = keys == key
        if int(np.sum(mask)) < 2:
            continue
        score = (float(np.std(view["target"][mask])), int(np.sum(mask)))
        if score > best_score:
            best_key = key
            best_mask = mask
            best_score = score
    return best_key, best_mask


def aligned_prediction(data: dict[str, np.ndarray], split: str, sample_ids: np.ndarray) -> np.ndarray:
    view = split_view(data, split)
    by_id = {sample_id: pred for sample_id, pred in zip(view["sample_id"], view["prediction"])}
    return np.asarray([by_id[sample_id] for sample_id in sample_ids], dtype=np.float64)


def draw_trends(path: Path, records: list[dict[str, Any]], *, split: str) -> None:
    width, height = 3200, 1160
    margin_x, margin_top, margin_bottom = 120, 230, 210
    gap_x = 120
    panel_w = (width - margin_x * 2 - gap_x) // 2
    panel_h = height - margin_top - margin_bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = load_font(56)
    font_panel = load_font(32)
    font_text = load_font(24)
    font_small = load_font(21)
    font_tiny = load_font(18)

    draw.text((margin_x, 34), f"Phase 4 三条候选路线：真实 fatigue 与预测趋势（{split}）", fill="#111827", font=font_title)
    draw.text(
        (margin_x, 105),
        "比较 Phase 0 raw baseline、Temporal GRU、subject-day mean+residual 和 high-fatigue-day oversampling。",
        fill="#526071",
        font=font_text,
    )
    draw.text(
        (margin_x, 145),
        "每个面板选取该路线 split 内真实 fatigue 波动最明显的 subject-day。",
        fill="#526071",
        font=font_text,
    )

    for idx, record in enumerate(records):
        x0 = margin_x + idx * (panel_w + gap_x)
        y0 = margin_top
        draw_panel(draw, x0, y0, x0 + panel_w, y0 + panel_h, record, split=split, font_panel=font_panel, font_text=font_text, font_small=font_small, font_tiny=font_tiny)

    draw_legend(draw, margin_x, height - 145, font_text)
    draw.text(
        (margin_x, height - 70),
        "读图：可继续推进的候选应同时提高相关性/动态范围，并且不明显放大误差或丢失真实 fatigue 的阶跃变化。",
        fill="#526071",
        font=font_small,
    )
    img.save(path)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    record: dict[str, Any],
    *,
    split: str,
    font_panel: ImageFont.ImageFont,
    font_text: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
    font_tiny: ImageFont.ImageFont,
) -> None:
    baseline = load_npz(record["baseline_path"])
    base_view = split_view(baseline, split)
    key, mask = selected_subject_day(base_view)
    sample_ids = base_view["sample_id"][mask]
    series = {
        "truth": base_view["target"][mask],
        "phase0": base_view["prediction"][mask],
    }
    for variant, run in record["variants"].items():
        series[variant] = aligned_prediction(load_npz(run["prediction_path"]), split, sample_ids)

    plot_left = x0 + 84
    plot_top = y0 + 130
    plot_right = x1 - 35
    plot_bottom = y1 - 92
    draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom, font_small)
    draw.text((x0, y0 - 6), f"{record['protocol']} / {record['experiment']}", fill="#111827", font=font_panel)
    draw.text((x0, y0 + 38), f"{key} | {int(np.sum(mask))} windows", fill="#526071", font=font_text)
    draw.text((x0, y0 + 70), route_note(record), fill="#526071", font=font_tiny)

    for key_name in ("truth", "phase0", "temporal_gru", "subject_day_mean_residual", "high_fatigue_day_oversample"):
        draw_series(draw, series[key_name], plot_left, plot_top, plot_right, plot_bottom, COLORS[key_name], width=7 if key_name == "truth" else 4)


def route_note(record: dict[str, Any]) -> str:
    parts = []
    for variant, run in record["variants"].items():
        val = run["val"]
        parts.append(f"{VARIANT_LABELS[variant]} val raw={val['raw_r']:.3f}, centered={val['within_subject_centered_r']:.3f}")
    return "; ".join(parts)


def draw_axes(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int, font: ImageFont.ImageFont) -> None:
    for value in range(1, 6):
        y = y_value(value, top, bottom)
        draw.line((left, y, right, y), fill="#e5e7eb", width=3)
        draw.text((left - 48, y - 14), str(value), fill="#111827", font=font)
    draw.line((left, bottom, right, bottom), fill="#cbd5e1", width=3)
    draw.line((left, top, left, bottom), fill="#cbd5e1", width=3)
    draw.text((left + 10, bottom + 28), "subject-day 内 10s 窗口顺序", fill="#111827", font=font)
    draw.text((left - 64, top - 7), "fatigue", fill="#111827", font=font)


def y_value(value: float, top: int, bottom: int) -> int:
    clipped = max(0.5, min(5.5, float(value)))
    return int(bottom - (clipped - 0.5) / 5.0 * (bottom - top))


def draw_series(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: str,
    *,
    width: int,
) -> None:
    if len(values) == 0:
        return
    points = []
    for idx, value in enumerate(values):
        x = int(left + (idx / max(1, len(values) - 1)) * (right - left))
        y = y_value(float(value), top, bottom)
        points.append((x, y))
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    else:
        draw.line(points, fill=color, width=width, joint="curve")


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font: ImageFont.ImageFont) -> None:
    items = [
        ("真实 fatigue", COLORS["truth"]),
        ("Phase 0 raw", COLORS["phase0"]),
        ("Temporal GRU", COLORS["temporal_gru"]),
        ("Day mean+residual", COLORS["subject_day_mean_residual"]),
        ("High fatigue oversample", COLORS["high_fatigue_day_oversample"]),
    ]
    cursor = x
    for label, color in items:
        draw.line((cursor, y, cursor + 70, y), fill=color, width=7)
        draw.text((cursor + 86, y - 17), label, fill="#111827", font=font)
        cursor += max(330, int(draw.textlength(label, font=font)) + 150)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "protocol",
        "experiment",
        "variant",
        "baseline_raw_r",
        "candidate_raw_r",
        "raw_r_delta",
        "baseline_centered_r",
        "candidate_centered_r",
        "centered_r_delta",
        "baseline_std_ratio",
        "candidate_std_ratio",
        "std_ratio_delta",
        "rmse_delta",
        "mae_delta",
        "high_abs_bias_reduction",
    ]
    lines = [
        "# Phase 4 next-three metric comparison",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[field]) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "nan"
    return str(value)


if __name__ == "__main__":
    main()
