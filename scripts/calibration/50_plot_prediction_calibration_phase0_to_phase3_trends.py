"""Plot Phase 0-3 fatigue prediction trends on selected test subject-days."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROUTES = (
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_full", "highest raw r"),
    ("cross_day", "eeg_eegpt_partial_ft_v1", "B0_Wphysio_no_audio", "lowest RMSE / stronger centered r"),
    ("date_in_order", "eeg_eegpt_partial_ft_v1", "B0_Wdeep_no_audio", "highest raw/centered r"),
    ("date_in_order", "eeg_eegpt_partial_ft_v1", "A2_Wdeep_full", "lowest RMSE"),
)

COLORS = {
    "truth": "#111827",
    "phase0": "#2563eb",
    "phase1": "#dc2626",
    "phase2": "#16a34a",
    "phase3": "#9333ea",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/server_sync/eeg_encoder_256d_5route_20260814/predictions/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw"),
    )
    parser.add_argument("--calibration-root", type=Path, default=Path("outputs/server_sync/fatigue_calibration_20260816"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/server_sync/fatigue_calibration_20260816/phase0_to_phase3_trend_visualizations"),
    )
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--phase3-seed", type=int, default=240800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(args)
    image_path = out_dir / f"phase0_to_phase3_route_trends_{args.split}.png"
    draw_trends(image_path, records, split=args.split)
    write_summary(out_dir / f"phase0_to_phase3_route_trends_{args.split}.csv", records)
    print(json.dumps({"image_path": str(image_path), "route_count": len(records)}, indent=2))


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    phase2_rows = list(csv.DictReader((args.calibration_root / "phase2_loss_sampler" / "metrics_val_test_with_deltas.csv").open(encoding="utf-8")))
    phase3_summary = json.loads((args.calibration_root / "phase3_ordinal_head" / "metrics_val.json").read_text(encoding="utf-8"))
    phase3_summaries = phase3_summary["summaries"]
    phase3_pairs = phase3_summary["paired_rows"]
    records = []
    for protocol, eeg, experiment, role in ROUTES:
        baseline_path = args.baseline_root / protocol / eeg / experiment / "raw_lambda_0.npz"
        phase1_path = args.calibration_root / "phase1_posthoc_calibration" / "predictions" / protocol / eeg / experiment / "variance_calibration.npz"
        phase2_choice = choose_phase2_row(phase2_rows, protocol, experiment)
        phase3_choice = choose_phase3_summary(phase3_summaries, protocol, experiment)
        phase3_pair = choose_phase3_pair(phase3_pairs, phase3_choice, args.phase3_seed)
        records.append(
            {
                "protocol": protocol,
                "eeg": eeg,
                "experiment": experiment,
                "role": role,
                "baseline_path": str(baseline_path),
                "phase1_path": str(phase1_path),
                "phase2_path": phase2_choice["prediction_path"],
                "phase2_label": phase2_choice["config"],
                "phase2_seed": phase2_choice["seed"],
                "phase2_std_delta": float(phase2_choice["std_ratio_delta"]),
                "phase3_path": phase3_pair["candidate_prediction_path"],
                "phase3_label": phase3_choice["candidate_id"],
                "phase3_seed": phase3_pair["seed"],
                "phase3_std_delta": float(phase3_choice["mean_val_std_ratio_delta"]),
                "phase3_corr_note": (
                    f"raw d={float(phase3_choice['mean_val_raw_r_delta']):+.3f}, "
                    f"centered d={float(phase3_choice['mean_val_centered_r_delta']):+.3f}"
                ),
            }
        )
    return records


def choose_phase2_row(rows: list[dict[str, str]], protocol: str, experiment: str) -> dict[str, str]:
    matches = [row for row in rows if row["protocol"] == protocol and row["experiment"] == experiment]
    if not matches:
        raise ValueError(f"missing Phase 2 row for {protocol}/{experiment}")
    return max(matches, key=lambda row: float(row["std_ratio_delta"]))


def choose_phase3_summary(rows: list[dict[str, Any]], protocol: str, experiment: str) -> dict[str, Any]:
    matches = [row for row in rows if row["protocol"] == protocol and row["experiment"] == experiment]
    if not matches:
        raise ValueError(f"missing Phase 3 summary for {protocol}/{experiment}")
    return max(matches, key=lambda row: float(row["mean_val_std_ratio_delta"]))


def choose_phase3_pair(rows: list[dict[str, Any]], summary: dict[str, Any], seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["candidate_id"] == summary["candidate_id"]
        and row["protocol"] == summary["protocol"]
        and row["experiment"] == summary["experiment"]
    ]
    if not matches:
        raise ValueError(f"missing Phase 3 pair for {summary['candidate_id']}")
    for row in matches:
        if int(row["seed"]) == int(seed):
            return row
    return matches[0]


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


def draw_trends(path: Path, records: list[dict[str, Any]], *, split: str) -> None:
    width, height = 3200, 1850
    margin_x, margin_top, margin_bottom = 120, 230, 230
    gap_x, gap_y = 95, 150
    panel_w = (width - margin_x * 2 - gap_x) // 2
    panel_h = (height - margin_top - margin_bottom - gap_y) // 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font_title = load_font(58)
    font_panel = load_font(31)
    font_text = load_font(24)
    font_small = load_font(21)
    font_tiny = load_font(18)

    draw.text((margin_x, 34), f"Fatigue Phase 0-3 trends on selected {split} subject-days", fill="#111827", font=font_title)
    draw.text(
        (margin_x, 105),
        "Each panel selects the subject-day with the clearest true fatigue variation for that route; Phase 2/3 lines use the best dynamic-range candidate.",
        fill="#526071",
        font=font_text,
    )
    draw.text(
        (margin_x, 145),
        "Phase 1 expands range but worsens error; Phase 2/3 remain close to the compressed raw predictions.",
        fill="#526071",
        font=font_text,
    )

    for idx, record in enumerate(records):
        row, col = divmod(idx, 2)
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_top + row * (panel_h + gap_y)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw_panel(draw, x0, y0, x1, y1, record, split=split, font_panel=font_panel, font_text=font_text, font_small=font_small, font_tiny=font_tiny)

    draw_legend(draw, margin_x, height - 150, font_text)
    draw.text(
        (margin_x, height - 70),
        "Reading: black step-like labels span 1-5; raw/Phase 2/Phase 3 predictions stay narrow, while Phase 1 variance calibration stretches predictions at the cost of larger errors.",
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
    phase1 = load_npz(record["phase1_path"])
    phase2 = load_npz(record["phase2_path"])
    phase3 = load_npz(record["phase3_path"])
    base_view = split_view(baseline, split)
    key, mask = selected_subject_day(base_view)
    series = {
        "truth": base_view["target"][mask],
        "phase0": base_view["prediction"][mask],
        "phase1": split_view(phase1, split)["prediction"][mask],
        "phase2": split_view(phase2, split)["prediction"][mask],
        "phase3": split_view(phase3, split)["prediction"][mask],
    }
    plot_left = x0 + 84
    plot_top = y0 + 82
    plot_right = x1 - 30
    plot_bottom = y1 - 90
    draw_axes(draw, plot_left, plot_top, plot_right, plot_bottom, font_small)
    draw.text((x0, y0 - 5), f"{record['protocol']} / {record['experiment']}", fill="#111827", font=font_panel)
    draw.text((x0, y0 + 36), f"{record['role']} | {key} | {int(np.sum(mask))} windows", fill="#526071", font=font_text)
    draw.text(
        (x0, y1 - 48),
        f"P2: {record['phase2_label']} (std d={record['phase2_std_delta']:+.3f}); "
        f"P3: {record['phase3_label']} seed {record['phase3_seed']} (std d={record['phase3_std_delta']:+.3f})",
        fill="#526071",
        font=font_tiny,
    )
    draw_series(draw, series["truth"], plot_left, plot_top, plot_right, plot_bottom, COLORS["truth"], width=7)
    draw_series(draw, series["phase0"], plot_left, plot_top, plot_right, plot_bottom, COLORS["phase0"], width=4)
    draw_series(draw, series["phase1"], plot_left, plot_top, plot_right, plot_bottom, COLORS["phase1"], width=4)
    draw_series(draw, series["phase2"], plot_left, plot_top, plot_right, plot_bottom, COLORS["phase2"], width=4)
    draw_series(draw, series["phase3"], plot_left, plot_top, plot_right, plot_bottom, COLORS["phase3"], width=4)


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
        ("Phase 1 variance", COLORS["phase1"]),
        ("Phase 2 best std", COLORS["phase2"]),
        ("Phase 3 best std", COLORS["phase3"]),
    ]
    cursor = x
    for label, color in items:
        draw.line((cursor, y, cursor + 76, y), fill=color, width=7)
        draw.text((cursor + 92, y - 17), label, fill="#111827", font=font)
        cursor += 430 if "variance" in label else 350


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


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    keys = [
        "protocol",
        "experiment",
        "phase2_label",
        "phase2_seed",
        "phase2_std_delta",
        "phase3_label",
        "phase3_seed",
        "phase3_std_delta",
        "baseline_path",
        "phase1_path",
        "phase2_path",
        "phase3_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in keys})


if __name__ == "__main__":
    main()
