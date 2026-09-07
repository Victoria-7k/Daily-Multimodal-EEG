"""Plot a compact Phase 0-3 fatigue calibration comparison."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/server_sync/fatigue_calibration_20260816")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase0_to_phase3_visual_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_stage_rows(args.root)
    write_csv(out_dir / "stage_metric_summary.csv", rows)
    write_markdown(out_dir / "phase0_to_phase3_metric_comparison.md", rows)
    plot_summary(out_dir / "phase0_to_phase3_metric_comparison.svg", rows)
    print(json.dumps({"out_dir": str(out_dir), "row_count": len(rows)}, indent=2))


def build_stage_rows(root: Path) -> list[dict[str, Any]]:
    phase1_val = json.loads((root / "phase1_posthoc_calibration" / "metrics_val.json").read_text(encoding="utf-8"))
    baseline_metrics = [route["baseline"] for route in phase1_val.values()]
    phase0 = mean_metric_row("Phase 0\nraw baseline", "raw baseline mean over 4 representative routes", baseline_metrics)
    phase0.update(
        {
            "stage_id": "phase0",
            "method": "raw baseline",
            "gate": "diagnostic: compressed dynamic range",
            "rmse_delta": 0.0,
            "mae_delta": 0.0,
            "raw_r_delta": 0.0,
            "centered_r_delta": 0.0,
            "std_ratio_delta": 0.0,
            "high_abs_bias_reduction": 0.0,
        }
    )

    variance_metrics = [route["calibrations"]["variance_calibration"]["metrics"] for route in phase1_val.values()]
    phase1 = mean_metric_row("Phase 1\nvariance calibration", "mean val effect over 4 representative routes", variance_metrics)
    phase1.update(mean_delta_row(variance_metrics, baseline_metrics))
    phase1.update({"stage_id": "phase1", "method": "post-hoc variance calibration", "gate": "failed: RMSE/MAE worsened"})

    phase2_summary = json.loads((root / "phase2_paired_gate" / "paired_gate_summary.json").read_text(encoding="utf-8"))
    phase2_choice = max(
        (row for row in phase2_summary["summaries"] if row.get("role") == "primary"),
        key=lambda row: float(row["mean_val_std_ratio_delta"]),
    )
    phase2_pairs = [
        row
        for row in phase2_summary["paired_rows"]
        if row["candidate_id"] == phase2_choice["candidate_id"]
    ]
    phase2 = paired_stage_row(
        "Phase 2\nloss/sampler",
        f"{phase2_choice['candidate_id']} ({phase2_choice['protocol']}/{phase2_choice['experiment']})",
        phase2_choice,
        phase2_pairs,
        "failed: 3-seed paired gate",
    )

    phase3_summary = json.loads((root / "phase3_ordinal_head" / "metrics_val.json").read_text(encoding="utf-8"))
    phase3_choice = max(phase3_summary["summaries"], key=lambda row: float(row["mean_val_std_ratio_delta"]))
    phase3_pairs = [
        row
        for row in phase3_summary["paired_rows"]
        if row["candidate_id"] == phase3_choice["candidate_id"]
        and row["protocol"] == phase3_choice["protocol"]
        and row["experiment"] == phase3_choice["experiment"]
    ]
    phase3 = paired_stage_row(
        "Phase 3\nordinal head",
        f"{phase3_choice['candidate_id']} ({phase3_choice['protocol']}/{phase3_choice['experiment']})",
        phase3_choice,
        phase3_pairs,
        "failed: no Phase 4 gate pass",
    )

    phase3_corr = max(
        phase3_summary["summaries"],
        key=lambda row: float(row["mean_val_raw_r_delta"]) + float(row["mean_val_centered_r_delta"]),
    )
    for row in (phase0, phase1, phase2, phase3):
        row["phase3_best_corr_note"] = (
            f"{phase3_corr['candidate_id']} / {phase3_corr['protocol']} / {phase3_corr['experiment']}: "
            f"raw r delta {phase3_corr['mean_val_raw_r_delta']:+.4f}, "
            f"centered r delta {phase3_corr['mean_val_centered_r_delta']:+.4f}, "
            f"std ratio delta {phase3_corr['mean_val_std_ratio_delta']:+.4f}"
        )
    return [phase0, phase1, phase2, phase3]


def mean_metric_row(stage: str, method: str, metrics: list[dict[str, float]]) -> dict[str, Any]:
    return {
        "stage": stage,
        "method": method,
        "rmse": mean(metrics, "rmse"),
        "mae": mean(metrics, "mae"),
        "raw_r": mean(metrics, "raw_r"),
        "centered_r": mean(metrics, "within_subject_centered_r"),
        "pred_std_over_true_std": mean(metrics, "pred_std_over_true_std"),
        "high_fatigue_bias": mean(metrics, "high_fatigue_bias"),
    }


def mean_delta_row(candidates: list[dict[str, float]], baselines: list[dict[str, float]]) -> dict[str, float]:
    deltas = []
    for cand, base in zip(candidates, baselines):
        base_high = abs(float(base["high_fatigue_bias"]))
        cand_high = abs(float(cand["high_fatigue_bias"]))
        deltas.append(
            {
                "rmse_delta": cand["rmse"] - base["rmse"],
                "mae_delta": cand["mae"] - base["mae"],
                "raw_r_delta": cand["raw_r"] - base["raw_r"],
                "centered_r_delta": cand["within_subject_centered_r"] - base["within_subject_centered_r"],
                "std_ratio_delta": cand["pred_std_over_true_std"] - base["pred_std_over_true_std"],
                "high_abs_bias_reduction": (base_high - cand_high) / base_high if base_high else float("nan"),
            }
        )
    return {key: mean(deltas, key) for key in deltas[0]}


def paired_stage_row(
    stage: str,
    method: str,
    summary: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    gate: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "stage_id": stage.split("\\n", 1)[0].lower().replace(" ", ""),
        "method": method,
        "gate": gate,
        "rmse": mean(pair_rows, "val_rmse"),
        "mae": mean(pair_rows, "val_mae"),
        "raw_r": mean(pair_rows, "val_raw_r"),
        "centered_r": mean(pair_rows, "val_within_subject_centered_r"),
        "pred_std_over_true_std": mean(pair_rows, "val_pred_std_over_true_std"),
        "high_fatigue_bias": mean(pair_rows, "val_high_fatigue_bias"),
        "rmse_delta": float(summary["mean_val_rmse_delta"]),
        "mae_delta": float(summary["mean_val_mae_delta"]),
        "raw_r_delta": float(summary["mean_val_raw_r_delta"]),
        "centered_r_delta": float(summary["mean_val_centered_r_delta"]),
        "std_ratio_delta": float(summary["mean_val_std_ratio_delta"]),
        "high_abs_bias_reduction": float(summary["mean_val_high_abs_bias_reduction"]),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(sum(values) / len(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "stage",
        "method",
        "gate",
        "pred_std_over_true_std",
        "high_fatigue_bias",
        "rmse_delta",
        "raw_r_delta",
        "centered_r_delta",
        "std_ratio_delta",
        "high_abs_bias_reduction",
    ]
    lines = [
        "# Phase 0-3 Fatigue Calibration Metric Comparison",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(key, "")) for key in keys) + " |")
    lines.extend(
        [
            "",
            "Phase 3 best correlation note: "
            + str(rows[-1].get("phase3_best_corr_note", "")),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "nan"
    return str(value).replace("\n", " ")


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["stage"] for row in rows]
    colors = ["#2f4b7c", "#f28e2b", "#59a14f", "#b07aa1"]
    width, height = 1400, 900
    panels = [
        (70, 110, 590, 300),
        (760, 110, 590, 300),
        (70, 500, 590, 300),
        (760, 500, 590, 300),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Fatigue calibration Phase 0 to Phase 3 metric comparison">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .title{font-size:26px;font-weight:700}.subtitle{font-size:14px;fill:#53606f}.panel-title{font-size:18px;font-weight:700}.axis{stroke:#5f6b7a;stroke-width:1}.grid{stroke:#d8dee8;stroke-width:1}.tick{font-size:12px;fill:#53606f}.label{font-size:13px}.value{font-size:12px;font-weight:700}.legend{font-size:12px;fill:#384252}",
        "</style>",
        '<rect x="0" y="0" width="1400" height="900" fill="#ffffff"/>',
        '<text class="title" x="70" y="46">Fatigue calibration Phase 0-3: dynamic range vs validation trade-offs</text>',
        '<text class="subtitle" x="70" y="74">Phase 1 uses variance calibration; Phase 2/3 use the best dynamic-range candidate from each paired gate.</text>',
    ]

    parts.extend(
        draw_single_bar_panel(
            panels[0],
            "Prediction dynamic range",
            "pred_std / true_std",
            labels,
            [row["pred_std_over_true_std"] for row in rows],
            colors,
            domain=(0.0, 1.08),
            reference_lines=[(1.0, "true std parity", "#555555", "6,4")],
        )
    )
    parts.extend(
        draw_single_bar_panel(
            panels[1],
            "High-fatigue bias",
            "mean(pred - true), high labels",
            labels,
            [row["high_fatigue_bias"] for row in rows],
            colors,
            domain=(-2.2, 0.35),
            reference_lines=[(0.0, "no bias", "#555555", "")],
        )
    )
    parts.extend(
        draw_grouped_bar_panel(
            panels[2],
            "Error and correlation delta",
            "validation delta vs paired baseline",
            labels,
            [
                ("RMSE", [row["rmse_delta"] for row in rows], "#d95f02"),
                ("MAE", [row["mae_delta"] for row in rows], "#e6ab02"),
                ("raw r", [row["raw_r_delta"] for row in rows], "#1b9e77"),
                ("centered r", [row["centered_r_delta"] for row in rows], "#7570b3"),
            ],
            domain=(-0.12, 0.12),
            reference_lines=[(0.0, "no change", "#555555", ""), (0.015, "RMSE gate +0.015", "#d95f02", "3,3")],
        )
    )
    parts.extend(
        draw_grouped_bar_panel(
            panels[3],
            "Dynamic-range target deltas",
            "validation delta / reduction",
            labels,
            [
                ("std ratio delta", [row["std_ratio_delta"] for row in rows], "#1f77b4"),
                ("high bias reduction", [row["high_abs_bias_reduction"] for row in rows], "#2ca02c"),
            ],
            domain=(-0.25, 0.75),
            reference_lines=[(0.0, "no change", "#555555", ""), (0.10, "std gate +0.10", "#1f77b4", "3,3"), (0.20, "bias gate +0.20", "#2ca02c", "3,3")],
        )
    )
    parts.append(
        '<text class="subtitle" x="70" y="862">Reading: Phase 1 can expand dynamic range, but with large error cost; Phase 2/3 do not reach the +0.10 std-ratio gate or stable high-fatigue correction.</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_single_bar_panel(
    panel: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    *,
    domain: tuple[float, float],
    reference_lines: list[tuple[float, str, str, str]],
) -> list[str]:
    x, y, w, h = panel
    margin = {"left": 70, "right": 20, "top": 62, "bottom": 72}
    cx, cy = x + margin["left"], y + margin["top"]
    cw, ch = w - margin["left"] - margin["right"], h - margin["top"] - margin["bottom"]
    y0, y1 = domain
    parts = panel_base(x, y, w, h, title, ylabel)
    parts.extend(draw_y_axis(cx, cy, cw, ch, y0, y1))
    for value, text, color, dash in reference_lines:
        py = scale_y(value, y0, y1, cy, ch)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{cx}" y1="{py:.1f}" x2="{cx+cw}" y2="{py:.1f}" stroke="{color}" stroke-width="1"{dash_attr}/>')
        parts.append(f'<text class="legend" x="{cx+cw-4}" y="{py-5:.1f}" text-anchor="end">{esc(text)}</text>')
    gap = 26
    bw = (cw - gap * (len(values) + 1)) / len(values)
    zero = scale_y(0.0, y0, y1, cy, ch)
    for idx, value in enumerate(values):
        bx = cx + gap + idx * (bw + gap)
        by = scale_y(max(value, 0.0), y0, y1, cy, ch)
        by2 = scale_y(min(value, 0.0), y0, y1, cy, ch)
        rect_y = by if value >= 0 else zero
        rect_h = abs((zero if value >= 0 else by2) - by)
        if value < 0:
            rect_y = zero
            rect_h = by2 - zero
        parts.append(f'<rect x="{bx:.1f}" y="{rect_y:.1f}" width="{bw:.1f}" height="{rect_h:.1f}" fill="{colors[idx]}"/>')
        ty = rect_y - 7 if value >= 0 else rect_y + rect_h + 15
        parts.append(f'<text class="value" x="{bx+bw/2:.1f}" y="{ty:.1f}" text-anchor="middle">{value:.2f}</text>')
        parts.extend(multiline_text(labels[idx], bx + bw / 2, y + h - 44, anchor="middle", size_class="tick"))
    return parts


def draw_grouped_bar_panel(
    panel: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    domain: tuple[float, float],
    reference_lines: list[tuple[float, str, str, str]],
) -> list[str]:
    x, y, w, h = panel
    margin = {"left": 70, "right": 20, "top": 72, "bottom": 72}
    cx, cy = x + margin["left"], y + margin["top"]
    cw, ch = w - margin["left"] - margin["right"], h - margin["top"] - margin["bottom"]
    y0, y1 = domain
    parts = panel_base(x, y, w, h, title, ylabel)
    parts.extend(draw_y_axis(cx, cy, cw, ch, y0, y1))
    for value, text, color, dash in reference_lines:
        py = scale_y(value, y0, y1, cy, ch)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{cx}" y1="{py:.1f}" x2="{cx+cw}" y2="{py:.1f}" stroke="{color}" stroke-width="1"{dash_attr}/>')
        parts.append(f'<text class="legend" x="{cx+cw-4}" y="{py-5:.1f}" text-anchor="end">{esc(text)}</text>')
    legend_x = x + 230
    for idx, (name, _, color) in enumerate(series):
        lx = legend_x + idx * 112
        parts.append(f'<rect x="{lx}" y="{y+15}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{lx+18}" y="{y+25}">{esc(name)}</text>')
    group_gap = 28
    group_w = (cw - group_gap * (len(labels) + 1)) / len(labels)
    bw = min(22, group_w / (len(series) + 0.8))
    zero = scale_y(0.0, y0, y1, cy, ch)
    for group_idx, label in enumerate(labels):
        gx = cx + group_gap + group_idx * (group_w + group_gap)
        for sidx, (_, vals, color) in enumerate(series):
            value = vals[group_idx]
            bx = gx + (group_w - bw * len(series)) / 2 + sidx * bw
            vy = scale_y(value, y0, y1, cy, ch)
            rect_y = min(zero, vy)
            rect_h = abs(zero - vy)
            parts.append(f'<rect x="{bx:.1f}" y="{rect_y:.1f}" width="{bw-2:.1f}" height="{rect_h:.1f}" fill="{color}"/>')
        parts.extend(multiline_text(label, gx + group_w / 2, y + h - 44, anchor="middle", size_class="tick"))
    return parts


def panel_base(x: int, y: int, w: int, h: int, title: str, ylabel: str) -> list[str]:
    return [
        f'<text class="panel-title" x="{x}" y="{y+22}">{esc(title)}</text>',
        f'<text class="subtitle" x="{x}" y="{y+44}">{esc(ylabel)}</text>',
    ]


def draw_y_axis(cx: float, cy: float, cw: float, ch: float, y0: float, y1: float) -> list[str]:
    parts = [
        f'<line class="axis" x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy+ch}"/>',
        f'<line class="axis" x1="{cx}" y1="{cy+ch}" x2="{cx+cw}" y2="{cy+ch}"/>',
    ]
    for idx in range(5):
        value = y0 + (y1 - y0) * idx / 4
        py = scale_y(value, y0, y1, cy, ch)
        parts.append(f'<line class="grid" x1="{cx}" y1="{py:.1f}" x2="{cx+cw}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tick" x="{cx-8}" y="{py+4:.1f}" text-anchor="end">{value:.2f}</text>')
    return parts


def scale_y(value: float, y0: float, y1: float, cy: float, ch: float) -> float:
    return cy + ch - (value - y0) / (y1 - y0) * ch


def multiline_text(text: str, x: float, y: float, *, anchor: str, size_class: str) -> list[str]:
    lines = str(text).split("\n")
    parts = [f'<text class="{size_class}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">']
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else 14
        parts.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    parts.append("</text>")
    return parts


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
