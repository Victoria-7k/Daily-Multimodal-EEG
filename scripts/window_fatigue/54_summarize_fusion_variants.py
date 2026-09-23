#!/usr/bin/env python3
"""Summarize the fusion-variant decision slice (attention vs concat vs pma vs eeg_anchor).

Reads the four variant report JSONs produced by scripts/window_fatigue/32_run_eegpt_centered_loss.py
(--experiment-seed-fixed --seed 240800, 8 runs each) and prints per-protocol mean
test metrics plus an optional reference column from an archived run (e.g. the 0814
video-only matrix, which used different seeds and is directional reference only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_VARIANTS = ("attention", "concat", "attention_multihead_pma", "eeg_anchor")


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _metric_rows(results: list[dict[str, Any]], archive: dict[str, Any] | None) -> list[dict[str, Any]]:
    archive_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if archive is not None:
        for row in archive.get("results", []):
            archive_index[(row["protocol"], row["experiment"], row.get("eeg_branch", "eeg"))] = row
    rows: list[dict[str, Any]] = []
    for row in sorted(
        results,
        key=lambda r: (r["protocol"], r["experiment"], r.get("eeg_branch", "eeg"), r.get("fusion_variant", "attention")),
    ):
        test = row["test"]
        ref = archive_index.get((row["protocol"], row["experiment"], row.get("eeg_branch", "eeg")))
        rows.append(
            {
                "protocol": row["protocol"],
                "experiment": row["experiment"],
                "eeg_branch": row.get("eeg_branch", "eeg"),
                "fusion_variant": row.get("fusion_variant", "attention"),
                "seed": row["seed"],
                "rmse": test["rmse"],
                "mae": test["mae"],
                "raw_r": test["raw_r"],
                "centered_r": test["within_subject_centered_r"],
                "reference_raw_r": ref["test"]["raw_r"] if ref else None,
                "reference_rmse": ref["test"]["rmse"] if ref else None,
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]], variants: tuple[str, ...]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    protocols = sorted({r["protocol"] for r in rows})
    for protocol in protocols:
        for variant in variants:
            group = [r for r in rows if r["protocol"] == protocol and r["fusion_variant"] == variant]
            if not group:
                continue
            summary.append(
                {
                    "protocol": protocol,
                    "fusion_variant": variant,
                    "mean_rmse": _mean([r["rmse"] for r in group]),
                    "mean_mae": _mean([r["mae"] for r in group]),
                    "mean_raw_r": _mean([r["raw_r"] for r in group]),
                    "mean_centered_r": _mean([r["centered_r"] for r in group]),
                    "run_count": len(group),
                }
            )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/reports/fusion_variant_decision_seed240800"))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--archive-json", type=Path, default=None, help="Optional archived matrix JSON for reference columns.")
    parser.add_argument("--out-md", type=Path, default=Path("outputs/reports/fusion_variant_decision_seed240800/summary.md"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/fusion_variant_decision_seed240800/summary.json"))
    args = parser.parse_args()

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    archive = _load_report(args.archive_json) if args.archive_json else None
    all_rows: list[dict[str, Any]] = []
    for variant in variants:
        report_path = args.reports_root / f"{variant}.json"
        if not report_path.exists():
            print(f"missing report: {report_path}")
            continue
        report = _load_report(report_path)
        print(f"loaded {report_path} run_count={report.get('run_count')}")
        all_rows.extend(_metric_rows(report["results"], archive))

    rows = all_rows
    summary = _summarize(rows, variants)
    summary.sort(key=lambda r: (r["protocol"], -1 if r["mean_raw_r"] is None else -r["mean_raw_r"]))

    lines = [
        "# Fusion Variant 决策切片汇总",
        "",
        f"- variants: `{'`, `'.join(variants)}`",
        f"- 每变体 runs: 8（cross_day/date_in_order × B0_Wphysio_full/B0_Wphysio_no_audio/A1_Wdeep_full/A1_Wdeep_no_audio，`eeg_eegpt_partial_ft_v1`，seed 240800 固定）",
        "",
        "## 按协议 × 变体的 test 均值",
        "",
        "| protocol | fusion variant | mean RMSE | mean MAE | mean raw r | mean centered r | runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary:
        lines.append(
            f"| {s['protocol']} | {s['fusion_variant']} | {_fmt(s['mean_rmse'])} | {_fmt(s['mean_mae'])} | "
            f"{_fmt(s['mean_raw_r'])} | {_fmt(s['mean_centered_r'])} | {s['run_count']} |"
        )
    lines.extend(["", "## 逐 run 明细", "", "| protocol | experiment | fusion variant | seed | RMSE | MAE | raw r | centered r | archive raw r | archive RMSE |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for r in sorted(rows, key=lambda x: (x["protocol"], x["experiment"], x["fusion_variant"])):
        lines.append(
            f"| {r['protocol']} | {r['experiment']} | {r['fusion_variant']} | {r['seed']} | {_fmt(r['rmse'])} | {_fmt(r['mae'])} | "
            f"{_fmt(r['raw_r'])} | {_fmt(r['centered_r'])} | {_fmt(r['reference_raw_r'])} | {_fmt(r['reference_rmse'])} |"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = {
        "summary": summary,
        "rows": rows,
        "decision_threshold": {"raw_r_meaningful": 0.01, "raw_r_noise": 0.005},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
