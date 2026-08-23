#!/usr/bin/env python3
"""Pair full-matrix fusion-variant runs against the archived attention matrix.

The archived 0814 video-only matrix (`eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json`)
was produced by the same script with the default seed sequence (240729 + run_number).  A full-matrix
run of another variant with the identical --experiments order therefore shares the same (protocol,
experiment, eeg_branch, seed) keys, so every row can be paired with its archived attention row.

Outputs per-protocol and per-EEG-branch mean deltas plus win/loss counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-json", type=Path, required=True, help="Archived attention matrix JSON.")
    parser.add_argument("--variant-jsons", type=str, required=True, help="Comma-separated variant matrix JSONs.")
    parser.add_argument("--out-md", type=Path, default=Path("outputs/reports/fusion_variant_full_paired_summary.md"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/fusion_variant_full_paired_summary.json"))
    args = parser.parse_args()

    archive = _load(args.archive_json)
    archive_index: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in archive.get("results", []):
        archive_index[(row["protocol"], row["experiment"], row.get("eeg_branch", "eeg"), int(row["seed"]))] = row

    variant_paths = [Path(v.strip()) for v in args.variant_jsons.split(",") if v.strip()]
    sections: list[str] = []
    all_rows: list[dict[str, Any]] = []
    for variant_path in variant_paths:
        report = _load(variant_path)
        variant_name = report.get("runtime", {}).get("fusion_variant", variant_path.stem)
        paired = 0
        unpaired = 0
        deltas: list[dict[str, Any]] = []
        for row in report.get("results", []):
            key = (row["protocol"], row["experiment"], row.get("eeg_branch", "eeg"), int(row["seed"]))
            ref = archive_index.get(key)
            if ref is None:
                unpaired += 1
                continue
            paired += 1
            t, rt = row["test"], ref["test"]
            deltas.append(
                {
                    "protocol": row["protocol"],
                    "experiment": row["experiment"],
                    "eeg_branch": row.get("eeg_branch", "eeg"),
                    "seed": row["seed"],
                    "delta_raw_r": t["raw_r"] - rt["raw_r"],
                    "delta_centered_r": t["within_subject_centered_r"] - rt["within_subject_centered_r"],
                    "delta_rmse": t["rmse"] - rt["rmse"],
                    "raw_r": t["raw_r"],
                    "ref_raw_r": rt["raw_r"],
                }
            )
        all_rows.extend({"variant": variant_name, **d} for d in deltas)
        lines = [
            f"## {variant_name} vs archived attention（paired {paired} / unpaired {unpaired}）",
            "",
            "| protocol | N | Δ raw r 均值 | Δ centered r 均值 | Δ RMSE 均值 | raw r 胜/负/平 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for protocol in ("cross_subject", "cross_day", "within_subject_day"):
            group = [d for d in deltas if d["protocol"] == protocol]
            if not group:
                continue
            wins = sum(1 for d in group if d["delta_raw_r"] > 0.005)
            losses = sum(1 for d in group if d["delta_raw_r"] < -0.005)
            lines.append(
                f"| {protocol} | {len(group)} | {_fmt(_mean([d['delta_raw_r'] for d in group]))} | "
                f"{_fmt(_mean([d['delta_centered_r'] for d in group]))} | {_fmt(_mean([d['delta_rmse'] for d in group]))} | "
                f"{wins}/{losses}/{len(group) - wins - losses} |"
            )
        lines.extend(
            [
                "",
                "| EEG branch | N | Δ raw r 均值 | Δ centered r 均值 | Δ RMSE 均值 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for branch in sorted({d["eeg_branch"] for d in deltas}):
            group = [d for d in deltas if d["eeg_branch"] == branch]
            lines.append(
                f"| {branch} | {len(group)} | {_fmt(_mean([d['delta_raw_r'] for d in group]))} | "
                f"{_fmt(_mean([d['delta_centered_r'] for d in group]))} | {_fmt(_mean([d['delta_rmse'] for d in group]))} |"
            )
        sections.extend(lines)

    header = ["# Fusion Variant 全量配对汇总（vs 归档 attention，同 seed 逐 run 配对）", ""]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(header + sections) + "\n", encoding="utf-8")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
