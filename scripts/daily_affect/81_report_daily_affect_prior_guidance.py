#!/usr/bin/env python3
"""Report paired prior-guidance ablations for the daily-affect route."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_RUN_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_routefix_20260905"
METRICS = (
    "qwk",
    "macro_f1",
    "ordinal_mae",
    "expected_rmse",
    "expected_raw_r",
    "expected_within_subject_centered_r",
)
HIGHER_IS_BETTER = {"qwk", "macro_f1", "expected_raw_r", "expected_within_subject_centered_r"}
COMPARISONS = (
    ("state_prior_only", "state_uniform", "prior_uniform"),
    ("ordinal_difficulty_on_state_prior", "prior_uniform", "prior_ordD_uniform"),
    ("state_prior_on_dynamic_kernel", "dynamic_kernel_no_prior", "dynamic_kernel_prior_uniform"),
    ("ordinal_difficulty_on_dynamic_prior", "dynamic_kernel_prior_uniform", "dynamic_kernel"),
    ("complete_prior_path_on_dynamic_kernel", "dynamic_kernel_no_prior", "dynamic_kernel"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "reports")
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--normalization", default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--experiment-id", default="state_matrix_native")
    args = parser.parse_args()

    rows = load_rows(
        args.run_root,
        protocol=args.protocol,
        route_id=args.route_id,
        normalization=args.normalization,
        adapter_mode=args.adapter_mode,
        experiment_id=args.experiment_id,
    )
    paired = paired_comparisons(rows)
    summary = summarize_pairs(paired)
    output = {
        "script": Path(__file__).name,
        "run_root": str(args.run_root),
        "protocol": args.protocol,
        "route_id": args.route_id,
        "normalization": args.normalization,
        "adapter_mode": args.adapter_mode,
        "experiment_id": args.experiment_id,
        "run_count": len(rows),
        "paired_deltas": paired,
        "comparison_summary": summary,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    json_path = args.out_root / "prior_guidance_ablation_report.json"
    md_path = args.out_root / "prior_guidance_ablation_report.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_root / "prior_guidance_paired_deltas.csv", paired)
    write_csv(args.out_root / "prior_guidance_summary.csv", summary)
    write_markdown(output, md_path)
    print(f"run_count={len(rows)}")
    print(f"out_json={json_path}")
    print(f"out_md={md_path}")
    return 0


def load_rows(
    root: Path,
    *,
    protocol: str,
    route_id: str,
    normalization: str,
    adapter_mode: str,
    experiment_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            row.get("protocol") != protocol
            or row.get("route_id") != route_id
            or row.get("normalization") != normalization
            or row.get("adapter_mode") != adapter_mode
            or row.get("experiment_id") != experiment_id
        ):
            continue
        row["metrics_path"] = str(path)
        rows.append(row)
    return rows


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row.get("model_id"), int(row.get("seed", -1))): row for row in rows}
    output: list[dict[str, Any]] = []
    for comparison_id, base_model, candidate_model in COMPARISONS:
        seeds = sorted(seed for (model, seed) in indexed if model == base_model and (candidate_model, seed) in indexed)
        for seed in seeds:
            base = indexed[(base_model, seed)]
            candidate = indexed[(candidate_model, seed)]
            row: dict[str, Any] = {
                "comparison_id": comparison_id,
                "seed": seed,
                "baseline_model_id": base_model,
                "candidate_model_id": candidate_model,
                "baseline_metrics_path": base["metrics_path"],
                "candidate_metrics_path": candidate["metrics_path"],
            }
            for metric in METRICS:
                left = base.get("test", {}).get(metric)
                right = candidate.get("test", {}).get(metric)
                row[f"delta_{metric}"] = "" if left is None or right is None else float(right) - float(left)
            output.append(row)
    return output


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["comparison_id"]].append(row)
    summary: list[dict[str, Any]] = []
    for comparison_id, values in sorted(grouped.items()):
        first = values[0]
        item: dict[str, Any] = {
            "comparison_id": comparison_id,
            "baseline_model_id": first["baseline_model_id"],
            "candidate_model_id": first["candidate_model_id"],
            "paired_seed_count": len(values),
        }
        for metric in METRICS:
            deltas = [float(row[f"delta_{metric}"]) for row in values if row[f"delta_{metric}"] != ""]
            item[f"mean_delta_{metric}"] = "" if not deltas else float(mean(deltas))
            if metric in HIGHER_IS_BETTER:
                item[f"win_count_{metric}"] = sum(value > 0.0 for value in deltas)
            else:
                item[f"win_count_{metric}"] = sum(value < 0.0 for value in deltas)
        qwk = item["mean_delta_qwk"]
        mae = item["mean_delta_ordinal_mae"]
        qwk_wins = item["win_count_qwk"]
        item["preliminary_qwk_mae_signal"] = (
            "pass" if qwk != "" and qwk > 0.0 and qwk_wins >= max(1, len(values) // 2 + 1) and (mae == "" or mae <= 0.0) else "no_signal"
        )
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Daily-Affect Prior-Guidance Ablation",
        "",
        f"- protocol: `{output['protocol']}`",
        f"- route: `{output['route_id']}`",
        f"- normalization: `{output['normalization']}`",
        f"- adapter mode: `{output['adapter_mode']}`",
        f"- experiment: `{output['experiment_id']}`",
        f"- run_count: `{output['run_count']}`",
        "",
        "Candidate minus baseline; QWK/Macro-F1/raw r/centered r favor positive values, while ordinal MAE and Expected RMSE favor negative values.",
        "",
        "| comparison | baseline -> candidate | seeds | delta QWK | QWK wins | delta ordinal MAE | delta Expected RMSE | delta raw r | delta centered r | signal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in output["comparison_summary"]:
        lines.append(
            f"| {row['comparison_id']} | {row['baseline_model_id']} -> {row['candidate_model_id']} | {row['paired_seed_count']} | "
            f"{fmt(row['mean_delta_qwk'])} | {row['win_count_qwk']} | {fmt(row['mean_delta_ordinal_mae'])} | "
            f"{fmt(row['mean_delta_expected_rmse'])} | {fmt(row['mean_delta_expected_raw_r'])} | "
            f"{fmt(row['mean_delta_expected_within_subject_centered_r'])} | {row['preliminary_qwk_mae_signal']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value == "" or value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
