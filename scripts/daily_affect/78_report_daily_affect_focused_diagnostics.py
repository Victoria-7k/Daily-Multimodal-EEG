#!/usr/bin/env python3
"""Write a focused diagnostic report for one daily-affect candidate route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_RUN_ROOT = DEFAULT_ROOT / "outputs/daily_affect_dynamic_a1_crossday_routefix_20260905"
MODALITIES = ("eeg", "wear", "video", "audio")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "focused_reports")
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--normalization", default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--candidate-model", default="dynamic_kernel")
    parser.add_argument("--baseline-model", default="bag_static")
    parser.add_argument("--candidate-experiment-id", default="focused_ablation_v2")
    parser.add_argument("--baseline-experiment-id", default="focused_ablation_v2")
    args = parser.parse_args()

    pairs = load_metric_pairs(
        args.run_root,
        protocol=args.protocol,
        route_id=args.route_id,
        normalization=args.normalization,
        adapter_mode=args.adapter_mode,
        candidate_model=args.candidate_model,
        baseline_model=args.baseline_model,
        candidate_experiment_id=args.candidate_experiment_id,
        baseline_experiment_id=args.baseline_experiment_id,
    )
    if not pairs:
        raise SystemExit("no matched candidate/baseline metric pairs found")

    seed_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for pair in pairs:
        baseline = pair["baseline"]
        candidate = pair["candidate"]
        seed = int(candidate["seed"])
        seed_rows.append(seed_delta_row(seed, baseline, candidate))
        diag = load_npz(candidate.get("diagnostics_path"))
        pred = load_npz(candidate.get("prediction_path"))
        if diag is None or pred is None:
            continue
        test_idx = pred["test_index"].astype(np.int64)
        truth = pred["label_zero_based"][test_idx].astype(np.int64)
        predicted = pred["test_predicted_class"].astype(np.int64)
        abs_error = np.abs(predicted - truth).astype(np.float64)
        diagnostic_rows.append(diagnostic_summary(seed, diag, test_idx, abs_error))
        temporal_rows.extend(temporal_summary_rows(seed, diag, test_idx))

    output = {
        "script": Path(__file__).name,
        "run_root": str(args.run_root),
        "protocol": args.protocol,
        "route_id": args.route_id,
        "normalization": args.normalization,
        "adapter_mode": args.adapter_mode,
        "candidate_model": args.candidate_model,
        "baseline_model": args.baseline_model,
        "paired_seed_count": len(seed_rows),
        "seed_rows": seed_rows,
        "seed_delta_summary": summarize_seed_deltas(seed_rows),
        "diagnostics": diagnostic_rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    stem = f"{args.protocol}_{args.route_id}_{args.candidate_model}_{args.normalization}"
    out_json = args.out_root / f"{stem}_focused_diagnostics.json"
    out_md = args.out_root / f"{stem}_focused_diagnostics.md"
    out_seed_csv = args.out_root / f"{stem}_seed_deltas.csv"
    out_diag_csv = args.out_root / f"{stem}_diagnostics.csv"
    out_temporal_csv = args.out_root / f"{stem}_temporal_weights.csv"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_seed_csv, seed_rows)
    write_csv(out_diag_csv, diagnostic_rows)
    write_csv(out_temporal_csv, temporal_rows)
    write_markdown(output, out_md)
    print(f"paired_seed_count={len(seed_rows)}")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    return 0


def load_metric_pairs(
    run_root: Path,
    *,
    protocol: str,
    route_id: str,
    normalization: str,
    adapter_mode: str,
    candidate_model: str,
    baseline_model: str,
    candidate_experiment_id: str,
    baseline_experiment_id: str,
) -> list[dict[str, Any]]:
    baseline_by_seed: dict[int, dict[str, Any]] = {}
    candidate_by_seed: dict[int, dict[str, Any]] = {}
    for path in sorted(run_root.rglob("metrics.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            row.get("protocol") != protocol
            or row.get("route_id") != route_id
            or row.get("normalization") != normalization
            or row.get("adapter_mode") != adapter_mode
        ):
            continue
        row["metrics_path"] = str(path)
        model_id = row.get("model_id")
        seed = int(row.get("seed", -1))
        if model_id == baseline_model and row.get("experiment_id") == baseline_experiment_id:
            baseline_by_seed.setdefault(seed, row)
        elif model_id == candidate_model and row.get("experiment_id") == candidate_experiment_id:
            candidate_by_seed[seed] = row
    return [
        {"seed": seed, "baseline": baseline_by_seed[seed], "candidate": candidate_by_seed[seed]}
        for seed in sorted(set(baseline_by_seed) & set(candidate_by_seed))
    ]


def seed_delta_row(seed: int, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "baseline_qwk": metric(baseline, "qwk"),
        "candidate_qwk": metric(candidate, "qwk"),
        "delta_qwk": metric(candidate, "qwk") - metric(baseline, "qwk"),
        "baseline_macro_f1": metric(baseline, "macro_f1"),
        "candidate_macro_f1": metric(candidate, "macro_f1"),
        "delta_macro_f1": metric(candidate, "macro_f1") - metric(baseline, "macro_f1"),
        "baseline_ordinal_mae": metric(baseline, "ordinal_mae"),
        "candidate_ordinal_mae": metric(candidate, "ordinal_mae"),
        "delta_ordinal_mae": metric(candidate, "ordinal_mae") - metric(baseline, "ordinal_mae"),
        "candidate_best_epoch": candidate.get("train_audit", {}).get("best_epoch"),
        "baseline_metrics_path": baseline.get("metrics_path", ""),
        "candidate_metrics_path": candidate.get("metrics_path", ""),
    }


def diagnostic_summary(seed: int, diag: dict[str, np.ndarray], test_idx: np.ndarray, abs_error: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"seed": seed, "test_count": int(test_idx.size)}
    temporal = diag.get("temporal_weights")
    if temporal is not None:
        tw = temporal[test_idx].astype(np.float64)
        row["temporal_entropy"] = float(np.mean(entropy(tw)))
        row["recent_5_mass"] = float(np.mean(tw[:, -5:].sum(axis=1)))
        row["early_5_mass"] = float(np.mean(tw[:, :5].sum(axis=1)))
        row["peak_window_mean"] = float(np.mean(np.argmax(tw, axis=1)))
        row["corr_abs_error_recent_5_mass"] = pearson(abs_error, tw[:, -5:].sum(axis=1))
        row["corr_abs_error_peak_window"] = pearson(abs_error, np.argmax(tw, axis=1).astype(np.float64))
    mixture = diag.get("kernel_mixture")
    if mixture is not None:
        mix = mixture[test_idx].astype(np.float64)
        for idx, name in enumerate(("short", "medium", "long")):
            row[f"kernel_{name}_mean"] = float(np.mean(mix[:, idx]))
            row[f"kernel_{name}_std"] = float(np.std(mix[:, idx]))
            row[f"corr_abs_error_kernel_{name}"] = pearson(abs_error, mix[:, idx])
    weights = diag.get("modality_weights")
    if weights is not None:
        mw = weights[test_idx].astype(np.float64)
        for idx, modality in enumerate(MODALITIES):
            values = mw[:, :, idx].mean(axis=1)
            row[f"{modality}_weight_mean"] = float(np.mean(values))
            row[f"{modality}_weight_std"] = float(np.std(values))
            row[f"corr_abs_error_{modality}_weight"] = pearson(abs_error, values)
    difficulty = diag.get("modality_difficulty")
    if difficulty is not None:
        diff = difficulty[test_idx].astype(np.float64)
        for idx, modality in enumerate(MODALITIES):
            row[f"{modality}_difficulty_mean"] = float(np.mean(diff[:, idx]))
            row[f"{modality}_difficulty_std"] = float(np.std(diff[:, idx]))
            row[f"corr_abs_error_{modality}_difficulty"] = pearson(abs_error, diff[:, idx])
    return row


def temporal_summary_rows(seed: int, diag: dict[str, np.ndarray], test_idx: np.ndarray) -> list[dict[str, Any]]:
    temporal = diag.get("temporal_weights")
    if temporal is None:
        return []
    tw = temporal[test_idx].astype(np.float64)
    return [
        {
            "seed": seed,
            "event_window_id": window,
            "mean_weight": float(np.mean(tw[:, window])),
            "std_weight": float(np.std(tw[:, window])),
        }
        for window in range(tw.shape[1])
    ]


def summarize_seed_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, better in (("delta_qwk", 1), ("delta_macro_f1", 1), ("delta_ordinal_mae", -1)):
        values = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = float(mean(values)) if values else ""
        out[f"{key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        out[f"{key}_win_count"] = int(sum(value * better > 0 for value in values))
    return out


def write_markdown(output: dict[str, Any], path: Path) -> None:
    summary = output["seed_delta_summary"]
    lines = [
        "# Daily-Affect Focused Diagnostics",
        "",
        f"- protocol: `{output['protocol']}`",
        f"- route: `{output['route_id']}`",
        f"- candidate: `{output['candidate_model']} / {output['normalization']} / {output['adapter_mode']}`",
        f"- baseline: `{output['baseline_model']} / {output['normalization']} / {output['adapter_mode']}`",
        f"- paired seeds: `{output['paired_seed_count']}`",
        "",
        "## Paired Delta",
        "",
        f"- mean delta QWK: `{fmt(summary.get('delta_qwk_mean'))}`; QWK wins: `{summary.get('delta_qwk_win_count')}/{output['paired_seed_count']}`",
        f"- mean delta Macro-F1: `{fmt(summary.get('delta_macro_f1_mean'))}`; Macro-F1 wins: `{summary.get('delta_macro_f1_win_count')}/{output['paired_seed_count']}`",
        f"- mean delta ordinal MAE: `{fmt(summary.get('delta_ordinal_mae_mean'))}`; MAE improves: `{summary.get('delta_ordinal_mae_win_count')}/{output['paired_seed_count']}`",
        "",
        "| seed | baseline QWK | candidate QWK | delta QWK | delta Macro-F1 | delta ordinal MAE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["seed_rows"]:
        lines.append(
            f"| {row['seed']} | {fmt(row['baseline_qwk'])} | {fmt(row['candidate_qwk'])} | "
            f"{fmt(row['delta_qwk'])} | {fmt(row['delta_macro_f1'])} | {fmt(row['delta_ordinal_mae'])} |"
        )
    if output["diagnostics"]:
        lines.extend(["", "## Kernel And Modality Signals", ""])
        for row in output["diagnostics"]:
            lines.append(
                f"- seed `{row['seed']}`: recent-5 mass `{fmt(row.get('recent_5_mass'))}`, "
                f"kernel short/medium/long `{fmt(row.get('kernel_short_mean'))}`/`{fmt(row.get('kernel_medium_mean'))}`/`{fmt(row.get('kernel_long_mean'))}`, "
                f"modality weights EEG/Wear/Video/Audio `{fmt(row.get('eeg_weight_mean'))}`/`{fmt(row.get('wear_weight_mean'))}`/`{fmt(row.get('video_weight_mean'))}`/`{fmt(row.get('audio_weight_mean'))}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path_value: Any) -> dict[str, np.ndarray] | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def metric(row: dict[str, Any], name: str) -> float:
    return float(row.get("test", {}).get(name))


def entropy(weights: np.ndarray) -> np.ndarray:
    values = np.clip(weights, 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=1) / np.log(weights.shape[1])


def pearson(left: np.ndarray, right: np.ndarray) -> float | str:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return ""
    x = x[valid]
    y = y[valid]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return ""
    return float(np.corrcoef(x, y)[0, 1])


def fmt(value: Any) -> str:
    return "NA" if value == "" or value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
