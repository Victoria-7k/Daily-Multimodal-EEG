#!/usr/bin/env python3
"""Audit daily-affect label timing, cross-day token drift, and video stability."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from daily_multimodal.daily_affect.training import DailyAffectBagDataset, load_bag_dataset, run_daily_affect_run


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_BAGS_ROOT = DEFAULT_ROOT / "outputs/daily_affect_dynamic_a1_crossday_20260903/bags"
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "outputs/daily_affect_bottleneck_audit_routefix_20260905"
MODALITIES = ("eeg", "wear", "video", "audio")
TEMPORAL_POLICIES = (
    ("full_2min", "uniform"),
    ("last_10s", "last_10s"),
    ("last_30s", "last_30s"),
    ("last_60s", "last_60s"),
    ("first_30s", "first_30s"),
    ("kernel_short", "kernel_short"),
    ("kernel_medium", "kernel_medium"),
    ("kernel_long", "kernel_long"),
)
METRIC_KEYS = (
    "qwk",
    "macro_f1",
    "ordinal_mae",
    "expected_rmse",
    "expected_raw_r",
    "expected_within_subject_centered_r",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--normalization", default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--seeds", default="240729,240730,240731")
    parser.add_argument("--video-source", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--selection-metric", choices=("qwk", "macro_f1", "accuracy", "ordinal_mae", "nll"), default="qwk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    seeds = [int(value) for value in split_csv(args.seeds)]
    args.out_root.mkdir(parents=True, exist_ok=True)

    temporal_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    video_quality_rows: list[dict[str, Any]] = []
    video_source_path: str | None = None
    for seed in seeds:
        dataset = load_bag_dataset(args.bags_root / args.protocol / args.route_id / f"seed_{seed}" / "ema_bags.npz")
        video_source = args.video_source or resolve_video_source(dataset)
        video_source_path = str(video_source) if video_source is not None else video_source_path
        drift_rows.extend(cross_day_drift_rows(dataset, seed=seed))
        if video_source is not None and video_source.is_file():
            video_quality_rows.extend(video_quality_audit_rows(dataset, video_source, seed=seed))
        else:
            video_quality_rows.extend(video_quality_unavailable_rows(dataset, seed=seed))
        temporal_rows.extend(run_temporal_screen(dataset, args=args, seed=seed))

    temporal_summary = summarize_temporal_screen(temporal_rows)
    temporal_deltas = paired_temporal_deltas(temporal_rows)
    drift_summary = summarize_drift(drift_rows)
    paths = {
        "temporal_screen": args.out_root / "temporal_screen.csv",
        "temporal_paired_deltas": args.out_root / "temporal_paired_deltas.csv",
        "modality_drift_by_day": args.out_root / "modality_drift_by_day.csv",
        "modality_drift_summary": args.out_root / "modality_drift_summary.csv",
        "video_quality_by_day": args.out_root / "video_quality_by_day.csv",
        "temporal_plot": args.out_root / "temporal_policy_metrics.png",
        "drift_plot": args.out_root / "cross_day_modality_drift.png",
        "report_json": args.out_root / "daily_affect_bottleneck_audit.json",
        "report_md": args.out_root / "daily_affect_bottleneck_audit.md",
    }
    write_csv(paths["temporal_screen"], temporal_rows)
    write_csv(paths["temporal_paired_deltas"], temporal_deltas)
    write_csv(paths["modality_drift_by_day"], drift_rows)
    write_csv(paths["modality_drift_summary"], drift_summary)
    write_csv(paths["video_quality_by_day"], video_quality_rows)
    write_plots(temporal_summary, drift_summary, paths)
    payload = {
        "script": Path(__file__).name,
        "protocol": args.protocol,
        "route_id": args.route_id,
        "normalization": args.normalization,
        "adapter_mode": args.adapter_mode,
        "seeds": seeds,
        "video_source": video_source_path,
        "interpretation_boundary": "Temporal screen is paired supervised evidence. Drift and quality rows are input-only diagnostics and do not establish causality.",
        "temporal_screen": temporal_rows,
        "temporal_summary": temporal_summary,
        "temporal_paired_deltas": temporal_deltas,
        "modality_drift_by_day": drift_rows,
        "modality_drift_summary": drift_summary,
        "video_quality_by_day": video_quality_rows,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    paths["report_json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, paths["report_md"])
    print(f"temporal_run_count={len(temporal_rows)}")
    print(f"drift_row_count={len(drift_rows)}")
    print(f"video_quality_row_count={len(video_quality_rows)}")
    print(f"report={paths['report_md']}")
    return 0


def run_temporal_screen(dataset: DailyAffectBagDataset, *, args: argparse.Namespace, seed: int) -> list[dict[str, Any]]:
    screens: list[tuple[str, str, DailyAffectBagDataset]] = [
        (analysis_id, policy, dataset) for analysis_id, policy in TEMPORAL_POLICIES
    ]
    screens.extend(
        [
            ("video_only_full_2min", "uniform", subset_modalities(dataset, (2,))),
            ("nonvideo_full_2min", "uniform", subset_modalities(dataset, (0, 1, 3))),
        ]
    )
    rows: list[dict[str, Any]] = []
    for analysis_id, policy, screened_dataset in screens:
        run_dir = args.out_root / "temporal_screen" / analysis_id / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        if args.skip_existing and metrics_path.is_file():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            result = run_daily_affect_run(
                dataset=screened_dataset,
                protocol=args.protocol,
                model_id="bag_static",
                normalization=args.normalization,
                adapter_mode=args.adapter_mode,
                seed=seed,
                run_dir=run_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                hidden_dim=args.hidden_dim,
                patience=args.patience,
                lambda_d=0.0,
                probe_loss_weight=0.0,
                selection_metric=args.selection_metric,
                calibrate_probe_temperature=False,
                temporal_policy=policy,
                device=args.device,
                include_diagnostics=False,
                experiment_id="bottleneck_audit_v2",
            )
        test = result["test"]
        row = {
            "seed": int(seed),
            "analysis_id": analysis_id,
            "temporal_policy": policy,
            "modality_view": modality_view(analysis_id),
            "test_event_count": int(len(screened_dataset.test_index)),
            "test_temporal_coverage": temporal_coverage(screened_dataset, policy),
            "metrics_path": str(metrics_path),
        }
        row.update({key: test.get(key) for key in METRIC_KEYS})
        rows.append(row)
        print(f"completed temporal audit analysis={analysis_id} seed={seed} qwk={fmt(test.get('qwk'))}", flush=True)
    return rows


def subset_modalities(dataset: DailyAffectBagDataset, kept: tuple[int, ...]) -> DailyAffectBagDataset:
    mask = np.zeros_like(dataset.modality_mask, dtype=bool)
    mask[:, :, list(kept)] = dataset.modality_mask[:, :, list(kept)]
    return replace(dataset, modality_mask=mask)


def temporal_coverage(dataset: DailyAffectBagDataset, policy: str) -> float:
    window_mask = dataset.modality_mask[dataset.test_index].any(axis=2)
    if policy == "last_10s":
        selected = window_mask[:, -1:]
    elif policy == "last_30s":
        selected = window_mask[:, -5:]
    elif policy == "last_60s":
        selected = window_mask[:, -11:]
    elif policy == "first_30s":
        selected = window_mask[:, :5]
    else:
        selected = window_mask
    return float(selected.any(axis=1).mean())


def cross_day_drift_rows(dataset: DailyAffectBagDataset, *, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality_idx, modality in enumerate(MODALITIES):
        train_features, _ = event_features(dataset, dataset.train_index, modality_idx)
        reference_mean = train_features.mean(axis=0)
        reference_std = np.maximum(train_features.std(axis=0), 1e-6)
        reference_var = np.maximum(train_features.var(axis=0), 1e-8)
        test_days = sorted(set(dataset.day_id[dataset.test_index].tolist()))
        for day_id in test_days:
            day_indices = dataset.test_index[dataset.day_id[dataset.test_index] == day_id]
            day_features, availability = event_features(dataset, day_indices, modality_idx)
            if len(day_features) == 0:
                shift_rms = None
                variance_ratio = None
            else:
                standardized_shift = (day_features.mean(axis=0) - reference_mean) / reference_std
                shift_rms = float(np.sqrt(np.mean(np.square(standardized_shift))))
                ratio = np.maximum(day_features.var(axis=0), 1e-8) / reference_var
                variance_ratio = float(np.exp(np.mean(np.log(ratio))))
            rows.append(
                {
                    "seed": int(seed),
                    "modality": modality,
                    "day_id": str(day_id),
                    "reference_train_event_count": int(len(train_features)),
                    "test_event_count": int(len(day_indices)),
                    "test_available_event_fraction": availability,
                    "standardized_centroid_shift_rms": shift_rms,
                    "geometric_variance_ratio_vs_train": variance_ratio,
                }
            )
    return rows


def event_features(dataset: DailyAffectBagDataset, indices: np.ndarray, modality_idx: int) -> tuple[np.ndarray, float]:
    features: list[np.ndarray] = []
    available = 0
    for index in indices.tolist():
        valid = dataset.modality_mask[index, :, modality_idx]
        if np.any(valid):
            features.append(dataset.tokens[index, valid, modality_idx].mean(axis=0))
            available += 1
    if not features:
        return np.zeros((0, dataset.tokens.shape[-1]), dtype=np.float32), 0.0
    return np.stack(features).astype(np.float32), float(available / max(1, len(indices)))


def summarize_drift(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for modality in MODALITIES:
        values = [row for row in rows if row["modality"] == modality]
        shifts = finite_values(values, "standardized_centroid_shift_rms")
        ratios = finite_values(values, "geometric_variance_ratio_vs_train")
        availability = finite_values(values, "test_available_event_fraction")
        summary.append(
            {
                "modality": modality,
                "seed_day_rows": len(values),
                "mean_standardized_centroid_shift_rms": safe_mean(shifts),
                "max_standardized_centroid_shift_rms": max(shifts) if shifts else None,
                "mean_geometric_variance_ratio_vs_train": safe_mean(ratios),
                "mean_test_available_event_fraction": safe_mean(availability),
            }
        )
    return summary


def resolve_video_source(dataset: DailyAffectBagDataset) -> Path | None:
    try:
        sources = json.loads(dataset.source_npz_json)
    except json.JSONDecodeError:
        return None
    for key, value in sources.items():
        if key.startswith("video_"):
            return Path(str(value))
    return None


def video_quality_audit_rows(dataset: DailyAffectBagDataset, source_path: Path, *, seed: int) -> list[dict[str, Any]]:
    quality = load_video_quality(source_path)
    rows: list[dict[str, Any]] = []
    for split_name, indices in dataset.split_indices().items():
        for day_id in sorted(set(dataset.day_id[indices].tolist())):
            day_indices = indices[dataset.day_id[indices] == day_id]
            window_records = [quality.get(str(sample_id), default_quality_record()) for index in day_indices for sample_id in dataset.sample_id_matrix[index]]
            event_valid_counts = [int(dataset.modality_mask[index, :, 2].sum()) for index in day_indices]
            sampled = sum(record["sampled_frame_count"] for record in window_records)
            usable = sum(record["usable_frame_count"] for record in window_records)
            known = [record for record in window_records if record["quality_known"]]
            partial = [record for record in known if record["usable_frame_count"] < record["sampled_frame_count"]]
            rows.append(
                {
                    "seed": int(seed),
                    "split": split_name,
                    "day_id": str(day_id),
                    "event_count": int(len(day_indices)),
                    "window_count": int(len(window_records)),
                    "video_available_window_fraction": float(np.mean([record["video_available"] for record in window_records])) if window_records else None,
                    "video_available_event_fraction": float(np.mean([count > 0 for count in event_valid_counts])) if event_valid_counts else None,
                    "video_complete_event_fraction": float(np.mean([count == 23 for count in event_valid_counts])) if event_valid_counts else None,
                    "quality_known_window_fraction": float(len(known) / len(window_records)) if window_records else None,
                    "partial_frame_window_fraction": float(len(partial) / len(known)) if known else None,
                    "usable_frame_fraction": float(usable / sampled) if sampled else None,
                }
            )
    return rows


def video_quality_unavailable_rows(dataset: DailyAffectBagDataset, *, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, indices in dataset.split_indices().items():
        for day_id in sorted(set(dataset.day_id[indices].tolist())):
            day_indices = indices[dataset.day_id[indices] == day_id]
            event_valid_counts = [int(dataset.modality_mask[index, :, 2].sum()) for index in day_indices]
            rows.append(
                {
                    "seed": int(seed),
                    "split": split_name,
                    "day_id": str(day_id),
                    "event_count": int(len(day_indices)),
                    "window_count": int(len(day_indices) * 23),
                    "video_available_window_fraction": float(dataset.modality_mask[day_indices, :, 2].mean()),
                    "video_available_event_fraction": float(np.mean([count > 0 for count in event_valid_counts])),
                    "video_complete_event_fraction": float(np.mean([count == 23 for count in event_valid_counts])),
                    "quality_known_window_fraction": None,
                    "partial_frame_window_fraction": None,
                    "usable_frame_fraction": None,
                }
            )
    return rows


def load_video_quality(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as loaded:
        for sample_id, mask, raw_flags in zip(loaded["sample_id"], loaded["video_mask"], loaded["quality_flags"]):
            if not bool(mask):
                records[str(sample_id)] = default_quality_record()
                continue
            try:
                flags = json.loads(str(raw_flags))
            except json.JSONDecodeError:
                flags = {}
            sampled = int(flags.get("sampled_frame_count", 0))
            usable = int(flags.get("usable_frame_count", 0))
            records[str(sample_id)] = {
                "video_available": True,
                "quality_known": sampled > 0,
                "sampled_frame_count": sampled,
                "usable_frame_count": usable,
            }
    return records


def default_quality_record() -> dict[str, Any]:
    return {
        "video_available": False,
        "quality_known": False,
        "sampled_frame_count": 0,
        "usable_frame_count": 0,
    }


def summarize_temporal_screen(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for analysis_id in sorted(set(str(row["analysis_id"]) for row in rows)):
        values = [row for row in rows if row["analysis_id"] == analysis_id]
        row: dict[str, Any] = {
            "analysis_id": analysis_id,
            "temporal_policy": values[0]["temporal_policy"],
            "modality_view": values[0]["modality_view"],
            "seed_count": len(values),
            "mean_test_temporal_coverage": safe_mean(finite_values(values, "test_temporal_coverage")),
        }
        row.update({f"mean_{key}": safe_mean(finite_values(values, key)) for key in METRIC_KEYS})
        row.update({f"std_{key}": safe_std(finite_values(values, key)) for key in METRIC_KEYS})
        summary.append(row)
    return summary


def paired_temporal_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {(int(row["seed"])): row for row in rows if row["analysis_id"] == "full_2min"}
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["analysis_id"] == "full_2min":
            continue
        reference = baseline.get(int(row["seed"]))
        if reference is None:
            continue
        output = {
            "seed": int(row["seed"]),
            "analysis_id": row["analysis_id"],
            "baseline": "full_2min",
        }
        for key in METRIC_KEYS:
            value = row.get(key)
            base = reference.get(key)
            output[f"delta_{key}"] = None if value is None or base is None else float(value) - float(base)
        deltas.append(output)
    return deltas


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Daily-Affect Bottleneck Audit",
        "",
        f"- protocol: `{payload['protocol']}`",
        f"- route: `{payload['route_id']}`",
        f"- normalization: `{payload['normalization']}`",
        f"- seeds: `{','.join(map(str, payload['seeds']))}`",
        "- Boundary: temporal rows are paired supervised tests; drift and quality rows are input-only diagnostics.",
        "",
        "## Event-Level Label Identifiability",
        "",
        "| analysis | temporal policy | modality view | seeds | coverage | QWK | Macro-F1 | ordinal MAE | Expected RMSE | raw r | centered r |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["temporal_summary"]:
        lines.append(
            f"| {row['analysis_id']} | {row['temporal_policy']} | {row['modality_view']} | {row['seed_count']} | "
            f"{fmt(row['mean_test_temporal_coverage'])} | {fmt(row['mean_qwk'])} | {fmt(row['mean_macro_f1'])} | "
            f"{fmt(row['mean_ordinal_mae'])} | {fmt(row['mean_expected_rmse'])} | {fmt(row['mean_expected_raw_r'])} | "
            f"{fmt(row['mean_expected_within_subject_centered_r'])} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Day Representation Drift",
            "",
            "| modality | seed-day rows | mean centroid shift RMS | max centroid shift RMS | geometric variance ratio | test event availability |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["modality_drift_summary"]:
        lines.append(
            f"| {row['modality']} | {row['seed_day_rows']} | {fmt(row['mean_standardized_centroid_shift_rms'])} | "
            f"{fmt(row['max_standardized_centroid_shift_rms'])} | {fmt(row['mean_geometric_variance_ratio_vs_train'])} | "
            f"{fmt(row['mean_test_available_event_fraction'])} |"
        )
    video_rows = [row for row in payload["video_quality_by_day"] if row["split"] == "test"]
    lines.extend(
        [
            "",
            "## Video Stability",
            "",
            "| test-day rows | video window availability | video event availability | complete-video events | quality-known windows | partial-frame windows | usable-frame fraction |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {len(video_rows)} | {fmt(safe_mean(finite_values(video_rows, 'video_available_window_fraction')))} | "
            f"{fmt(safe_mean(finite_values(video_rows, 'video_available_event_fraction')))} | "
            f"{fmt(safe_mean(finite_values(video_rows, 'video_complete_event_fraction')))} | "
            f"{fmt(safe_mean(finite_values(video_rows, 'quality_known_window_fraction')))} | "
            f"{fmt(safe_mean(finite_values(video_rows, 'partial_frame_window_fraction')))} | "
            f"{fmt(safe_mean(finite_values(video_rows, 'usable_frame_fraction')))} |",
            "",
            "See the CSV files for seed-level paired temporal deltas, per-day drift, and video quality strata.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(temporal_summary: list[dict[str, Any]], drift_summary: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    temporal = [row for row in temporal_summary if row["modality_view"] == "all_modalities"]
    labels = [str(row["analysis_id"]).replace("kernel_", "k_").replace("_", "\n") for row in temporal]
    qwk = [float(row["mean_qwk"]) for row in temporal]
    qwk_std = [float(row["std_qwk"] or 0.0) for row in temporal]
    raw_r = [float(row["mean_expected_raw_r"]) for row in temporal]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(labels, qwk, yerr=qwk_std, color="#277da1", capsize=3)
    axes[0].set_title("Temporal policy: QWK")
    axes[0].set_ylabel("test QWK")
    axes[0].tick_params(axis="x", labelrotation=0)
    axes[1].bar(labels, raw_r, color="#43aa8b")
    axes[1].set_title("Temporal policy: expected raw r")
    axes[1].set_ylabel("test raw r")
    axes[1].tick_params(axis="x", labelrotation=0)
    figure.savefig(paths["temporal_plot"], dpi=180)
    plt.close(figure)

    labels = [str(row["modality"]) for row in drift_summary]
    shift = [float(row["mean_standardized_centroid_shift_rms"] or 0.0) for row in drift_summary]
    availability = [float(row["mean_test_available_event_fraction"] or 0.0) for row in drift_summary]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    width = 0.38
    axis.bar(positions - width / 2, shift, width, label="centroid shift RMS", color="#f94144")
    axis.bar(positions + width / 2, availability, width, label="test event availability", color="#577590")
    axis.set_xticks(positions, labels)
    axis.set_title("Cross-day modality drift and availability")
    axis.legend()
    figure.savefig(paths["drift_plot"], dpi=180)
    plt.close(figure)


def modality_view(analysis_id: str) -> str:
    if analysis_id == "video_only_full_2min":
        return "video_only"
    if analysis_id == "nonvideo_full_2min":
        return "eeg_wear_audio"
    return "all_modalities"


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]


def safe_mean(values: list[float]) -> float | None:
    return None if not values else float(mean(values))


def safe_std(values: list[float]) -> float | None:
    return None if len(values) < 2 else float(pstdev(values))


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
