#!/usr/bin/env python3
"""Evaluate focused daily-affect runs under test-time missing/corrupted modalities."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import fields, replace
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from daily_multimodal.daily_affect.metrics import classification_metrics, metric_aliases
from daily_multimodal.daily_affect.model import DailyAffectConfig, DailyAffectOrdinalModel
from daily_multimodal.daily_affect.training import load_bag_dataset, predict_daily_affect_model


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_RUN_ROOT = DEFAULT_ROOT / "outputs/daily_affect_dynamic_a1_crossday_routefix_20260905"
MODALITIES = ("eeg", "wear", "video", "audio")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "robustness")
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--normalization", default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--models", default="bag_static,dynamic_kernel")
    parser.add_argument("--seeds", default="240729,240730,240731,240732,240733,240734,240735")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    models = split_csv(args.models)
    seeds = [int(value) for value in split_csv(args.seeds)]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for model_id in models:
            metrics_path = find_metrics_path(
                args.run_root,
                args.protocol,
                args.route_id,
                model_id,
                args.normalization,
                args.adapter_mode,
                seed,
            )
            if metrics_path is None:
                print(f"missing metrics model={model_id} seed={seed}", flush=True)
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            dataset = load_bag_dataset(Path(metrics["bag_path"]))
            model = load_checkpoint_model(Path(metrics["checkpoint_path"]), device=args.device)
            scenarios = [("clean", "none")]
            scenarios.extend(("missing", modality) for modality in MODALITIES)
            scenarios.extend(("missing_pair", "+".join(pair)) for pair in combinations(MODALITIES, 2))
            scenarios.extend(("noise", modality) for modality in MODALITIES)
            scenarios.extend(("shuffle", modality) for modality in MODALITIES)
            for scenario, modality in scenarios:
                perturbed, perturbation = perturb_dataset(
                    dataset,
                    scenario=scenario,
                    modality=modality,
                    seed=seed,
                    noise_scale=args.noise_scale,
                )
                pred = predict_daily_affect_model(model, perturbed, indices=perturbed.test_index, device=args.device, include_diagnostics=True)
                test = metric_aliases(
                    classification_metrics(
                        perturbed.label_zero_based[perturbed.test_index],
                        pred["predicted_class"],
                        expected_score=pred["expected_score"],
                        subject_ids=perturbed.subject_id[perturbed.test_index],
                        probabilities=pred["probabilities"],
                    )
                )
                rows.append(
                    {
                        "protocol": args.protocol,
                        "route_id": args.route_id,
                        "normalization": args.normalization,
                        "adapter_mode": args.adapter_mode,
                        "model_id": model_id,
                        "seed": seed,
                        "scenario": scenario,
                        "modality": modality,
                        "test_qwk": test.get("qwk"),
                        "test_macro_f1": test.get("macro_f1"),
                        "test_accuracy": test.get("accuracy"),
                        "test_balanced_accuracy": test.get("balanced_accuracy"),
                        "test_ordinal_mae": test.get("ordinal_mae"),
                        "test_severe_error_rate": test.get("severe_error_rate"),
                        "test_expected_rmse": test.get("expected_rmse"),
                        "test_expected_raw_r": test.get("expected_raw_r"),
                        "test_expected_within_subject_centered_r": test.get("expected_within_subject_centered_r"),
                        "metrics_path": str(metrics_path),
                        **perturbation,
                        **diagnostic_means(pred),
                    }
                )
                print(f"evaluated model={model_id} seed={seed} scenario={scenario}:{modality} qwk={fmt(test.get('qwk'))}", flush=True)
    add_clean_deltas(rows)
    summary = summarize(rows)
    args.out_root.mkdir(parents=True, exist_ok=True)
    details_csv = args.out_root / "focused_robustness_details.csv"
    summary_csv = args.out_root / "focused_robustness_summary.csv"
    out_json = args.out_root / "focused_robustness_report.json"
    out_md = args.out_root / "focused_robustness_report.md"
    write_csv(details_csv, rows)
    write_csv(summary_csv, summary)
    out_json.write_text(json.dumps({"script": Path(__file__).name, "details": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"detail_count={len(rows)}")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    return 0


def find_metrics_path(
    run_root: Path,
    protocol: str,
    route_id: str,
    model_id: str,
    normalization: str,
    adapter_mode: str,
    seed: int,
) -> Path | None:
    condition_id = f"norm_{normalization}__adapter_{adapter_mode}"
    if model_id == "bag_static":
        direct = run_root / "phase0" / protocol / route_id / condition_id / f"seed_{seed}" / "metrics.json"
    else:
        direct = run_root / "runs" / protocol / route_id / model_id / condition_id / f"seed_{seed}" / "metrics.json"
    if direct.is_file():
        return direct
    for path in sorted(run_root.rglob("metrics.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            row.get("protocol") == protocol
            and row.get("route_id") == route_id
            and row.get("model_id") == model_id
            and row.get("normalization") == normalization
            and row.get("adapter_mode") == adapter_mode
            and int(row.get("seed", -1)) == seed
        ):
            return path
    return None


def load_checkpoint_model(checkpoint_path: Path, *, device: str) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_keys = {field.name for field in fields(DailyAffectConfig)}
    config = dict(ckpt["config"])
    if "adapter_mode" not in config and "normalization" in config:
        config["adapter_mode"] = config["normalization"]
    cfg = DailyAffectConfig(**{key: value for key, value in config.items() if key in cfg_keys})
    module = DailyAffectOrdinalModel(cfg).to(torch.device(device))
    module.load_state_dict(ckpt["state_dict"])
    module.eval()
    return {"module": module, "x_mean": ckpt["x_mean"], "x_std": ckpt["x_std"], "config": config}


def perturb_dataset(
    dataset: Any,
    *,
    scenario: str,
    modality: str,
    seed: int,
    noise_scale: float,
) -> tuple[Any, dict[str, Any]]:
    tokens = dataset.tokens.copy()
    mask = dataset.modality_mask.copy()
    test = dataset.test_index
    if scenario == "clean":
        return dataset, {"requested_modalities": "none", "perturbed_test_event_count": 0, "protected_test_event_count": 0}
    requested = tuple(part.strip() for part in modality.split("+") if part.strip())
    if not requested or any(part not in MODALITIES for part in requested):
        raise ValueError(f"unsupported modality specification: {modality}")
    indices = tuple(MODALITIES.index(part) for part in requested)
    if scenario in {"noise", "shuffle"} and len(indices) != 1:
        raise ValueError(f"scenario={scenario} accepts exactly one modality")
    changed = 0
    protected = 0
    if scenario == "missing":
        changed, protected = remove_modalities_preserving_one(tokens, mask, test, indices)
    elif scenario == "missing_pair":
        changed, protected = remove_modalities_preserving_one(tokens, mask, test, indices)
    elif scenario == "noise":
        idx = indices[0]
        rng = np.random.default_rng(seed + idx * 1009)
        valid_train = dataset.modality_mask[dataset.train_index, :, idx].astype(bool)
        train_values = dataset.tokens[dataset.train_index, :, idx, :][valid_train]
        scale = float(np.nanstd(train_values)) if train_values.size else 1.0
        scale = max(scale, 1e-6) * float(noise_scale)
        valid_test = mask[test, :, idx].astype(bool)
        noise = rng.normal(0.0, scale, size=tokens[test, :, idx, :].shape).astype(np.float32)
        tokens[test, :, idx, :] = np.where(valid_test[:, :, None], tokens[test, :, idx, :] + noise, tokens[test, :, idx, :])
    elif scenario == "shuffle":
        idx = indices[0]
        rng = random.Random(seed + idx * 917)
        order = list(range(len(test)))
        rng.shuffle(order)
        tokens[test, :, idx, :] = tokens[test[order], :, idx, :]
        mask[test, :, idx] = mask[test[order], :, idx]
    else:
        raise ValueError(f"unsupported scenario: {scenario}")
    return (
        replace(dataset, tokens=tokens, modality_mask=mask),
        {
            "requested_modalities": "+".join(requested),
            "perturbed_test_event_count": int(changed if scenario.startswith("missing") else len(test)),
            "protected_test_event_count": int(protected),
        },
    )


def remove_modalities_preserving_one(
    tokens: np.ndarray,
    mask: np.ndarray,
    test_index: np.ndarray,
    modality_indices: tuple[int, ...],
) -> tuple[int, int]:
    """Remove requested bag-wide modalities only when another active modality remains."""
    changed = 0
    protected = 0
    for event_index in test_index.tolist():
        active = mask[event_index].any(axis=0)
        remaining = active.copy()
        remaining[list(modality_indices)] = False
        if not bool(remaining.any()):
            protected += 1
            continue
        before = int(mask[event_index, :, list(modality_indices)].sum())
        mask[event_index, :, list(modality_indices)] = False
        tokens[event_index, :, list(modality_indices), :] = 0.0
        if before:
            changed += 1
    return changed, protected


DIAGNOSTIC_FIELDS = tuple(f"test_routing_weight_{modality}" for modality in MODALITIES) + tuple(
    f"test_probe_difficulty_{modality}" for modality in MODALITIES
)
METRIC_FIELDS = (
    "test_qwk",
    "test_macro_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_ordinal_mae",
    "test_severe_error_rate",
    "test_expected_rmse",
    "test_expected_raw_r",
    "test_expected_within_subject_centered_r",
)


def diagnostic_means(prediction: dict[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {field: None for field in DIAGNOSTIC_FIELDS}
    weights = prediction.get("modality_weights")
    if weights is not None:
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim == 3 and values.shape[-1] == len(MODALITIES):
            for index, modality in enumerate(MODALITIES):
                result[f"test_routing_weight_{modality}"] = float(np.nanmean(values[:, :, index]))
    difficulty = prediction.get("modality_difficulty")
    if difficulty is not None:
        values = np.asarray(difficulty, dtype=np.float64)
        if values.ndim == 2 and values.shape[-1] == len(MODALITIES):
            for index, modality in enumerate(MODALITIES):
                result[f"test_probe_difficulty_{modality}"] = float(np.nanmean(values[:, index]))
    return result


def add_clean_deltas(rows: list[dict[str, Any]]) -> None:
    clean_by_key = {
        (row["model_id"], row["seed"]): row
        for row in rows
        if row["scenario"] == "clean"
    }
    for row in rows:
        clean = clean_by_key.get((row["model_id"], row["seed"]))
        row["clean_reference_available"] = clean is not None
        for field in METRIC_FIELDS + DIAGNOSTIC_FIELDS:
            row[f"delta_{field}_vs_clean"] = numeric_delta(row.get(field), clean.get(field) if clean else None)


def numeric_delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    try:
        left = float(value)
        right = float(baseline)
    except (TypeError, ValueError):
        return None
    return left - right if np.isfinite(left) and np.isfinite(right) else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model_id"], row["scenario"], row["modality"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (model_id, scenario, modality), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "model_id": model_id,
            "scenario": scenario,
            "modality": modality,
            "seed_count": len(values),
            "mean_perturbed_test_event_count": safe_mean(numeric_values(values, "perturbed_test_event_count")),
            "mean_protected_test_event_count": safe_mean(numeric_values(values, "protected_test_event_count")),
        }
        for field in METRIC_FIELDS + DIAGNOSTIC_FIELDS:
            label = field.removeprefix("test_")
            row[f"mean_{label}"] = safe_mean(numeric_values(values, field))
            row[f"mean_delta_{label}_vs_clean"] = safe_mean(numeric_values(values, f"delta_{field}_vs_clean"))
        row["std_qwk"] = safe_std(numeric_values(values, "test_qwk"))
        out.append(row)
    return out


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def write_markdown(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Daily-Affect Focused Robustness Report",
        "",
        "| model | scenario | modality | seeds | perturbed events | protected events | QWK | delta QWK vs clean | Macro-F1 | balanced accuracy | ordinal MAE | severe error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['model_id']} | {row['scenario']} | {row['modality']} | {row['seed_count']} | "
            f"{fmt(row['mean_perturbed_test_event_count'])} | {fmt(row['mean_protected_test_event_count'])} | "
            f"{fmt(row['mean_qwk'])} | {fmt(row['mean_delta_qwk_vs_clean'])} | {fmt(row['mean_macro_f1'])} | "
            f"{fmt(row['mean_balanced_accuracy'])} | {fmt(row['mean_ordinal_mae'])} | {fmt(row['mean_severe_error_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Regression Bridge",
            "",
            "| model | scenario | modality | Expected RMSE | delta RMSE vs clean | Expected raw r | delta raw r vs clean | Expected centered r | delta centered r vs clean |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['model_id']} | {row['scenario']} | {row['modality']} | {fmt(row['mean_expected_rmse'])} | "
            f"{fmt(row['mean_delta_expected_rmse_vs_clean'])} | {fmt(row['mean_expected_raw_r'])} | "
            f"{fmt(row['mean_delta_expected_raw_r_vs_clean'])} | {fmt(row['mean_expected_within_subject_centered_r'])} | "
            f"{fmt(row['mean_delta_expected_within_subject_centered_r_vs_clean'])} |"
        )
    lines.extend(
        [
            "",
            "## Routing Response",
            "",
            "| model | scenario | modality | EEG delta weight | Wear delta weight | Video delta weight | Audio delta weight | EEG difficulty | Wear difficulty | Video difficulty | Audio difficulty |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['model_id']} | {row['scenario']} | {row['modality']} | "
            + " | ".join(fmt(row[f"mean_delta_routing_weight_{name}_vs_clean"]) for name in MODALITIES)
            + " | "
            + " | ".join(fmt(row[f"mean_probe_difficulty_{name}"]) for name in MODALITIES)
            + " |"
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


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def safe_mean(values: list[float]) -> float | str:
    return "" if not values else float(mean(values))


def safe_std(values: list[float]) -> float | str:
    return "" if len(values) < 2 else float(pstdev(values))


def fmt(value: Any) -> str:
    return "NA" if value == "" or value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
