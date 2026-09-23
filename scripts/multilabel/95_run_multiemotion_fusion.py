#!/usr/bin/env python3
"""Run event-aware E0/H0/H1 fixed-token multi-emotion fusion experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.split_paths import resolve_protocol_split_root
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.training.multihead_regression import (
    HEAD_VARIANTS,
    LABEL_NAMES,
    evaluate_event_level,
    event_level_arrays,
    fit_event_aware_model,
    predict,
)


def load_base_module():
    path = Path(__file__).with_name("48_run_fixed_tokens_multilabel_fusion.py")
    spec = importlib.util.spec_from_file_location("_fixed_multilabel_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
install_numpy_core_pickle_aliases()
DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_SPLITS_ROOT = DEFAULT_ROOT / "outputs/splits"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--protocols", default="cross_day,date_in_order")
    parser.add_argument("--experiments", default=",".join(BASE.EXPERIMENT_BRANCHES))
    parser.add_argument("--head-variants", default="E0_existing_11out")
    parser.add_argument("--seeds", default="240800,240801,240802")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    protocols = split_csv(args.protocols)
    experiments = split_csv(args.experiments)
    head_variants = split_csv(args.head_variants)
    seeds = tuple(int(value) for value in split_csv(args.seeds))
    unknown_heads = sorted(set(head_variants) - set(HEAD_VARIANTS))
    unknown_experiments = sorted(set(experiments) - set(BASE.EXPERIMENT_BRANCHES))
    if unknown_heads or unknown_experiments:
        raise ValueError(f"unknown heads={unknown_heads} experiments={unknown_experiments}")

    rows = BASE._load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([BASE._norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    day_id = np.asarray([str(row.get("day_id", "")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    targets = BASE._load_targets(rows)
    cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    run_number = 0
    for protocol in protocols:
        protocol_root = resolve_protocol_split_root(args.splits_root, protocol)
        split = BASE._load_split(protocol_root, len(rows))
        for experiment in experiments:
            branches = BASE.EXPERIMENT_BRANCHES[experiment]
            branch_data = BASE._load_all_branches(args.embeddings_root, sample_id, branches, cache)
            tokens, token_mask, branch_report = BASE._build_tokens(branch_data, branches)
            for head_variant in head_variants:
                for seed in seeds:
                    run_number += 1
                    if args.limit_runs and run_number > args.limit_runs:
                        break
                    print(f"starting protocol={protocol} route={experiment} head={head_variant} seed={seed}", flush=True)
                    bundle, train_audit = fit_event_aware_model(
                        tokens=tokens,
                        token_mask=token_mask,
                        targets=targets,
                        event_id=event_id,
                        train_idx=split["train"],
                        val_idx=split["val"],
                        head_variant=head_variant,
                        hidden_dim=args.hidden_dim,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        dropout=args.dropout,
                        patience=args.patience,
                        seed=seed,
                        device=args.device,
                    )
                    row: dict[str, Any] = {
                        "protocol": protocol,
                        "protocol_split_root": str(protocol_root),
                        "experiment": experiment,
                        "head_variant": head_variant,
                        "seed": seed,
                        "branches": list(branches),
                        "branch_report": branch_report,
                        "train_audit": train_audit,
                        "split_counts": {name: int(len(index)) for name, index in split.items()},
                    }
                    prediction_payload: dict[str, dict[str, np.ndarray]] = {}
                    for leaf in ("val", "test"):
                        prediction = predict(bundle, tokens, token_mask, split[leaf], args.device)
                        arrays = event_level_arrays(
                            targets[split[leaf]], prediction, event_id[split[leaf]], subject_id[split[leaf]], day_id[split[leaf]]
                        )
                        row[leaf] = evaluate_event_level(arrays, bundle["y_std"])
                        prediction_payload[leaf] = arrays
                    run_dir = args.out_root / "runs" / protocol / experiment / head_variant / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    metrics_path = run_dir / "metrics.json"
                    metrics_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                    np.savez_compressed(
                        run_dir / "event_predictions.npz",
                        label_names=np.asarray(LABEL_NAMES),
                        val_event_id=prediction_payload["val"]["event_id"],
                        val_subject_id=prediction_payload["val"]["subject_id"],
                        val_day_id=prediction_payload["val"]["day_id"],
                        val_target=prediction_payload["val"]["target"],
                        val_prediction=prediction_payload["val"]["prediction"],
                        test_event_id=prediction_payload["test"]["event_id"],
                        test_subject_id=prediction_payload["test"]["subject_id"],
                        test_day_id=prediction_payload["test"]["day_id"],
                        test_target=prediction_payload["test"]["target"],
                        test_prediction=prediction_payload["test"]["prediction"],
                    )
                    row["metrics_path"] = str(metrics_path)
                    results.append(row)
                    summary = row["test"]["summary"]
                    print(
                        f"completed protocol={protocol} route={experiment} head={head_variant} seed={seed} "
                        f"srmse={summary['standardized_rmse']:.4f} raw_r={summary['raw_r']:.4f}",
                        flush=True,
                    )
                if args.limit_runs and run_number >= args.limit_runs:
                    break
            if args.limit_runs and run_number >= args.limit_runs:
                break
        if args.limit_runs and run_number >= args.limit_runs:
            break
    output = {
        "stage": "multiemotion_fixed_token_event_screen",
        "root": str(args.root),
        "embeddings_root": str(args.embeddings_root),
        "splits_root": str(args.splits_root),
        "protocols": list(protocols),
        "experiments": list(experiments),
        "head_variants": list(head_variants),
        "seeds": list(seeds),
        "run_count": len(results),
        "selection_metric": "validation event-level macro standardized RMSE",
        "results": results,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, args.out_root / "report.md")
    print(f"run_count={len(results)}")
    return 0


def write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Multi-emotion Fixed-token Event Screen",
        "",
        f"run_count: `{output['run_count']}`",
        "",
        "| protocol | route | head | seed | val sRMSE | test sRMSE | test raw r | fatigue raw r |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        lines.append(
            f"| {row['protocol']} | {row['experiment']} | {row['head_variant']} | {row['seed']} | "
            f"{row['val']['summary']['standardized_rmse']:.4f} | {row['test']['summary']['standardized_rmse']:.4f} | "
            f"{row['test']['summary']['raw_r']:.4f} | {row['test']['per_label']['fatigue']['raw_r']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
