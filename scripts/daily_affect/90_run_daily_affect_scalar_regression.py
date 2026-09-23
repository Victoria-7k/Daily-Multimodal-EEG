#!/usr/bin/env python3
"""Run the paired Daily-affect window and EMA-bag scalar-regression matrix."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.daily_affect.regression_training import run_daily_affect_regression_run
from daily_multimodal.daily_affect.training import load_bag_dataset


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_BAGS_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_20260903/bags"
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "outputs/daily_affect_scalar_regression_20260908"
DEFAULT_SEEDS = (240729, 240730, 240731)
DEFAULT_PROTOCOLS = ("cross_day", "date_in_order", "cross_subject")
ROUTE_BY_PROTOCOL = {"cross_day": "A1_Wphysio_full", "date_in_order": "B0_Wphysio_full", "cross_subject": "A2_Wdeep_full"}
NORM_BY_PROTOCOL = {"cross_day": "per_modality", "date_in_order": "per_modality", "cross_subject": "shared"}
BAG_MODEL_IDS = (
    "bag_static", "state_uniform", "prior_uniform", "prior_ordD_uniform", "global_kernel_no_prior",
    "dynamic_kernel_no_prior", "dynamic_kernel_prior_uniform", "dynamic_kernel", "dynamic_fixed_short",
    "dynamic_fixed_medium", "dynamic_fixed_long",
)
STATIC_POLICIES = ("uniform", "last_10s", "last_30s", "last_60s", "first_30s", "kernel_short", "kernel_medium", "kernel_long")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--route-map", default=csv_map(ROUTE_BY_PROTOCOL))
    parser.add_argument("--normalization-map", default=csv_map(NORM_BY_PROTOCOL))
    parser.add_argument("--model-ids", default=",".join(BAG_MODEL_IDS))
    parser.add_argument("--static-temporal-policies", default=",".join(STATIC_POLICIES))
    parser.add_argument("--stage", choices=("preflight", "smoke", "matrix"), default="matrix")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lambda-d", type=float, default=0.25)
    parser.add_argument("--probe-loss-weight", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.1)
    parser.add_argument("--probe-warmup-epochs", type=int, default=5)
    parser.add_argument("--difficulty-ramp-epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    protocols = split_csv(args.protocols)
    seeds = tuple(int(value) for value in split_csv(args.seeds))
    routes, normalizations = parse_map(args.route_map), parse_map(args.normalization_map)
    missing = [name for name in protocols if name not in routes or name not in normalizations]
    if missing:
        raise ValueError(f"route/normalization mapping missing protocols: {missing}")
    conditions = make_conditions(split_csv(args.model_ids), split_csv(args.static_temporal_policies))
    if args.stage == "preflight":
        return write_preflight(args.out_root / "preflight", args.bags_root, protocols, seeds, routes, normalizations)
    if args.stage == "smoke":
        smoke_names = {"window_attention_regression_full_mean", "bag_static_reg__temporal_uniform", "prior_regD_uniform_reg", "dynamic_kernel_reg"}
        conditions = [row for row in conditions if row["condition_id"] in smoke_names]
        protocols, seeds = ("cross_day",), (DEFAULT_SEEDS[0],)
    started, results, run_count = time.time(), [], 0
    for protocol in protocols:
        for seed in seeds:
            route_id, normalization = routes[protocol], normalizations[protocol]
            dataset = load_bag_dataset(args.bags_root / protocol / route_id / f"seed_{seed}" / "ema_bags.npz")
            for condition in conditions:
                if args.limit_runs and run_count >= args.limit_runs:
                    break
                run_dir = args.out_root / "runs" / protocol / route_id / condition["condition_id"] / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                if args.skip_existing and metrics_path.is_file():
                    results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    continue
                print(f"starting protocol={protocol} seed={seed} condition={condition['condition_id']}", flush=True)
                result = run_daily_affect_regression_run(
                    dataset=dataset, protocol=protocol, condition_id=condition["condition_id"], model_id=condition["model_id"],
                    normalization=normalization, adapter_mode=normalization, temporal_policy=condition["temporal_policy"], seed=seed,
                    run_dir=run_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, dropout=args.dropout, hidden_dim=args.hidden_dim, patience=args.patience,
                    lambda_d=args.lambda_d, probe_loss_weight=args.probe_loss_weight,
                    modality_dropout_prob=args.modality_dropout_prob, probe_warmup_epochs=args.probe_warmup_epochs,
                    difficulty_ramp_epochs=args.difficulty_ramp_epochs, device=args.device,
                )
                results.append(result); run_count += 1
                print(f"completed condition={condition['condition_id']} raw_r={fmt(result['test']['raw_r'])} rmse={fmt(result['test']['rmse'])}", flush=True)
            if args.limit_runs and run_count >= args.limit_runs:
                break
        if args.limit_runs and run_count >= args.limit_runs:
            break
    report = {"script": Path(__file__).name, "stage": args.stage, "run_count": len(results), "elapsed_seconds": time.time() - started, "results": results}
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / f"scalar_regression_{args.stage}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_count={len(results)} out_root={args.out_root}")
    return 0


def make_conditions(model_ids: tuple[str, ...], policies: tuple[str, ...]) -> list[dict[str, str]]:
    invalid = set(model_ids) - set(BAG_MODEL_IDS)
    if invalid:
        raise ValueError(f"unknown EMA-bag scalar-regression model ids: {sorted(invalid)}")
    invalid_policies = set(policies) - set(STATIC_POLICIES)
    if invalid_policies:
        raise ValueError(f"unknown static temporal policies: {sorted(invalid_policies)}")
    conditions = [{"condition_id": "window_attention_regression_full_mean", "model_id": "window_replicated", "temporal_policy": "uniform"}]
    for model_id in model_ids:
        model_name = "prior_regD_uniform_reg" if model_id == "prior_ordD_uniform" else f"{model_id}_reg"
        policy_values = policies if model_id == "bag_static" else ("uniform",)
        for policy in policy_values:
            suffix = f"__temporal_{policy}" if model_id == "bag_static" else ""
            conditions.append({"condition_id": f"{model_name}{suffix}", "model_id": model_id, "temporal_policy": policy})
    return conditions


def write_preflight(out_dir: Path, bags_root: Path, protocols: tuple[str, ...], seeds: tuple[int, ...], routes: dict[str, str], normalizations: dict[str, str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        for seed in seeds:
            path = bags_root / protocol / routes[protocol] / f"seed_{seed}" / "ema_bags.npz"
            dataset = load_bag_dataset(path)
            split = dataset.split_indices()
            memberships = {name: dataset.event_id[indices] for name, indices in split.items()}
            train_val_test = [set(memberships[name].tolist()) for name in ("train", "val", "test")]
            overlap = sum(len(train_val_test[left] & train_val_test[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
            row = {
                "protocol": protocol, "seed": seed, "route_id": routes[protocol], "normalization": normalizations[protocol],
                "bag_path": str(path), "event_count": dataset.row_count, "token_shape": list(dataset.tokens.shape),
                "window_count_per_event": int(dataset.tokens.shape[1]), "mask_shape": list(dataset.modality_mask.shape),
                "train_events": len(split["train"]), "val_events": len(split["val"]), "test_events": len(split["test"]),
                "duplicate_event_ids": int(dataset.row_count - len(set(dataset.event_id.tolist()))), "split_event_overlap": overlap,
                "supervision_boundary": dataset.supervision_boundary, "source_npz_json": dataset.source_npz_json,
            }
            if row["token_shape"][1:] != [23, 4, 256] or row["mask_shape"] != row["token_shape"][:3] or overlap or row["duplicate_event_ids"]:
                raise ValueError(f"failed EMA-bag preflight: {row}")
            rows.append(row)
    fields = list(rows[0]) if rows else []
    with (out_dir / "event_split_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out_dir / "manifest.json").write_text(json.dumps({"status": "passed", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"preflight passed rows={len(rows)} out_dir={out_dir}")
    return 0


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_map(value: str) -> dict[str, str]:
    return {part.split("=", 1)[0].strip(): part.split("=", 1)[1].strip() for part in split_csv(value) if "=" in part}


def csv_map(values: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in values.items())


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
