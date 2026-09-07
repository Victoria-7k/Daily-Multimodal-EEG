#!/usr/bin/env python3
"""Separate token normalization and adapter sharing in a daily-affect routing diagnostic."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.daily_affect.training import load_bag_dataset, run_daily_affect_run


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SOURCE_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_20260903"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "outputs/daily_affect_routing_factorial_routefix_20260905"
DEFAULT_SEEDS = (240729, 240730, 240731)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_SOURCE_ROOT / "bags")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--model-id", default="state_uniform")
    parser.add_argument("--normalizations", default="shared,per_modality")
    parser.add_argument("--adapter-modes", default="shared,per_modality")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    torch.set_num_threads(max(1, int(args.torch_threads)))
    normalizations = split_csv(args.normalizations)
    adapter_modes = split_csv(args.adapter_modes)
    seeds = tuple(int(value) for value in split_csv(args.seeds))
    assert_valid_modes(normalizations, "normalizations")
    assert_valid_modes(adapter_modes, "adapter_modes")

    started = time.time()
    results: list[dict[str, Any]] = []
    for seed in seeds:
        bag_path = args.bags_root / args.protocol / args.route_id / f"seed_{seed}" / "ema_bags.npz"
        dataset = load_bag_dataset(bag_path)
        for normalization in normalizations:
            for adapter_mode in adapter_modes:
                condition_id = make_condition_id(normalization, adapter_mode)
                run_dir = args.out_root / "runs" / args.protocol / args.route_id / args.model_id / condition_id / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                if args.skip_existing and metrics_path.is_file():
                    result = json.loads(metrics_path.read_text(encoding="utf-8"))
                    print(f"skipped existing condition={condition_id} seed={seed}", flush=True)
                else:
                    print(f"starting condition={condition_id} seed={seed}", flush=True)
                    result = run_daily_affect_run(
                        dataset=dataset,
                        protocol=args.protocol,
                        model_id=args.model_id,
                        normalization=normalization,
                        adapter_mode=adapter_mode,
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
                        beta_ord=0.25,
                        probe_loss_weight=0.0,
                        ordinal_loss_weight=args.ordinal_loss_weight,
                        rank_loss_weight=args.rank_loss_weight,
                        modality_dropout_prob=args.modality_dropout_prob,
                        probe_warmup_epochs=0,
                        difficulty_ramp_epochs=0,
                        probe_kind="cumulative",
                        difficulty_mode="none",
                        detach_difficulty=True,
                        selection_metric="qwk",
                        class_balanced_loss=True,
                        calibrate_probe_temperature=False,
                        device=args.device,
                        include_diagnostics=True,
                        experiment_id="routing_factorial_v2",
                    )
                    print(
                        f"completed condition={condition_id} seed={seed} "
                        f"qwk={float(result['test']['qwk']):.4f}",
                        flush=True,
                    )
                result["condition_id"] = condition_id
                result["normalization"] = normalization
                result["adapter_mode"] = adapter_mode
                result["test_modality_weights"] = test_modality_weights(run_dir / "diagnostics.npz", dataset.test_index)
                results.append(result)

    payload = {
        "script": Path(__file__).name,
        "stage": "daily_affect_routing_factorial_diagnostic",
        "purpose": "Separate train-only token normalization scope from adapter sharing; diagnostic only, not a promotion gate.",
        "fixed_contract": {
            "protocol": args.protocol,
            "route_id": args.route_id,
            "model_id": args.model_id,
            "seeds": list(seeds),
            "bag_shape": "(N_ema,23,4,256)",
            "mask_contract": "modality_mask (N_ema,23,4)",
            "selection_metric": "validation QWK",
            "test_usage": "frozen evaluation after validation early stopping",
        },
        "interpretation_rule": (
            "Compare conditions while holding one factor fixed. A reduced Wear routing weight when only normalization changes "
            "attributes the saturation to normalization; a reduction when only adapter mode changes attributes it to the adapter; "
            "a reduction only in per_modality/per_modality indicates an interaction."
        ),
        "elapsed_seconds": float(time.time() - started),
        "results": results,
        "summary": summarize(results),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    json_path = args.out_root / "daily_affect_routing_factorial_report.json"
    md_path = args.out_root / "daily_affect_routing_factorial_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(f"run_count={len(results)}")
    print(f"out_json={json_path}")
    print(f"out_md={md_path}")
    return 0


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def assert_valid_modes(values: tuple[str, ...], name: str) -> None:
    invalid = sorted(set(values) - {"shared", "per_modality"})
    if invalid:
        raise ValueError(f"{name} has unsupported values: {invalid}")


def make_condition_id(normalization: str, adapter_mode: str) -> str:
    return f"norm_{normalization}__adapter_{adapter_mode}"


def test_modality_weights(path: Path, test_index: np.ndarray) -> list[float]:
    with np.load(path, allow_pickle=False) as data:
        weights = data["modality_weights"][test_index]
    return [float(value) for value in weights.mean(axis=(0, 1))]


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["normalization"], result["adapter_mode"])].append(result)
    summary: list[dict[str, Any]] = []
    for (normalization, adapter_mode), rows in sorted(grouped.items()):
        qwk = np.asarray([row["test"]["qwk"] for row in rows], dtype=float)
        mae = np.asarray([row["test"]["ordinal_mae"] for row in rows], dtype=float)
        weights = np.asarray([row["test_modality_weights"] for row in rows], dtype=float)
        summary.append(
            {
                "condition_id": make_condition_id(normalization, adapter_mode),
                "normalization": normalization,
                "adapter_mode": adapter_mode,
                "seed_count": len(rows),
                "test_qwk_mean": float(qwk.mean()),
                "test_qwk_std": float(qwk.std(ddof=1)) if len(qwk) > 1 else 0.0,
                "test_ordinal_mae_mean": float(mae.mean()),
                "test_ordinal_mae_std": float(mae.std(ddof=1)) if len(mae) > 1 else 0.0,
                "test_modality_weights_mean": [float(value) for value in weights.mean(axis=0)],
                "test_modality_weights_std": [float(value) for value in weights.std(axis=0, ddof=1)] if len(weights) > 1 else [0.0] * 4,
            }
        )
    return summary


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Daily-Affect Routing Factorial Diagnostic",
        "",
        "This is a mechanism diagnostic, not a mainline promotion gate.",
        "",
        f"- protocol: `{payload['fixed_contract']['protocol']}`",
        f"- route: `{payload['fixed_contract']['route_id']}`",
        f"- model: `{payload['fixed_contract']['model_id']}`",
        f"- seeds: `{payload['fixed_contract']['seeds']}`",
        "- comparison: train-only normalization scope x adapter sharing",
        "",
        "| condition | seeds | Test QWK | Test ordinal MAE | EEG weight | Wear weight | Video weight | Audio weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        weights = row["test_modality_weights_mean"]
        lines.append(
            f"| {row['condition_id']} | {row['seed_count']} | "
            f"{row['test_qwk_mean']:.4f} +/- {row['test_qwk_std']:.4f} | "
            f"{row['test_ordinal_mae_mean']:.4f} +/- {row['test_ordinal_mae_std']:.4f} | "
            + " | ".join(f"{value:.4f}" for value in weights)
            + " |"
        )
    lines.extend(["", "## Interpretation Rule", "", payload["interpretation_rule"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
