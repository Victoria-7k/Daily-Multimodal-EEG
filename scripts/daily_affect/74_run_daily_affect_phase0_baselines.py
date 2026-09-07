#!/usr/bin/env python3
"""Run paired window-replicated and EMA-bag Phase 0 daily-affect baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from daily_multimodal.daily_affect.training import load_bag_dataset, run_daily_affect_run


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SOURCE_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_20260903"
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_routefix_20260906"
DEFAULT_PROTOCOLS = ("cross_subject", "cross_day", "within_subject_day")
DEFAULT_SEEDS = (240729, 240730, 240731)
DEFAULT_MODELS = ("window_replicated", "bag_static")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_SOURCE_ROOT / "bags")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT / "phase0")
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--route-id", default="B0_Wphysio_full")
    parser.add_argument("--route-ids", help="Comma-separated route ids. Overrides --route-id.")
    parser.add_argument("--model-ids", default=",".join(DEFAULT_MODELS))
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
    parser.add_argument("--selection-metric", choices=("qwk", "macro_f1", "accuracy", "ordinal_mae", "nll"), default="qwk")
    parser.add_argument("--class-balanced-loss", dest="class_balanced_loss", action="store_true", default=True)
    parser.add_argument("--no-class-balanced-loss", dest="class_balanced_loss", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    protocols = split_csv(args.protocols)
    route_ids = split_csv(args.route_ids) if args.route_ids else (args.route_id,)
    model_ids = split_csv(args.model_ids)
    normalizations = split_csv(args.normalizations)
    adapter_modes = split_csv(args.adapter_modes)
    seeds = [int(value) for value in split_csv(args.seeds)]
    results: list[dict[str, Any]] = []
    run_number = 0
    started = time.time()
    for protocol in protocols:
        for route_id in route_ids:
            for seed in seeds:
                bag_path = args.bags_root / protocol / route_id / f"seed_{seed}" / "ema_bags.npz"
                dataset = load_bag_dataset(bag_path)
                for model_id in model_ids:
                    if model_id not in DEFAULT_MODELS:
                        raise ValueError(f"Phase 0 supports only {DEFAULT_MODELS}, got {model_id}")
                    for normalization in normalizations:
                        for adapter_mode in adapter_modes:
                            run_number += 1
                            if args.limit_runs and run_number > args.limit_runs:
                                break
                            condition_id = make_condition_id(normalization, adapter_mode)
                            run_dir = args.out_root / protocol / route_id / model_id / condition_id / f"seed_{seed}"
                            metrics_path = run_dir / "metrics.json"
                            if args.skip_existing and metrics_path.is_file():
                                results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                                print(f"skipped existing protocol={protocol} route={route_id} model={model_id} condition={condition_id} seed={seed}", flush=True)
                                continue
                            print(f"starting protocol={protocol} route={route_id} model={model_id} condition={condition_id} seed={seed}", flush=True)
                            result = run_daily_affect_run(
                                dataset=dataset,
                                protocol=protocol,
                                model_id=model_id,
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
                                selection_metric=args.selection_metric,
                                class_balanced_loss=args.class_balanced_loss,
                                calibrate_probe_temperature=False,
                                device=args.device,
                                include_diagnostics=False,
                                experiment_id="phase0_attention_supervision_v3",
                            )
                            results.append(result)
                            print(
                                f"completed protocol={protocol} route={route_id} model={model_id} condition={condition_id} seed={seed} "
                                f"qwk={fmt(result['test']['qwk'])} macro_f1={fmt(result['test']['macro_f1'])}",
                                flush=True,
                            )
                            if args.limit_runs and run_number >= args.limit_runs:
                                break
                        if args.limit_runs and run_number >= args.limit_runs:
                            break
                    if args.limit_runs and run_number >= args.limit_runs:
                        break
                if args.limit_runs and run_number >= args.limit_runs:
                    break
            if args.limit_runs and run_number >= args.limit_runs:
                break
        if args.limit_runs and run_number >= args.limit_runs:
            break
    output = {
        "script": Path(__file__).name,
        "stage": "daily_affect_phase0_attention_supervision",
        "run_count": len(results),
        "elapsed_seconds": float(time.time() - started),
        "results": results,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    report_json = args.out_root / "daily_affect_phase0_report.json"
    report_md = args.out_root / "daily_affect_phase0_report.md"
    report_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, report_md)
    print(f"run_count={len(results)}")
    print(f"out_json={report_json}")
    print(f"out_md={report_md}")
    return 0


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Daily-Affect Phase 0 Attention Supervision Report",
        "",
        f"- run_count: `{output['run_count']}`",
        "",
        "| protocol | route | model | supervision | normalization | adapter | objective | seed | test QWK | test Macro-F1 | test accuracy | test ordinal MAE | best epoch |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        lines.append(
            f"| {row['protocol']} | {row['route_id']} | {row['model_id']} | {row.get('supervision_unit', '')} | {row['normalization']} | {row['adapter_mode']} | {row['objective_id']} | {row['seed']} | "
            f"{fmt(row['test']['qwk'])} | {fmt(row['test']['macro_f1'])} | {fmt(row['test']['accuracy'])} | "
            f"{fmt(row['test']['ordinal_mae'])} | {row['train_audit']['best_epoch']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def make_condition_id(normalization: str, adapter_mode: str) -> str:
    return f"norm_{normalization}__adapter_{adapter_mode}"


if __name__ == "__main__":
    raise SystemExit(main())
