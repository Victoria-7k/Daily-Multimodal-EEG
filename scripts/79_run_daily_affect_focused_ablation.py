#!/usr/bin/env python3
"""Run a focused daily-affect ablation for one promising route."""

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
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "outputs/daily_affect_dynamic_a1_crossday_routefix_20260905"
DEFAULT_MODELS = (
    "bag_static",
    "state_uniform",
    "prior_ordD_uniform",
    "dynamic_fixed_short",
    "dynamic_fixed_medium",
    "dynamic_fixed_long",
    "global_kernel_no_prior",
    "dynamic_kernel_no_prior",
    "dynamic_kernel_prior_uniform",
    "dynamic_kernel",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_SOURCE_ROOT / "bags")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--normalization", default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--model-ids", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="240729,240730,240731")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lambda-d", type=float, default=0.25)
    parser.add_argument("--beta-ord", type=float, default=0.25)
    parser.add_argument("--probe-loss-weight", type=float, default=0.1)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.1)
    parser.add_argument("--probe-warmup-epochs", type=int, default=5)
    parser.add_argument("--difficulty-ramp-epochs", type=int, default=5)
    parser.add_argument("--selection-metric", choices=("qwk", "macro_f1", "accuracy", "ordinal_mae", "nll"), default="qwk")
    parser.add_argument("--class-balanced-loss", dest="class_balanced_loss", action="store_true", default=True)
    parser.add_argument("--no-class-balanced-loss", dest="class_balanced_loss", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    model_ids = split_csv(args.model_ids)
    seeds = [int(value) for value in split_csv(args.seeds)]
    started = time.time()
    results: list[dict[str, Any]] = []
    for seed in seeds:
        dataset = load_bag_dataset(args.bags_root / args.protocol / args.route_id / f"seed_{seed}" / "ema_bags.npz")
        for model_id in model_ids:
            section = "phase0" if model_id == "bag_static" else "runs"
            condition_id = f"norm_{args.normalization}__adapter_{args.adapter_mode}"
            run_dir = args.out_root / section / args.protocol / args.route_id / model_id / condition_id / f"seed_{seed}"
            if model_id == "bag_static":
                run_dir = args.out_root / section / args.protocol / args.route_id / condition_id / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            if args.skip_existing and metrics_path.is_file():
                results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                print(f"skipped existing protocol={args.protocol} route={args.route_id} model={model_id} normalization={args.normalization} seed={seed}", flush=True)
                continue
            print(f"starting protocol={args.protocol} route={args.route_id} model={model_id} normalization={args.normalization} seed={seed}", flush=True)
            lambda_d = 0.0 if model_id in {"bag_static", "state_uniform", "prior_uniform", "dynamic_kernel_no_prior", "dynamic_kernel_prior_uniform", "global_kernel_no_prior"} else float(args.lambda_d)
            probe_weight = 0.0 if model_id in {"bag_static", "state_uniform"} else float(args.probe_loss_weight)
            result = run_daily_affect_run(
                dataset=dataset,
                protocol=args.protocol,
                model_id=model_id,
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
                lambda_d=lambda_d,
                beta_ord=args.beta_ord,
                probe_loss_weight=probe_weight,
                ordinal_loss_weight=args.ordinal_loss_weight,
                rank_loss_weight=args.rank_loss_weight,
                modality_dropout_prob=args.modality_dropout_prob,
                probe_warmup_epochs=args.probe_warmup_epochs,
                difficulty_ramp_epochs=args.difficulty_ramp_epochs,
                probe_kind="cumulative",
                difficulty_mode="ordinal" if lambda_d > 0.0 else "none",
                detach_difficulty=True,
                selection_metric=args.selection_metric,
                class_balanced_loss=args.class_balanced_loss,
                calibrate_temperature=False,
                device=args.device,
                include_diagnostics=model_id != "bag_static",
                experiment_id="focused_ablation_v2",
            )
            results.append(result)
            print(
                f"completed protocol={args.protocol} route={args.route_id} model={model_id} normalization={args.normalization} seed={seed} "
                f"qwk={fmt(result['test']['qwk'])} macro_f1={fmt(result['test']['macro_f1'])}",
                flush=True,
            )
    output = {
        "script": Path(__file__).name,
        "stage": "daily_affect_focused_ablation",
        "protocol": args.protocol,
        "route_id": args.route_id,
        "normalization": args.normalization,
        "run_count": len(results),
        "elapsed_seconds": float(time.time() - started),
        "results": results,
    }
    report_json = args.out_root / "focused_ablation_report.json"
    report_md = args.out_root / "focused_ablation_report.md"
    args.out_root.mkdir(parents=True, exist_ok=True)
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
        "# Daily-Affect Focused Ablation Report",
        "",
        f"- protocol: `{output['protocol']}`",
        f"- route: `{output['route_id']}`",
        f"- normalization: `{output['normalization']}`",
        f"- run_count: `{output['run_count']}`",
        "",
        "| model | seed | QWK | Macro-F1 | accuracy | ordinal MAE | best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        lines.append(
            f"| {row['model_id']} | {row['seed']} | {fmt(row['test']['qwk'])} | {fmt(row['test']['macro_f1'])} | "
            f"{fmt(row['test']['accuracy'])} | {fmt(row['test']['ordinal_mae'])} | {row['train_audit']['best_epoch']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
