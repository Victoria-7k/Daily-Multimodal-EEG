#!/usr/bin/env python3
"""Run daily-affect state/prior/kernel model matrix from EMA bags."""

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
DEFAULT_MODELS = ("state_uniform", "prior_uniform", "prior_ordD_uniform", "global_kernel_no_prior", "dynamic_kernel_no_prior", "dynamic_kernel")
DEFAULT_SEEDS = (240729, 240730, 240731)
PRIOR_GUIDED_MODELS = {
    "prior_uniform",
    "prior_ordD_uniform",
    "dynamic_kernel_prior_uniform",
    "dynamic_kernel",
    "dynamic_fixed_short",
    "dynamic_fixed_medium",
    "dynamic_fixed_long",
}
NO_PRIOR_MODELS = {"bag_static", "state_uniform", "dynamic_kernel_no_prior", "global_kernel_no_prior"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_SOURCE_ROOT / "bags")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT / "runs")
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
    parser.add_argument("--lambda-d", type=float, default=0.25)
    parser.add_argument("--beta-ord", type=float, default=0.25)
    parser.add_argument("--probe-loss-weight", type=float, default=0.1)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.5)
    parser.add_argument("--rank-loss-weight", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.1)
    parser.add_argument("--probe-warmup-epochs", type=int, default=5)
    parser.add_argument("--difficulty-ramp-epochs", type=int, default=5)
    parser.add_argument("--probe-kind", choices=("cumulative", "categorical"), default="cumulative")
    parser.add_argument("--difficulty-mode", choices=("auto", "none", "entropy", "ordinal"), default="auto")
    parser.add_argument("--no-detach-difficulty", dest="detach_difficulty", action="store_false", default=True)
    parser.add_argument(
        "--routing-profiles",
        default="native",
        help="Comma-separated native,p0_no_prior,p1_categorical_entropy,p2_cumulative_entropy,p3_ordinal_mix,p4_probe_calibrated,p5_end_to_end.",
    )
    parser.add_argument("--selection-metric", choices=("qwk", "macro_f1", "accuracy", "ordinal_mae", "nll"), default="qwk")
    parser.add_argument("--class-balanced-loss", dest="class_balanced_loss", action="store_true", default=True)
    parser.add_argument("--no-class-balanced-loss", dest="class_balanced_loss", action="store_false")
    parser.add_argument("--calibrate-probe-temperature", action="store_true")
    parser.add_argument("--calibration-finetune-epochs", type=int, default=5)
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
    routing_profiles = split_csv(args.routing_profiles)
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
                    for routing_profile in routing_profiles:
                        routing = routing_settings(routing_profile, model_id, args)
                        for normalization in normalizations:
                            for adapter_mode in adapter_modes:
                                run_number += 1
                                if args.limit_runs and run_number > args.limit_runs:
                                    break
                                condition_id = make_condition_id(normalization, adapter_mode, routing_profile)
                                run_dir = args.out_root / protocol / route_id / model_id / condition_id / f"seed_{seed}"
                                metrics_path = run_dir / "metrics.json"
                                if args.skip_existing and metrics_path.is_file():
                                    results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                                    print(f"skipped protocol={protocol} route={route_id} model={model_id} profile={routing_profile} condition={condition_id} seed={seed}", flush=True)
                                    continue
                                print(f"starting protocol={protocol} route={route_id} model={model_id} profile={routing_profile} condition={condition_id} seed={seed}", flush=True)
                                probe_weight = 0.0 if model_id in NO_PRIOR_MODELS else float(args.probe_loss_weight)
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
                                    lambda_d=routing["lambda_d"],
                                    beta_ord=routing["beta_ord"],
                                    probe_loss_weight=probe_weight,
                                    ordinal_loss_weight=args.ordinal_loss_weight,
                                    rank_loss_weight=args.rank_loss_weight,
                                    modality_dropout_prob=args.modality_dropout_prob,
                                    probe_warmup_epochs=args.probe_warmup_epochs,
                                    difficulty_ramp_epochs=args.difficulty_ramp_epochs,
                                    probe_kind=routing["probe_kind"],
                                    difficulty_mode=routing["difficulty_mode"],
                                    detach_difficulty=routing["detach_difficulty"],
                                    selection_metric=args.selection_metric,
                                    class_balanced_loss=args.class_balanced_loss,
                                    calibrate_probe_temperature=routing["calibrate_probe_temperature"],
                                    calibration_finetune_epochs=args.calibration_finetune_epochs,
                                    device=args.device,
                                    include_diagnostics=True,
                                    experiment_id=f"state_matrix_{routing_profile}",
                                )
                                results.append(result)
                                print(
                                    f"completed protocol={protocol} route={route_id} model={model_id} profile={routing_profile} condition={condition_id} seed={seed} "
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
    output = {
        "script": Path(__file__).name,
        "stage": "daily_affect_state_matrix",
        "run_count": len(results),
        "elapsed_seconds": float(time.time() - started),
        "results": results,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    report_json = args.out_root / "daily_affect_state_matrix_report.json"
    report_md = args.out_root / "daily_affect_state_matrix_report.md"
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
        "# Daily-Affect State Matrix Report",
        "",
        f"- run_count: `{output['run_count']}`",
        "",
        "| protocol | route | model | experiment | normalization | adapter | objective | routing | seed | test QWK | test Macro-F1 | test accuracy | test ordinal MAE | best epoch |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        lines.append(
            f"| {row['protocol']} | {row['route_id']} | {row['model_id']} | {row['experiment_id']} | {row['normalization']} | {row['adapter_mode']} | {row['objective_id']} | {row['routing_id']} | {row['seed']} | "
            f"{fmt(row['test']['qwk'])} | {fmt(row['test']['macro_f1'])} | {fmt(row['test']['accuracy'])} | "
            f"{fmt(row['test']['ordinal_mae'])} | {row['train_audit']['best_epoch']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def make_condition_id(normalization: str, adapter_mode: str, routing_profile: str) -> str:
    return f"routing_{routing_profile}__norm_{normalization}__adapter_{adapter_mode}"


def routing_settings(profile: str, model_id: str, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the P0--P5 routing factorization into an auditable run config."""
    native = {
        "lambda_d": 0.0 if model_id in NO_PRIOR_MODELS or model_id == "prior_uniform" else float(args.lambda_d),
        "beta_ord": float(args.beta_ord),
        "probe_kind": args.probe_kind,
        "difficulty_mode": args.difficulty_mode,
        "detach_difficulty": bool(args.detach_difficulty),
        "calibrate_probe_temperature": bool(args.calibrate_probe_temperature),
    }
    if profile == "native":
        if model_id in NO_PRIOR_MODELS and args.difficulty_mode not in {"auto", "none"}:
            raise ValueError(f"model_id={model_id} does not use prior-guided routing")
        if model_id in PRIOR_GUIDED_MODELS and args.difficulty_mode in {"entropy", "ordinal"}:
            native["lambda_d"] = float(args.lambda_d)
        return native
    if profile == "p0_no_prior":
        if model_id not in NO_PRIOR_MODELS:
            raise ValueError("p0_no_prior requires a no-prior model id")
        return {**native, "lambda_d": 0.0, "difficulty_mode": "none", "calibrate_probe_temperature": False}
    if model_id not in PRIOR_GUIDED_MODELS:
        raise ValueError(f"routing profile {profile} requires a prior-guided model id")
    if profile == "p1_categorical_entropy":
        return {**native, "lambda_d": float(args.lambda_d), "probe_kind": "categorical", "difficulty_mode": "entropy", "detach_difficulty": True, "calibrate_probe_temperature": False}
    if profile == "p2_cumulative_entropy":
        return {**native, "lambda_d": float(args.lambda_d), "probe_kind": "cumulative", "difficulty_mode": "entropy", "detach_difficulty": True, "calibrate_probe_temperature": False}
    if profile == "p3_ordinal_mix":
        return {**native, "lambda_d": float(args.lambda_d), "probe_kind": "cumulative", "difficulty_mode": "ordinal", "detach_difficulty": True, "calibrate_probe_temperature": False}
    if profile == "p4_probe_calibrated":
        return {**native, "lambda_d": float(args.lambda_d), "probe_kind": "cumulative", "difficulty_mode": "ordinal", "detach_difficulty": True, "calibrate_probe_temperature": True}
    if profile == "p5_end_to_end":
        return {**native, "lambda_d": float(args.lambda_d), "probe_kind": "cumulative", "difficulty_mode": "ordinal", "detach_difficulty": False, "calibrate_probe_temperature": False}
    raise ValueError(f"unsupported routing profile: {profile}")


if __name__ == "__main__":
    raise SystemExit(main())
