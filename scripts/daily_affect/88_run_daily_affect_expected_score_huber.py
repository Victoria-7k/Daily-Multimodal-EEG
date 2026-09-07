#!/usr/bin/env python3
"""Validation-locked expected-score Huber screen for the matched daily-affect bridge.

The screen changes only the event-level expected-score Huber weight on top of
the existing class-weighted CE, cumulative ordinal, and ranking objective.
Candidate selection uses validation metrics only.  Test rows are emitted only
for the validation-locked candidate versus the zero-weight control, and the
three-seed result remains exploratory rather than a promotion decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.daily_affect.training import load_bag_dataset, run_daily_affect_run


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_BRIDGE_ROOT = DEFAULT_ROOT / "outputs/daily_affect_window_event_bridge_20260906"
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "outputs/daily_affect_expected_score_huber_20260907"
DEFAULT_ROUTE_ID = "A1_Wphysio_full__eeg_eegpt_partial_ft_v1"
DEFAULT_SEEDS = (240729, 240730, 240731)
METRICS = ("qwk", "expected_raw_r", "expected_within_subject_centered_r", "expected_rmse")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BRIDGE_ROOT / "bags")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default=DEFAULT_ROUTE_ID)
    parser.add_argument("--normalization", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--score-huber-weights", default="0,0.025,0.05,0.1,0.2")
    parser.add_argument("--score-huber-delta", type=float, default=1.0)
    parser.add_argument("--qwk-tolerance", type=float, default=0.02)
    parser.add_argument("--min-raw-r-wins", type=int, default=2)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not float(args.score_huber_delta) > 0.0:
        raise ValueError("--score-huber-delta must be positive")
    if not float(args.qwk_tolerance) >= 0.0:
        raise ValueError("--qwk-tolerance must be nonnegative")
    weights = parse_float_csv(args.score_huber_weights)
    if 0.0 not in weights:
        raise ValueError("--score-huber-weights must include the zero-weight control")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("--score-huber-weights must be nonnegative")
    seeds = parse_int_csv(args.seeds)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    torch.set_num_threads(max(1, int(args.torch_threads)))

    condition_id = f"norm_{args.normalization}__adapter_{args.adapter_mode}"
    started = time.time()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        bag_path = args.bags_root / args.protocol / args.route_id / f"seed_{seed}" / "ema_bags.npz"
        dataset = load_bag_dataset(bag_path)
        assert_dataset_contract(dataset, args.protocol, args.route_id)
        for weight in weights:
            run_dir = (
                args.out_root
                / "runs"
                / args.protocol
                / args.route_id
                / "bag_static"
                / condition_id
                / score_condition_id(weight, args.score_huber_delta)
                / f"seed_{seed}"
            )
            metrics_path = run_dir / "metrics.json"
            if args.skip_existing and metrics_path.is_file():
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                print(f"skipped seed={seed} score_huber_weight={weight:g}", flush=True)
            else:
                print(f"starting seed={seed} score_huber_weight={weight:g}", flush=True)
                result = run_daily_affect_run(
                    dataset=dataset,
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
                    beta_ord=0.25,
                    probe_loss_weight=0.0,
                    ordinal_loss_weight=args.ordinal_loss_weight,
                    rank_loss_weight=args.rank_loss_weight,
                    expected_score_loss_weight=weight,
                    expected_score_huber_delta=args.score_huber_delta,
                    modality_dropout_prob=args.modality_dropout_prob,
                    probe_warmup_epochs=0,
                    difficulty_ramp_epochs=0,
                    probe_kind="cumulative",
                    difficulty_mode="none",
                    detach_difficulty=True,
                    selection_metric=args.selection_metric,
                    class_balanced_loss=True,
                    calibrate_probe_temperature=False,
                    device=args.device,
                    include_diagnostics=False,
                    experiment_id="expected_score_huber_validation_locked_v1",
                )
            runs.append(run_row(result, weight, args.score_huber_delta, dataset))
            print(
                f"completed seed={seed} score_huber_weight={weight:g} "
                f"val_raw_r={fmt(result['val'].get('expected_raw_r'))} "
                f"test_raw_r={fmt(result['test'].get('expected_raw_r'))}",
                flush=True,
            )

    validation = validation_summary(
        runs,
        weights=weights,
        qwk_tolerance=args.qwk_tolerance,
        min_raw_r_wins=args.min_raw_r_wins,
    )
    selected_weight = choose_weight(validation)
    test_comparison = selected_test_comparison(runs, selected_weight)
    output = {
        "script": Path(__file__).name,
        "status": "exploratory_validation_locked_three_seed_screen",
        "promotion_statement": (
            "The Huber weight is selected solely from validation metrics. The selected candidate's three-seed "
            "test comparison is an exploratory gate and cannot replace the five-to-seven-seed promotion requirement."
        ),
        "contract": {
            "protocol": args.protocol,
            "route_id": args.route_id,
            "model_id": "bag_static",
            "normalization": args.normalization,
            "adapter_mode": args.adapter_mode,
            "bag_shape": "(N_event,23,4,256)",
            "baseline_objective": "weighted CE + 0.5 cumulative ordinal + 0.1 within-subject ranking",
            "candidate_change": "add lambda * event-level expected-score Huber loss",
            "checkpoint_selection": args.selection_metric,
            "weight_selection": "validation expected raw r with validation QWK non-inferiority constraint",
            "test_selection_rule": "test metrics are not used to select Huber weight",
        },
        "gate": {
            "validation_raw_r_win_requirement": f"at least {args.min_raw_r_wins}/{len(seeds)} seeds",
            "validation_mean_qwk_delta_minimum": -float(args.qwk_tolerance),
            "stop_condition": "no nonzero weight passes the validation gate",
        },
        "runs": runs,
        "validation_summary": validation,
        "selected_score_huber_weight": selected_weight,
        "selected_test_comparison": test_comparison,
        "elapsed_seconds": float(time.time() - started),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "expected_score_huber_screen.json", output)
    write_csv(args.out_root / "expected_score_huber_runs.csv", runs)
    write_markdown(args.out_root / "expected_score_huber_screen.md", output)
    print(f"selected_score_huber_weight={selected_weight}")
    print(f"screen_json={args.out_root / 'expected_score_huber_screen.json'}")
    return 0


def assert_dataset_contract(dataset: Any, protocol: str, route_id: str) -> None:
    if protocol not in dataset.bag_path.parts or dataset.route_id != route_id:
        raise ValueError(
            f"bag contract mismatch: expected path protocol/route {protocol}/{route_id}, "
            f"got {dataset.bag_path}/{dataset.route_id}"
        )
    if tuple(dataset.tokens.shape[1:]) != (23, 4, 256):
        raise ValueError(f"expected canonical EMA bags (N,23,4,256), got {tuple(dataset.tokens.shape)}")


def run_row(result: dict[str, Any], weight: float, delta: float, dataset: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": int(result["seed"]),
        "score_huber_weight": float(weight),
        "score_huber_delta": float(delta),
        "objective_id": result["objective_id"],
        "best_epoch": int(result["train_audit"]["best_epoch"]),
    }
    for split in ("val", "test"):
        for metric in METRICS:
            row[f"{split}_{metric}"] = result[split].get(metric)
    with np.load(result["prediction_path"], allow_pickle=False) as values:
        test_index = values["test_index"].astype(np.int64)
        score = values["test_expected_score"].astype(np.float64)
    if not np.array_equal(test_index, dataset.test_index):
        raise ValueError(f"{result['prediction_path']} does not preserve the bag test index")
    row.update(range_diagnostics(score, dataset.label[test_index].astype(np.float64)))
    return row


def range_diagnostics(score: np.ndarray, label: np.ndarray) -> dict[str, float]:
    if score.size == 0 or score.shape != label.shape:
        raise ValueError("range diagnostics require aligned nonempty score and label vectors")
    unique = np.unique(label)
    group_means = np.asarray([score[label == value].mean() for value in unique], dtype=np.float64)
    residuals = np.concatenate([score[label == value] - score[label == value].mean() for value in unique])
    between_variance = float(np.sum([
        np.sum(label == value) * (score[label == value].mean() - score.mean()) ** 2 for value in unique
    ]) / score.size)
    total_variance = float(np.var(score))
    return {
        "test_prediction_std": float(np.std(score)),
        "test_prediction_p10_p90_span": float(np.quantile(score, 0.90) - np.quantile(score, 0.10)),
        "test_label_mean_span": float(group_means.max() - group_means.min()),
        "test_within_label_std": float(np.std(residuals)),
        "test_label_explained_variance_fraction": float(between_variance / total_variance) if total_variance > 0.0 else 0.0,
    }


def validation_summary(
    rows: list[dict[str, Any]],
    *,
    weights: tuple[float, ...],
    qwk_tolerance: float,
    min_raw_r_wins: int,
) -> list[dict[str, Any]]:
    by_weight = {weight: sorted((row for row in rows if math.isclose(row["score_huber_weight"], weight)), key=lambda row: row["seed"]) for weight in weights}
    baseline = by_weight[0.0]
    if not baseline:
        raise ValueError("zero-weight control produced no rows")
    baseline_by_seed = {int(row["seed"]): row for row in baseline}
    output: list[dict[str, Any]] = []
    for weight in weights:
        group = by_weight[weight]
        if {int(row["seed"]) for row in group} != set(baseline_by_seed):
            raise ValueError(f"weight {weight:g} is not matched to all zero-weight seeds")
        qwk_delta = np.asarray([float(row["val_qwk"]) - float(baseline_by_seed[int(row["seed"])]["val_qwk"]) for row in group])
        raw_r_delta = np.asarray([
            float(row["val_expected_raw_r"]) - float(baseline_by_seed[int(row["seed"])]["val_expected_raw_r"])
            for row in group
        ])
        summary = {
            "score_huber_weight": float(weight),
            "seed_count": len(group),
            "val_qwk_mean": float(np.mean([float(row["val_qwk"]) for row in group])),
            "val_expected_raw_r_mean": float(np.mean([float(row["val_expected_raw_r"]) for row in group])),
            "mean_delta_val_qwk": float(qwk_delta.mean()),
            "std_delta_val_qwk": float(qwk_delta.std(ddof=1)) if len(qwk_delta) > 1 else 0.0,
            "mean_delta_val_expected_raw_r": float(raw_r_delta.mean()),
            "std_delta_val_expected_raw_r": float(raw_r_delta.std(ddof=1)) if len(raw_r_delta) > 1 else 0.0,
            "val_expected_raw_r_wins": int(np.sum(raw_r_delta > 0.0)),
            "validation_gate_pass": bool(
                weight > 0.0
                and int(np.sum(raw_r_delta > 0.0)) >= int(min_raw_r_wins)
                and float(qwk_delta.mean()) >= -float(qwk_tolerance)
            ),
        }
        output.append(summary)
    return output


def choose_weight(validation: list[dict[str, Any]]) -> float:
    passing = [row for row in validation if row["validation_gate_pass"]]
    if not passing:
        return 0.0
    winner = max(
        passing,
        key=lambda row: (
            float(row["mean_delta_val_expected_raw_r"]),
            float(row["mean_delta_val_qwk"]),
            -float(row["score_huber_weight"]),
        ),
    )
    return float(winner["score_huber_weight"])


def selected_test_comparison(rows: list[dict[str, Any]], selected_weight: float) -> dict[str, Any]:
    baseline = {int(row["seed"]): row for row in rows if math.isclose(row["score_huber_weight"], 0.0)}
    candidate = {int(row["seed"]): row for row in rows if math.isclose(row["score_huber_weight"], selected_weight)}
    if not baseline or set(candidate) != set(baseline):
        raise ValueError("selected candidate is not matched to the zero-weight control")
    result: dict[str, Any] = {"selected_weight": float(selected_weight), "per_seed": []}
    for seed in sorted(baseline):
        row = {"seed": int(seed)}
        for metric in METRICS:
            row[f"delta_test_{metric}"] = float(candidate[seed][f"test_{metric}"]) - float(baseline[seed][f"test_{metric}"])
        result["per_seed"].append(row)
    for metric in METRICS:
        values = np.asarray([row[f"delta_test_{metric}"] for row in result["per_seed"]], dtype=np.float64)
        result[f"mean_delta_test_{metric}"] = float(values.mean())
        result[f"std_delta_test_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"test_win_count_{metric}"] = int(np.sum(values > 0.0)) if metric != "expected_rmse" else int(np.sum(values < 0.0))
    return result


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def score_condition_id(weight: float, delta: float) -> str:
    return f"score_huber_{float(weight):g}__delta_{float(delta):g}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, output: dict[str, Any]) -> None:
    validation = output["validation_summary"]
    selected = float(output["selected_score_huber_weight"])
    lines = [
        "# Daily-Affect Expected-Score Huber Screen",
        "",
        f"- status: `{output['status']}`",
        f"- selected Huber weight from validation only: `{selected:g}`",
        f"- promotion: {output['promotion_statement']}",
        "",
        "## Validation Selection",
        "",
        "| Huber weight | val QWK delta | val raw-r delta | raw-r wins | gate |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in validation:
        lines.append(
            f"| {row['score_huber_weight']:g} | {row['mean_delta_val_qwk']:+.4f} +/- {row['std_delta_val_qwk']:.4f} | "
            f"{row['mean_delta_val_expected_raw_r']:+.4f} +/- {row['std_delta_val_expected_raw_r']:.4f} | "
            f"{row['val_expected_raw_r_wins']}/{row['seed_count']} | {'pass' if row['validation_gate_pass'] else 'fail'} |"
        )
    comparison = output["selected_test_comparison"]
    lines.extend(
        [
            "",
            "## Locked Test Comparison",
            "",
            "| metric | selected minus zero-weight control | wins |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        lines.append(
            f"| {metric} | {comparison[f'mean_delta_test_{metric}']:+.4f} +/- {comparison[f'std_delta_test_{metric}']:.4f} | "
            f"{comparison[f'test_win_count_{metric}']}/{len(comparison['per_seed'])} |"
        )
    selected_rows = [row for row in output["runs"] if math.isclose(row["score_huber_weight"], selected)]
    baseline_rows = [row for row in output["runs"] if math.isclose(row["score_huber_weight"], 0.0)]
    lines.extend(
        [
            "",
            "## Test Range Diagnostics",
            "",
            "| quantity | zero-weight | locked candidate |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in (
        "test_prediction_std",
        "test_prediction_p10_p90_span",
        "test_label_mean_span",
        "test_within_label_std",
        "test_label_explained_variance_fraction",
    ):
        lines.append(
            f"| {metric} | {np.mean([float(row[metric]) for row in baseline_rows]):.4f} | "
            f"{np.mean([float(row[metric]) for row in selected_rows]):.4f} |"
        )
    if selected == 0.0:
        lines.extend(["", "No nonzero Huber weight passed the validation gate; the route stops at the zero-weight control."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
