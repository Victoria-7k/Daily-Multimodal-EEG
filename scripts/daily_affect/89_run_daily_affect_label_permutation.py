#!/usr/bin/env python3
"""Leakage-safe EMA-event label-permutation controls for Daily-affect.

It permutes train and validation labels independently; test labels, test events,
EMA-window membership, token arrays, masks, and splits stay unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.daily_affect.training import DailyAffectBagDataset, load_bag_dataset, run_daily_affect_run

ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_BAGS = ROOT / "outputs/daily_affect_ordinal_20260903/bags"
DEFAULT_CLEAN = ROOT / "outputs/daily_affect_dynamic_a1_crossday_routefix_20260906"
DEFAULT_OUT = ROOT / "outputs/daily_affect_label_permutation_20260907"
METRICS = ("qwk", "macro_f1", "expected_raw_r", "expected_within_subject_centered_r", "ordinal_mae", "expected_rmse")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("global_smoke", "within_subject"), default="global_smoke")
    parser.add_argument("--bags-root", type=Path, default=DEFAULT_BAGS)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", default="cross_day")
    parser.add_argument("--route-id", default="A1_Wphysio_full")
    parser.add_argument("--model-id", default="dynamic_kernel_prior_uniform")
    parser.add_argument("--normalization", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--adapter-mode", choices=("shared", "per_modality"), default="per_modality")
    parser.add_argument("--seeds", default="240729")
    parser.add_argument("--permutation-ids", default="0,1,2,3,4")
    parser.add_argument("--permutation-seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    seeds, permutation_ids = parse_ids(args.seeds), parse_ids(args.permutation_ids)
    if args.phase == "global_smoke" and (len(seeds), len(permutation_ids)) != (1, 5):
        raise ValueError("global_smoke requires exactly 5 permutation IDs and 1 model seed")
    if args.phase == "within_subject" and (len(seeds) != 3 or len(permutation_ids) < 30):
        raise ValueError("within_subject requires 3 matched seeds and at least 30 permutation IDs")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but it is unavailable")
    torch.set_num_threads(max(1, args.torch_threads))
    mode = "global_shuffle" if args.phase == "global_smoke" else "within_subject_shuffle"
    clean = {seed: load_clean(args, seed) for seed in seeds}
    started, audits, rows = time.time(), [], []
    for permutation_id in permutation_ids:
        for seed in seeds:
            source = load_bag_dataset(args.bags_root / args.protocol / args.route_id / f"seed_{seed}" / "ema_bags.npz")
            assert_contract(source, args.protocol, args.route_id)
            permuted, audit = permuted_dataset(source, mode, permutation_seed(args.permutation_seed, permutation_id), permutation_id)
            assert_audit(audit, mode)
            audits.append({"model_seed": seed, **audit})
            run_dir = args.out_root / "runs" / mode / f"permutation_{permutation_id:03d}" / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            if args.skip_existing and metrics_path.is_file():
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                print(f"skipped permutation={permutation_id} seed={seed}", flush=True)
            else:
                print(f"starting permutation={permutation_id} seed={seed}", flush=True)
                result = run_daily_affect_run(
                    dataset=permuted, protocol=args.protocol, model_id=args.model_id,
                    normalization=args.normalization, adapter_mode=args.adapter_mode, seed=seed,
                    run_dir=run_dir, epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    dropout=args.dropout, hidden_dim=args.hidden_dim, patience=args.patience,
                    lambda_d=0.0, beta_ord=0.25, probe_loss_weight=0.1,
                    ordinal_loss_weight=0.5, rank_loss_weight=0.1, modality_dropout_prob=0.1,
                    probe_warmup_epochs=5, difficulty_ramp_epochs=5, probe_kind="cumulative",
                    difficulty_mode="none", detach_difficulty=True, selection_metric="qwk",
                    class_balanced_loss=True, calibrate_probe_temperature=False, device=args.device,
                    include_diagnostics=False, experiment_id="daily_affect_label_permutation_20260907",
                )
                write_metadata(run_dir, audit, mode, permutation_id)
            rows.append(metric_row(result, clean[seed], mode, permutation_id, audit))
            print(f"completed permutation={permutation_id} seed={seed} qwk={fmt(result['test'].get('qwk'))} raw_r={fmt(result['test'].get('expected_raw_r'))}", flush=True)
    report = summarize(args, mode, clean, rows, audits, time.time() - started)
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "manifest.json", {"script": Path(__file__).name, "phase": args.phase, "mode": mode, "seeds": seeds, "permutation_ids": permutation_ids, "report": report})
    write_csv(args.out_root / "permutation_audit.csv", audits)
    write_csv(args.out_root / "run_metrics.csv", rows)
    write_json(args.out_root / "phase_report.json", report)
    write_report(args.out_root / "label_permutation_report.md", report)
    print(f"status={report['status']}")
    return 0


def permuted_dataset(dataset: DailyAffectBagDataset, mode: str, seed: int, permutation_id: int) -> tuple[DailyAffectBagDataset, dict[str, Any]]:
    label, zero = dataset.label.copy(), dataset.label_zero_based.copy()
    split_audits, singleton_subjects = [], set()
    for offset, (name, indices) in enumerate((("train", dataset.train_index), ("val", dataset.val_index))):
        before = zero[indices].copy()
        after, singleton = permute(before, dataset.subject_id[indices], mode, seed + offset)
        zero[indices], label[indices] = after, after.astype(np.float32) + 1.0
        singleton_subjects.update(singleton)
        split_audits.append(split_audit(name, before, after, dataset.subject_id[indices], singleton))
    supervised = np.concatenate((dataset.train_index, dataset.val_index))
    audit = {
        "permutation_id": permutation_id, "permutation_seed": seed, "mode": mode,
        "train_val_label_value_changed_fraction": float(np.mean(zero[supervised] != dataset.label_zero_based[supervised])),
        "singleton_subject_count": len(singleton_subjects),
        "permuted_subject_count": len(np.unique(dataset.subject_id[supervised].astype(str))),
        "permuted_event_count": len(supervised),
        "test_labels_unchanged": bool(np.array_equal(zero[dataset.test_index], dataset.label_zero_based[dataset.test_index]) and np.array_equal(label[dataset.test_index], dataset.label[dataset.test_index])),
        "split_membership_unchanged": True,
        "event_23_window_membership_unchanged": bool(dataset.sample_id_matrix.shape == (dataset.row_count, 23)),
        "split_audits": split_audits,
    }
    return replace(dataset, label=label, label_zero_based=zero), audit


def permute(labels: np.ndarray, subjects: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, set[str]]:
    rng, result, singleton = np.random.default_rng(seed), labels.copy(), set()
    if mode == "global_shuffle":
        return result[rng.permutation(result.size)], singleton
    for subject in np.unique(subjects.astype(str)):
        positions = np.flatnonzero(subjects.astype(str) == subject)
        if positions.size < 2:
            singleton.add(str(subject))
        else:
            result[positions] = result[positions][rng.permutation(positions.size)]
    return result, singleton


def split_audit(name: str, before: np.ndarray, after: np.ndarray, subjects: np.ndarray, singleton: set[str]) -> dict[str, Any]:
    return {"split": name, "event_count": int(before.size), "subject_count": int(np.unique(subjects.astype(str)).size), "singleton_subject_count": len(singleton), "label_value_changed_fraction": float(np.mean(before != after)), "histogram_before": histogram(before), "histogram_after": histogram(after), "histogram_preserved": histogram(before) == histogram(after)}


def assert_contract(dataset: DailyAffectBagDataset, protocol: str, route_id: str) -> None:
    if dataset.route_id != route_id or protocol not in dataset.bag_path.parts:
        raise ValueError(f"bag contract mismatch: {dataset.bag_path} / {dataset.route_id}")
    if tuple(dataset.tokens.shape[1:]) != (23, 4, 256) or dataset.sample_id_matrix.shape != (dataset.row_count, 23):
        raise ValueError("invalid canonical EMA-bag shape")
    if dataset.supervision_boundary != "label_free_or_fixed_embeddings_only":
        raise ValueError(f"primary null requires label-free/fixed tokens, got {dataset.supervision_boundary}")


def assert_audit(audit: dict[str, Any], mode: str) -> None:
    if not all((audit["test_labels_unchanged"], audit["split_membership_unchanged"], audit["event_23_window_membership_unchanged"])):
        raise ValueError(f"immutable boundary audit failed: {audit}")
    if not all(row["histogram_preserved"] for row in audit["split_audits"]):
        raise ValueError(f"class histogram audit failed: {audit}")
    threshold = 0.5 if mode == "global_shuffle" else 0.3
    if audit["train_val_label_value_changed_fraction"] < threshold:
        raise ValueError(f"label-change fraction below {threshold}: {audit}")


def load_clean(args: Any, seed: int) -> dict[str, Any]:
    condition = f"norm_{args.normalization}__adapter_{args.adapter_mode}"
    candidates = [
        args.clean_root / condition / "runs" / args.protocol / args.route_id / args.model_id / condition / f"seed_{seed}" / "metrics.json",
        args.clean_root / "runs" / args.protocol / args.route_id / args.model_id / condition / f"seed_{seed}" / "metrics.json",
    ]
    if args.model_id == "bag_static":
        candidates.append(args.clean_root / "phase0" / args.protocol / args.route_id / condition / f"seed_{seed}" / "metrics.json")
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one matched clean result, found {matches}; candidates={candidates}")
    path = matches[0]
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("protocol"), result.get("route_id"), result.get("model_id"), result.get("seed")) != (args.protocol, args.route_id, args.model_id, seed):
        raise ValueError(f"clean-result contract mismatch: {path}")
    return result


def write_metadata(run_dir: Path, audit: dict[str, Any], mode: str, permutation_id: int) -> None:
    for filename in ("metrics.json", "config.json"):
        path = run_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update({"permutation": audit, "permutation_mode": mode, "permutation_id": permutation_id, "test_evaluation_labels": "original_true_labels"})
        write_json(path, payload)


def metric_row(result: dict[str, Any], clean: dict[str, Any], mode: str, permutation_id: int, audit: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"mode": mode, "permutation_id": permutation_id, "seed": result["seed"], "best_epoch": result["train_audit"]["best_epoch"], "label_change_fraction": audit["train_val_label_value_changed_fraction"]}
    for metric in METRICS:
        row[f"clean_{metric}"] = clean["test"].get(metric)
        row[f"shuffled_{metric}"] = result["test"].get(metric)
        row[f"clean_minus_shuffled_{metric}"] = difference(row[f"clean_{metric}"], row[f"shuffled_{metric}"])
    return row


def summarize(args: Any, mode: str, clean: dict[int, dict[str, Any]], rows: list[dict[str, Any]], audits: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    clean_mean = {metric: mean([item["test"].get(metric) for item in clean.values()]) for metric in METRICS}
    null = []
    for permutation_id in sorted({row["permutation_id"] for row in rows}):
        group = [row for row in rows if row["permutation_id"] == permutation_id]
        null.append({"permutation_id": permutation_id, **{metric: mean([row[f"shuffled_{metric}"] for row in group]) for metric in METRICS}})
    if mode == "global_shuffle":
        qwk_wins = sum(float(row["clean_minus_shuffled_qwk"]) > 0.0 for row in rows)
        raw_wins = sum(float(row["clean_minus_shuffled_expected_raw_r"]) > 0.0 for row in rows)
        centered_positive = sum(float(row["shuffled_expected_within_subject_centered_r"] or 0.0) > 0.0 for row in rows)
        gate = {"all_audits_pass": len(audits) == 5, "qwk_clean_wins": f"{qwk_wins}/5", "raw_r_clean_wins": f"{raw_wins}/5", "centered_r_positive_null_runs": f"{centered_positive}/5", "pass": qwk_wins >= 4 and raw_wins >= 4 and centered_positive < 4}
        status = "phase1_pass_expand_to_within_subject" if gate["pass"] else "stop_and_audit"
    else:
        gate = {"permutation_count": len(null), "empirical_p": {metric: empirical_p(clean_mean[metric], [row[metric] for row in null], metric not in {"ordinal_mae", "expected_rmse"}) for metric in METRICS}}
        status = "phase2_completed_interpret_by_metric"
    return {"status": status, "phase": args.phase, "mode": mode, "contract": {"protocol": args.protocol, "route_id": args.route_id, "model_id": args.model_id, "normalization": args.normalization, "adapter_mode": args.adapter_mode, "supervision_boundary": "label_free_or_fixed_embeddings_only", "test_labels": "untouched true labels"}, "actual_permutation_count": len(null), "actual_run_count": len(rows), "clean_test_mean": clean_mean, "gate": gate, "permutation_null": null, "elapsed_seconds": elapsed}


def empirical_p(clean: float | None, null: list[float | None], higher: bool) -> float | None:
    values = [float(value) for value in null if value is not None and math.isfinite(float(value))]
    if clean is None or not values:
        return None
    return (1 + sum(value >= clean for value in values) if higher else 1 + sum(value <= clean for value in values)) / (len(values) + 1)


def permutation_seed(base: int, permutation_id: int) -> int:
    return int(np.random.SeedSequence([base, permutation_id]).generate_state(1, dtype=np.uint32)[0])


def parse_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("IDs must be nonempty and unique")
    return ids


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(index + 1): int(count) for index, count in enumerate(np.bincount(values.astype(np.int64), minlength=5))}


def difference(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def mean(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows]
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Daily-Affect Label-Permutation Report", "", f"- status: `{report['status']}`", f"- phase: `{report['phase']}`", f"- actual permutations: `{report['actual_permutation_count']}`", f"- actual runs: `{report['actual_run_count']}`", f"- supervision boundary: `{report['contract']['supervision_boundary']}`", f"- test labels: `{report['contract']['test_labels']}`", "", "## Clean Test Mean", "", "| metric | value |", "| --- | ---: |"]
    lines.extend(f"| {metric} | {fmt(value)} |" for metric, value in report["clean_test_mean"].items())
    lines.extend(["", "## Gate", "", "```json", json.dumps(report["gate"], ensure_ascii=False, indent=2), "```", ""])
    if report["mode"] == "global_shuffle":
        lines.append("Expand to the 30×3 within-subject null only when this gate passes; otherwise stop for the planned leakage and supervision audit.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
