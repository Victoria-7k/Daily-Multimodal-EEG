#!/usr/bin/env python3
"""Run matched frozen/supervised EEG downstream fusion for ST-11 or MT-11."""

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

from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.split_paths import resolve_protocol_split_root
from daily_multimodal.training.multihead_regression import (
    LABEL_NAMES,
    evaluate_event_level,
    evaluate_scalar_event_level,
    event_level_arrays,
    fit_event_aware_model,
    fit_event_aware_scalar,
    predict,
    predict_scalar,
)


def load_base_module():
    path = Path(__file__).with_name("48_run_fixed_tokens_multilabel_fusion.py")
    spec = importlib.util.spec_from_file_location("_fixed_multilabel_base97", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
install_numpy_core_pickle_aliases()
DEFAULT_ALIGNED_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_ROUTES = {
    "cross_day": "B0_Wphysio_full",
    "within_subject_day": "A1_Wphysio_no_audio",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multitask"), required=True)
    parser.add_argument("--conditions", default="frozen,partial_ft")
    parser.add_argument("--root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--splits-root", type=Path)
    parser.add_argument("--protocols", default="cross_day,within_subject_day")
    parser.add_argument("--labels", default=",".join(LABEL_NAMES))
    parser.add_argument("--seeds", default="240800,240801,240802")
    parser.add_argument("--cross-day-route", default=DEFAULT_ROUTES["cross_day"])
    parser.add_argument("--within-subject-day-route", default=DEFAULT_ROUTES["within_subject_day"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    args.splits_root = args.splits_root or args.root / "outputs/splits"
    protocols = split_csv(args.protocols)
    conditions = split_csv(args.conditions)
    labels = split_csv(args.labels)
    seeds = tuple(int(value) for value in split_csv(args.seeds))
    if sorted(set(conditions) - {"frozen", "partial_ft", "fatigue_partial_ft"}):
        raise ValueError(f"unsupported conditions: {conditions}")
    if sorted(set(labels) - set(LABEL_NAMES)):
        raise ValueError(f"unsupported labels: {labels}")
    routes = {
        "cross_day": args.cross_day_route,
        "within_subject_day": args.within_subject_day_route,
    }
    rows = BASE._load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([BASE._norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    day_id = np.asarray([str(row.get("day_id", "")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    targets = BASE._load_targets(rows)
    fixed_cache: dict[str, dict[str, Any]] = {}
    output_rows = []
    attempted = 0
    for protocol in protocols:
        route = routes[protocol]
        branches = BASE.EXPERIMENT_BRANCHES[route]
        other_names = tuple(branches[1:])
        other_data = BASE._load_all_branches(args.embeddings_root, sample_id, other_names, fixed_cache)
        protocol_root = resolve_protocol_split_root(args.splits_root, protocol)
        split = BASE._load_split(protocol_root, len(rows))
        tasks = [(label,) for label in labels] if args.mode == "single" else [tuple(LABEL_NAMES)]
        for task_labels in tasks:
            task_name = task_labels[0] if args.mode == "single" else "all_11_labels"
            for condition in conditions:
                for seed in seeds:
                    if args.limit_runs and attempted >= args.limit_runs:
                        break
                    attempted += 1
                    run_dir = args.out_root / protocol / task_name / condition / f"seed_{seed}"
                    metrics_path = run_dir / "metrics.json"
                    if not args.force and metrics_path.is_file():
                        old = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if old.get("status") == "ok":
                            old["resumed"] = True
                            output_rows.append(old)
                            print(f"skipped completed protocol={protocol} task={task_name} condition={condition} seed={seed}", flush=True)
                            continue
                    eeg_path = resolve_eeg_path(args.embeddings_root, args.mode, condition, protocol, task_name, seed)
                    tokens, token_mask, branch_report = build_route_tokens(
                        eeg_path, sample_id, other_names, other_data
                    )
                    print(
                        f"starting mode={args.mode} protocol={protocol} route={route} task={task_name} "
                        f"condition={condition} seed={seed}",
                        flush=True,
                    )
                    if args.mode == "single":
                        label_index = LABEL_NAMES.index(task_name)
                        bundle, audit = fit_event_aware_scalar(
                            tokens=tokens,
                            token_mask=token_mask,
                            targets=targets[:, label_index],
                            event_id=event_id,
                            train_idx=split["train"],
                            val_idx=split["val"],
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
                    else:
                        bundle, audit = fit_event_aware_model(
                            tokens=tokens,
                            token_mask=token_mask,
                            targets=targets,
                            event_id=event_id,
                            train_idx=split["train"],
                            val_idx=split["val"],
                            head_variant="H1_shared2_11xhead2",
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
                    result: dict[str, Any] = {
                        "status": "ok",
                        "mode": args.mode,
                        "protocol": protocol,
                        "protocol_split_root": str(protocol_root),
                        "route": route,
                        "task_name": task_name,
                        "label_names": list(task_labels),
                        "condition": condition,
                        "seed": seed,
                        "eeg_token_path": str(eeg_path),
                        "branch_report": branch_report,
                        "train_audit": audit,
                    }
                    event_payload = {}
                    for leaf in ("val", "test"):
                        indices = split[leaf]
                        if args.mode == "single":
                            prediction = predict_scalar(bundle, tokens, token_mask, indices, args.device)
                            true = targets[indices, LABEL_NAMES.index(task_name)].reshape(-1, 1)
                        else:
                            prediction = predict(bundle, tokens, token_mask, indices, args.device)
                            true = targets[indices]
                        arrays = event_level_arrays(
                            true, prediction, event_id[indices], subject_id[indices], day_id[indices]
                        )
                        event_payload[leaf] = arrays
                        if args.mode == "single":
                            result[leaf] = evaluate_scalar_event_level(arrays, float(bundle["y_std"][0, 0]))
                        else:
                            result[leaf] = evaluate_event_level(arrays, bundle["y_std"])
                    run_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        run_dir / "event_predictions.npz",
                        label_names=np.asarray(task_labels),
                        val_event_id=event_payload["val"]["event_id"],
                        val_subject_id=event_payload["val"]["subject_id"],
                        val_day_id=event_payload["val"]["day_id"],
                        val_target=event_payload["val"]["target"],
                        val_prediction=event_payload["val"]["prediction"],
                        test_event_id=event_payload["test"]["event_id"],
                        test_subject_id=event_payload["test"]["subject_id"],
                        test_day_id=event_payload["test"]["day_id"],
                        test_target=event_payload["test"]["target"],
                        test_prediction=event_payload["test"]["prediction"],
                    )
                    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    output_rows.append(result)
                    metric = result["test"] if args.mode == "single" else result["test"]["summary"]
                    print(
                        f"completed protocol={protocol} task={task_name} condition={condition} seed={seed} "
                        f"srmse={metric['standardized_rmse']:.4f} raw_r={metric['raw_r']:.4f}",
                        flush=True,
                    )
                    write_manifest(args, routes, output_rows)
                if args.limit_runs and attempted >= args.limit_runs:
                    break
            if args.limit_runs and attempted >= args.limit_runs:
                break
        if args.limit_runs and attempted >= args.limit_runs:
            break
    write_manifest(args, routes, output_rows)
    print(f"run_count={len(output_rows)}", flush=True)
    return 0


def resolve_eeg_path(root: Path, mode: str, condition: str, protocol: str, task_name: str, seed: int) -> Path:
    if condition == "frozen":
        return root / "eeg/eeg_eegpt_eeg23win_embeddings.npz"
    token_root = root / "eeg_encoder_256d_tokens"
    if condition == "fatigue_partial_ft":
        return token_root / protocol / "eegpt_partial_ft_v1" / f"seed_{seed}.npz"
    if mode == "single":
        return token_root / "single_task" / protocol / task_name / f"seed_{seed}.npz"
    return token_root / "multitask_11label" / protocol / f"seed_{seed}.npz"


def build_route_tokens(
    eeg_path: Path,
    sample_id: np.ndarray,
    other_names: tuple[str, ...],
    other_data: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(eeg_path, allow_pickle=True) as loaded:
        if not np.array_equal(loaded["sample_id"].astype(str), sample_id):
            raise ValueError(f"EEG sample_id mismatch: {eeg_path}")
        eeg = loaded["eeg_emb"].astype(np.float32)
        eeg_mask = loaded["eeg_mask"].astype(bool)
    tokens = np.stack([eeg] + [other_data[name]["embedding"] for name in other_names], axis=1)
    mask = np.stack([eeg_mask] + [other_data[name]["mask"] for name in other_names], axis=1)
    report = {"eeg": {"path": str(eeg_path), "mask_sum": int(eeg_mask.sum())}}
    report.update({name: {"path": other_data[name]["path"], "mask_sum": other_data[name]["mask_sum"]} for name in other_names})
    return tokens.astype(np.float32), mask.astype(bool), report


def write_manifest(args: Any, routes: dict[str, str], rows: list[dict[str, Any]]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "supervised_eeg_downstream_fusion",
        "mode": args.mode,
        "conditions": list(split_csv(args.conditions)),
        "protocols": list(split_csv(args.protocols)),
        "labels": list(split_csv(args.labels)),
        "seeds": [int(value) for value in split_csv(args.seeds)],
        "routes": routes,
        "run_count": len(rows),
        "results": rows,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
