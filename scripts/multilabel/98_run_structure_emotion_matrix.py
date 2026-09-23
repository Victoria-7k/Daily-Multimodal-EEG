#!/usr/bin/env python3
"""Run 0814/0906 structures with one 11-label-supervised EEGPT input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.daily_affect.ema_bags import build_daily_affect_bags, load_jsonl
from daily_multimodal.daily_affect.training import load_bag_dataset
from daily_multimodal.training.structure_emotion import (
    EEG_BRANCH, EMBEDDING_SEED, ROUTE_ID, audit_bag, audit_multitask_eeg_token,
    conditions, event_targets, run_condition,
)


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"))
    parser.add_argument("--splits-root", type=Path, help="Root containing <protocol> split directories; defaults to <root>/outputs/splits.")
    parser.add_argument("--embeddings-root", type=Path, default=Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/multiemotion_20260913/structure_matrix_A1"))
    parser.add_argument("--protocols", default="cross_day,date_in_order")
    parser.add_argument("--seeds", default="240800,240801,240802", help="Downstream regression seeds; EEG embedding remains seed_240800.")
    parser.add_argument("--conditions", help="Comma-separated structure condition ids; default is all 19.")
    parser.add_argument("--stage", choices=("preflight", "matrix"), default="matrix")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    protocols = split_csv(args.protocols)
    if not protocols or set(protocols) - {"cross_day", "within_subject_day", "date_in_order"}:
        raise ValueError("unsupported structure-matrix protocol")
    seeds = tuple(int(item) for item in split_csv(args.seeds))
    selected = split_csv(args.conditions) if args.conditions else tuple(conditions())
    unknown = set(selected) - set(conditions())
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if not selected or not seeds:
        raise ValueError("nonempty condition and downstream seed lists are required")
    index_path = args.root / "index/eeg_aligned_window_index.jsonl"
    splits_root = args.splits_root or args.root / "outputs/splits"
    index_rows = load_jsonl(index_path)
    args.out_root.mkdir(parents=True, exist_ok=True)
    audits = []
    attempted = 0
    for protocol in protocols:
        bag_dir = args.out_root / "bags" / protocol / ROUTE_ID / f"seed_{EMBEDDING_SEED}"
        bag_path = bag_dir / "ema_bags.npz"
        if not bag_path.is_file():
            build_daily_affect_bags(
                index_path=index_path, splits_root=splits_root,
                embeddings_root=args.embeddings_root, protocol=protocol,
                route_id="A1_Wphysio_no_audio", out_dir=bag_dir, target_label="fatigue",
                eeg_branch=EEG_BRANCH, eeg_seed=EMBEDDING_SEED,
            )
        dataset = load_bag_dataset(bag_path)
        targets = event_targets(dataset, index_rows)
        audit = audit_bag(dataset, targets, protocol)
        audit.update(audit_multitask_eeg_token(
            args.embeddings_root / "eeg_encoder_256d_tokens" / "multitask_11label" / protocol / f"seed_{EMBEDDING_SEED}.npz",
            index_rows, splits_root, protocol,
        ))
        audits.append(audit)
        print(f"preflight passed protocol={protocol} events={dataset.row_count} route={ROUTE_ID}", flush=True)
        preflight_path = args.out_root / "preflight.json"
        existing = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() and args.stage == "matrix" else []
        merged = {item["protocol"]: item for item in existing}
        merged.update({item["protocol"]: item for item in audits})
        preflight_path.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2), encoding="utf-8")
        if args.stage == "preflight":
            continue
        for condition_id in selected:
            model_id, temporal_policy = conditions()[condition_id]
            for seed in seeds:
                run_dir = args.out_root / "runs" / protocol / condition_id / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                predictions_path = run_dir / "event_predictions.npz"
                if args.skip_existing and metrics_path.is_file() and predictions_path.is_file():
                    existing = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "ok" and existing.get("embedding_seed") == EMBEDDING_SEED:
                        print(f"skipped completed protocol={protocol} condition={condition_id} seed={seed}", flush=True)
                        continue
                print(f"starting protocol={protocol} condition={condition_id} downstream_seed={seed} embedding_seed={EMBEDDING_SEED}", flush=True)
                result = run_condition(
                    dataset=dataset, targets=targets, protocol=protocol, condition_id=condition_id,
                    model_id=model_id, temporal_policy=temporal_policy, seed=seed, out_dir=run_dir,
                    epochs=args.epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay, dropout=args.dropout,
                    patience=args.patience, modality_dropout_prob=args.modality_dropout_prob, device=args.device,
                )
                attempted += 1
                print(
                    f"completed protocol={protocol} condition={condition_id} seed={seed} "
                    f"val_srmse={result['val']['summary']['standardized_rmse']:.4f} "
                    f"test_raw_r={result['test']['summary']['raw_r']:.4f} "
                    f"seconds={result['duration_seconds']:.1f}", flush=True,
                )
                if args.limit_runs and attempted >= args.limit_runs:
                    print(f"run_count={attempted} limit_reached=true", flush=True)
                    return 0
    print(f"run_count={attempted} stage={args.stage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
