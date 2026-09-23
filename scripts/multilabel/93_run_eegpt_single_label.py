#!/usr/bin/env python3
"""Run resumable event-supervised single-label EEGPT partial fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.training.eeg_encoder_matrix import load_eeg_aligned_dataset, load_split_protocols
from daily_multimodal.training.eeg_multitask import (
    EEGSupervisedRuntime,
    LABEL_NAMES,
    load_event_metadata,
    prepare_protocol_input_cache,
    run_eeg_supervised,
)


DEFAULT_ALIGNED_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_DATA_ROOT = Path("/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg_encoder_256d_tokens")


def main(default_mode: str = "single") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "multitask"), default=default_mode)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--splits-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--protocols", default="cross_day,date_in_order")
    parser.add_argument("--labels", default=",".join(LABEL_NAMES))
    parser.add_argument("--seeds", default="240800,240801,240802")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fallback-batch-size", type=int, default=64)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--partial-last-n-blocks", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=0)
    args = parser.parse_args()

    args.index_path = args.index_path or args.aligned_root / "index/eeg_aligned_window_index.jsonl"
    args.splits_root = args.splits_root or args.aligned_root / "outputs/splits"
    args.checkpoint = args.checkpoint or args.aligned_root / "outputs/checkpoints/eegpt-pretrained"
    default_stage = "phase3_single_task_eeg" if args.mode == "single" else "phase4_multitask_eeg"
    args.out_root = args.out_root or args.aligned_root / "outputs/multiemotion_20260913" / default_stage
    protocols = split_csv(args.protocols)
    labels = split_csv(args.labels)
    unknown = sorted(set(labels) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    if args.mode == "multitask" and tuple(labels) != LABEL_NAMES:
        raise ValueError("multitask mode requires the canonical 11 labels in canonical order")
    seeds = tuple(int(value) for value in split_csv(args.seeds))
    runtime = EEGSupervisedRuntime(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        fallback_batch_size=args.fallback_batch_size,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        grad_clip=args.grad_clip,
        partial_last_n_blocks=args.partial_last_n_blocks,
        device=args.device,
        amp=not args.disable_amp,
        torch_threads=args.torch_threads,
    )
    dataset = load_eeg_aligned_dataset(data_root=args.data_root, index_path=args.index_path)
    metadata = load_event_metadata(args.index_path, dataset.row_count)
    if not np.array_equal(metadata["sample_id"].astype(str), dataset.sample_id.astype(str)):
        raise ValueError("index sample_id order differs from loaded EEG dataset")
    splits = load_split_protocols(args.splits_root, protocols, row_count=dataset.row_count)
    task_labels = [(label,) for label in labels] if args.mode == "single" else [tuple(labels)]
    manifest_rows: list[dict[str, Any]] = []
    attempted = 0
    for protocol in protocols:
        print(f"caching normalized EEG protocol={protocol} device={args.device}", flush=True)
        input_cache = prepare_protocol_input_cache(dataset, splits[protocol], device=args.device)
        print(
            f"cached normalized EEG protocol={protocol} shape={tuple(input_cache['tensor'].shape)}",
            flush=True,
        )
        for current_labels in task_labels:
            task_name = current_labels[0] if args.mode == "single" else "all_11_labels"
            for seed in seeds:
                if args.limit_runs and attempted >= args.limit_runs:
                    break
                attempted += 1
                run_dir = args.out_root / protocol / task_name / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                token_path = token_output_path(args.embeddings_root, args.mode, protocol, task_name, seed)
                if not args.force and completed(metrics_path, token_path):
                    row = json.loads(metrics_path.read_text(encoding="utf-8"))
                    row["resumed"] = True
                    manifest_rows.append(row)
                    print(f"skipped completed protocol={protocol} task={task_name} seed={seed}", flush=True)
                    continue
                print(f"starting mode={args.mode} protocol={protocol} task={task_name} seed={seed}", flush=True)
                started = time.time()
                try:
                    result = run_eeg_supervised(
                        dataset=dataset,
                        split=splits[protocol],
                        metadata=metadata,
                        label_names=current_labels,
                        protocol=protocol,
                        seed=seed,
                        checkpoint=args.checkpoint,
                        runtime=runtime,
                        input_cache=input_cache,
                    )
                    arrays = result.pop("event_predictions")
                    embeddings = result.pop("embeddings")
                    run_dir.mkdir(parents=True, exist_ok=True)
                    prediction_path = run_dir / "event_predictions.npz"
                    np.savez_compressed(
                        prediction_path,
                        label_names=np.asarray(current_labels),
                        val_event_id=arrays["val"]["event_id"],
                        val_subject_id=arrays["val"]["subject_id"],
                        val_day_id=arrays["val"]["day_id"],
                        val_target=arrays["val"]["target"],
                        val_prediction=arrays["val"]["prediction"],
                        test_event_id=arrays["test"]["event_id"],
                        test_subject_id=arrays["test"]["subject_id"],
                        test_day_id=arrays["test"]["day_id"],
                        test_target=arrays["test"]["target"],
                        test_prediction=arrays["test"]["prediction"],
                    )
                    write_token(
                        token_path,
                        embeddings,
                        dataset=dataset,
                        metadata=metadata,
                        split=splits[protocol],
                        mode=args.mode,
                        task_name=task_name,
                        label_names=current_labels,
                        protocol=protocol,
                        seed=seed,
                        source_prediction=prediction_path,
                    )
                    result.update({
                        "mode": args.mode,
                        "task_name": task_name,
                        "prediction_path": str(prediction_path),
                        "embedding_path": str(token_path),
                    })
                    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    manifest_rows.append(result)
                    summary = result["test"]["summary"]
                    print(
                        f"completed mode={args.mode} protocol={protocol} task={task_name} seed={seed} "
                        f"srmse={summary['standardized_rmse']:.4f} raw_r={summary['raw_r']:.4f} "
                        f"seconds={result['duration_seconds']:.1f}",
                        flush=True,
                    )
                except Exception as exc:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    result = {
                        "status": "failed",
                        "mode": args.mode,
                        "protocol": protocol,
                        "task_name": task_name,
                        "label_names": list(current_labels),
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "duration_seconds": time.time() - started,
                    }
                    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    manifest_rows.append(result)
                    write_manifest(args, runtime, manifest_rows)
                    print(f"failed protocol={protocol} task={task_name} seed={seed}: {exc}", flush=True)
                    return 1
                write_manifest(args, runtime, manifest_rows)
            if args.limit_runs and attempted >= args.limit_runs:
                break
        if args.limit_runs and attempted >= args.limit_runs:
            break
    write_manifest(args, runtime, manifest_rows)
    print(f"run_count={len(manifest_rows)}", flush=True)
    return 0


def token_output_path(root: Path, mode: str, protocol: str, task_name: str, seed: int) -> Path:
    if mode == "single":
        return root / "single_task" / protocol / task_name / f"seed_{seed}.npz"
    return root / "multitask_11label" / protocol / f"seed_{seed}.npz"


def completed(metrics_path: Path, token_path: Path) -> bool:
    if not metrics_path.is_file() or not token_path.is_file():
        return False
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8")).get("status") == "ok"
    except (OSError, ValueError):
        return False


def write_token(
    path: Path,
    embeddings: np.ndarray,
    *,
    dataset: Any,
    metadata: dict[str, np.ndarray],
    split: Any,
    mode: str,
    task_name: str,
    label_names: tuple[str, ...],
    protocol: str,
    seed: int,
    source_prediction: Path,
) -> None:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (dataset.row_count, 256) or not np.isfinite(values).all():
        raise ValueError(f"invalid EEG token shape/values: {values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = (
        f"eegpt_partial_ft_single_{task_name}_v1"
        if mode == "single"
        else "eegpt_partial_ft_multitask_11label_v1"
    )
    np.savez_compressed(
        path,
        sample_id=dataset.sample_id,
        subject_id=metadata["subject_id"],
        day_id=metadata["day_id"],
        eeg_emb=values,
        eeg_mask=np.ones(dataset.row_count, dtype=np.int8),
        encoder_profile=np.asarray([profile] * dataset.row_count, dtype=object),
        protocol=np.asarray([protocol], dtype=object),
        seed=np.asarray([seed], dtype=np.int64),
        target_label=np.asarray([task_name], dtype=object),
        target_labels=np.asarray(label_names, dtype=object),
        train_index=split.train,
        val_index=split.val,
        test_index=split.test,
        train_supervision=np.asarray([f"{mode}_event_supervised_eegpt_partial_ft"], dtype=object),
        supervision_boundary=np.asarray(["raw_eeg_encoder_and_eeg_only_heads"], dtype=object),
        source_prediction_npz=np.asarray([str(source_prediction)], dtype=object),
    )


def write_manifest(args: Any, runtime: EEGSupervisedRuntime, rows: list[dict[str, Any]]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "phase3_single_task_eeg" if args.mode == "single" else "phase4_multitask_eeg",
        "mode": args.mode,
        "protocols": list(split_csv(args.protocols)),
        "labels": list(split_csv(args.labels)),
        "seeds": [int(value) for value in split_csv(args.seeds)],
        "checkpoint": str(args.checkpoint),
        "splits_root": str(args.splits_root),
        "runtime": runtime.__dict__,
        "selection_metric": "validation event-level macro standardized RMSE",
        "run_count": len(rows),
        "completed_count": sum(row.get("status") == "ok" for row in rows),
        "failed_count": sum(row.get("status") == "failed" for row in rows),
        "results": rows,
    }
    (args.out_root / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
