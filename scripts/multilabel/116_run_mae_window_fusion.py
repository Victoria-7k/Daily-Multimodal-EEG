"""Evaluate fixed label-free MAE tokens in the window-level 11-label fusion route.

This runner deliberately keeps the existing EEGPT/Wphysio/DINO-B0 tokens
unchanged.  It evaluates the paired replacements E1 (EEG-MAE) and W1
(Wear-MAE), alongside the matching B0 reference, with identical downstream
fusion settings and seeds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_MAE_ROOT = DEFAULT_ROOT / "outputs/mae_20260924"
CONDITIONS = {
    "B0": ("eeg_eegpt_frozen_v1", "wear_physio", "video_B0"),
    "E1": ("eeg_mae", "wear_physio", "video_B0"),
    "W1": ("eeg_eegpt_frozen_v1", "wear_mae", "video_B0"),
}
MODALITIES = {
    "eeg_eegpt_frozen_v1": "eeg",
    "eeg_mae": "eeg",
    "wear_physio": "wear",
    "wear_mae": "wear",
    "video_B0": "video",
}


def _load_base() -> Any:
    path = Path(__file__).with_name("48_run_fixed_tokens_multilabel_fusion.py")
    spec = importlib.util.spec_from_file_location("fixed_tokens_fusion", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shared fusion implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _source_path(name: str, *, protocol: str, embeddings_root: Path, mae_root: Path) -> tuple[Path, str, str]:
    if name == "eeg_mae":
        return mae_root / protocol / "eeg_seed_240800" / "window_embeddings.npz", "embedding", "valid_mask"
    if name == "wear_mae":
        return mae_root / protocol / "wear_seed_240800" / "window_embeddings.npz", "embedding", "valid_mask"
    branch = BASE.BRANCHES[name]
    return embeddings_root / branch.filename, branch.emb_key, branch.mask_key


def _load_branch(
    name: str,
    *,
    protocol: str,
    sample_id: np.ndarray,
    embeddings_root: Path,
    mae_root: Path,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (protocol if name.endswith("_mae") else "shared", name)
    if key in cache:
        return cache[key]
    path, emb_key, mask_key = _source_path(name, protocol=protocol, embeddings_root=embeddings_root, mae_root=mae_root)
    # Older H20 NumPy is compatible with the numeric arrays but needs these
    # aliases to unpickle object-dtype sample_id fields written by NumPy 2.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)
    with np.load(path, allow_pickle=True) as loaded:
        if "sample_id" not in loaded.files:
            raise ValueError(f"{name} lacks sample_id: {path}")
        if not np.array_equal(loaded["sample_id"].astype(str), sample_id):
            raise ValueError(f"{name} sample_id order does not match canonical index: {path}")
        if emb_key not in loaded.files:
            raise ValueError(f"{name} lacks {emb_key}: {path}")
        embedding = loaded[emb_key].astype(np.float32)
        if mask_key in loaded.files:
            mask = loaded[mask_key].astype(bool)
        else:
            branch = BASE.BRANCHES[name]
            mask = loaded["modality_mask"][:, branch.modality_index].astype(bool)
    if embedding.shape != (len(sample_id), 256) or mask.shape != (len(sample_id),):
        raise ValueError(f"invalid {name} shape: embedding={embedding.shape}, mask={mask.shape}")
    if not np.all(np.isfinite(embedding)):
        raise ValueError(f"{name} contains non-finite values")
    result = {"embedding": embedding, "mask": mask, "path": str(path), "mask_sum": int(mask.sum())}
    cache[key] = result
    return result


def _tokens(
    names: tuple[str, ...],
    *,
    protocol: str,
    sample_id: np.ndarray,
    embeddings_root: Path,
    mae_root: Path,
    cache: dict[tuple[str, str], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = {
        name: _load_branch(
            name,
            protocol=protocol,
            sample_id=sample_id,
            embeddings_root=embeddings_root,
            mae_root=mae_root,
            cache=cache,
        )
        for name in names
    }
    token = np.stack([data[name]["embedding"] for name in names], axis=1).astype(np.float32)
    mask = np.stack([data[name]["mask"] for name in names], axis=1).astype(bool)
    report = {
        name: {"path": data[name]["path"], "mask_sum": data[name]["mask_sum"], "modality": MODALITIES[name]}
        for name in names
    }
    return token, mask, report


def _output_header(args: argparse.Namespace, label_audit: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "mae_window_fixed_token_fusion_v1",
        "root": str(args.root),
        "splits_root": str(args.splits_root),
        "embeddings_root": str(args.embeddings_root),
        "mae_root": str(args.mae_root),
        "label_names": list(BASE.LABEL_NAMES),
        "label_audit": label_audit,
        "runtime": {
            "protocols": list(_parse_csv(args.protocols)),
            "conditions": list(_parse_csv(args.conditions)),
            "downstream_seeds": [int(value) for value in _parse_csv(args.seeds)],
            "upstream_mae_seed": 240800,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "patience": args.patience,
            "device": args.device,
            "train_rule": "pretrain + finetune",
            "checkpoint_selection": "validation standardized 11-label MSE",
            "test_target_usage": "evaluation_only_after_checkpoint_selection",
        },
        "run_count": len(results),
        "results": results,
        "route_winners": BASE._route_winners(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--mae-root", type=Path, default=DEFAULT_MAE_ROOT)
    parser.add_argument("--protocols", default="cross_day,within_subject_day")
    parser.add_argument("--conditions", default="B0,E1,W1")
    parser.add_argument("--seeds", default="240800,240801,240802")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    torch.set_num_threads(max(1, args.torch_threads))
    protocols = _parse_csv(args.protocols)
    conditions = _parse_csv(args.conditions)
    seeds = [int(value) for value in _parse_csv(args.seeds)]
    if not protocols or not conditions or not seeds:
        raise ValueError("protocols, conditions, and seeds must be non-empty")
    unknown = set(conditions).difference(CONDITIONS)
    if unknown:
        raise ValueError(f"unsupported conditions: {sorted(unknown)}")

    rows = BASE._load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([BASE._norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    targets = BASE._load_targets(rows)
    event_window_id = np.asarray([int(row.get("event_window_id", -1)) for row in rows], dtype=np.int64)
    label_audit = BASE._audit_labels(targets, rows, event_window_id)
    if not label_audit["gate_passed"]:
        output = _output_header(args, label_audit, [])
        BASE._write_json(output, args.out_json)
        BASE._write_markdown(output, args.out_md)
        print("label_audit_gate_passed=false", flush=True)
        return 2

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for protocol in protocols:
        split = BASE._load_split(args.splits_root / protocol, len(rows))
        for condition in conditions:
            branches = CONDITIONS[condition]
            tokens, token_mask, branch_report = _tokens(
                branches,
                protocol=protocol,
                sample_id=sample_id,
                embeddings_root=args.embeddings_root,
                mae_root=args.mae_root,
                cache=cache,
            )
            for seed in seeds:
                print(f"starting protocol={protocol} condition={condition} downstream_seed={seed}", flush=True)
                model, train_audit = BASE._fit_model(
                    tokens=tokens, token_mask=token_mask, targets=targets,
                    train_idx=split["train"], val_idx=split["val"], hidden_dim=args.hidden_dim,
                    epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, dropout=args.dropout, patience=args.patience,
                    seed=seed, device=args.device,
                )
                predictions = {
                    name: BASE._predict(model, tokens, token_mask, indices=indices, device=args.device)
                    for name, indices in split.items() if name in {"train", "val", "test"}
                }
                result = {
                    "protocol": protocol,
                    "experiment": condition,
                    "route_name": "mae_window_fixed_token_fusion_v1",
                    "seed": seed,
                    "embedding_seed": 240800,
                    "branches": list(branches),
                    "enabled_modalities": [MODALITIES[name] for name in branches],
                    "row_count": len(rows),
                    "label_names": list(BASE.LABEL_NAMES),
                    "supervision_boundary": "label_free_fixed_mae_or_frozen_tokens__train_fusion_and_11label_head_only",
                    "protocol_interpretation": (
                        "held_out_subject_days" if protocol == "cross_day" else "legacy_same_subject_same_day_window_holdout"
                    ),
                    "split_counts": {name: int(len(values)) for name, values in split.items()},
                    "mask_coverage_by_split": BASE._mask_coverage(token_mask, branches, split),
                    "train": BASE._evaluate_multilabel(targets[split["train"]], predictions["train"], subject_id[split["train"]]),
                    "val": BASE._evaluate_multilabel(targets[split["val"]], predictions["val"], subject_id[split["val"]]),
                    "test": BASE._evaluate_multilabel(targets[split["test"]], predictions["test"], subject_id[split["test"]]),
                    "train_audit": train_audit,
                    "branch_report": branch_report,
                }
                if args.predictions_dir:
                    pred_path = args.predictions_dir / protocol / condition / f"seed_{seed}.npz"
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        pred_path, train_index=split["train"], val_index=split["val"], test_index=split["test"],
                        train_prediction=predictions["train"], val_prediction=predictions["val"],
                        test_prediction=predictions["test"], target=targets,
                        label_names=np.asarray(BASE.LABEL_NAMES, dtype=object), sample_id=sample_id,
                        subject_id=subject_id, event_id=event_id,
                    )
                    result["prediction_path"] = str(pred_path)
                results.append(result)
                summary = result["test"]["summary"]
                fatigue = result["test"]["per_label"]["fatigue"]
                print(
                    f"completed protocol={protocol} condition={condition} downstream_seed={seed} "
                    f"mean_rmse={BASE._fmt(summary['mean_rmse'])} mean_raw_r={BASE._fmt(summary['mean_raw_r'])} "
                    f"fatigue_raw_r={BASE._fmt(fatigue['raw_r'])}", flush=True,
                )

    output = _output_header(args, label_audit, results)
    BASE._write_json(output, args.out_json)
    BASE._write_markdown(output, args.out_md)
    print(f"run_count={len(results)}", flush=True)
    print(f"out_json={args.out_json}", flush=True)
    print(f"out_md={args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
