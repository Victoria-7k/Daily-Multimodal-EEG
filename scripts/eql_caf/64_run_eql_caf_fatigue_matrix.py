#!/usr/bin/env python3
"""Train EQL-CAF temporal-token fatigue models from packed Phase 1 NPZ files."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered
from daily_multimodal.training.eql_caf import make_eql_caf_variant


DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/splits")
DEFAULT_PROTOCOLS = ("cross_day", "within_subject_day")
DEFAULT_MODELS = ("B1", "B2", "M1", "M2", "M3", "M4", "M5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-npz", type=Path, help="Single packed temporal NPZ. Overrides --token-root template.")
    parser.add_argument("--token-root", type=Path, help="Root containing {protocol}/eql_caf_temporal_tokens_v1_seed_{seed}.npz.")
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="240800")
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    if args.packed_npz is None and args.token_root is None:
        raise ValueError("one of --packed-npz or --token-root is required")

    protocols = _split_csv(args.protocols)
    model_names = _split_csv(args.models)
    seeds = [int(value) for value in _split_csv(args.seeds)]
    results: list[dict[str, Any]] = []
    run_count = 0
    for protocol in protocols:
        for seed in seeds:
            packed_path = args.packed_npz or args.token_root / protocol / f"eql_caf_temporal_tokens_v1_seed_{seed}.npz"
            data = _load_packed_dataset(packed_path, target_label=args.target_label)
            split = _load_split(args.splits_root / protocol, data["row_count"])
            for model_name in model_names:
                run_count += 1
                if args.limit_runs and run_count > args.limit_runs:
                    break
                print(f"starting protocol={protocol} model={model_name} seed={seed}", flush=True)
                model, audit = _fit_model(
                    data=data,
                    split=split,
                    model_name=model_name,
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    patience=args.patience,
                    modality_dropout=args.modality_dropout if model_name == "M5" else 0.0,
                    device=args.device,
                )
                predictions = {
                    name: _predict(model, data, indices=indices, device=args.device)
                    for name, indices in split.items()
                    if name in {"train", "val", "test"}
                }
                result = {
                    "protocol": protocol,
                    "model": model_name,
                    "seed": int(seed),
                    "packed_npz": str(packed_path),
                    "row_count": int(data["row_count"]),
                    "target_label": args.target_label,
                    "split_counts": {name: int(len(values)) for name, values in split.items() if name in {"train", "val", "test"}},
                    "train": evaluate_regression_with_centered(data["target"][split["train"]], predictions["train"], data["subject_id"][split["train"]]),
                    "val": evaluate_regression_with_centered(data["target"][split["val"]], predictions["val"], data["subject_id"][split["val"]]),
                    "test": evaluate_regression_with_centered(data["target"][split["test"]], predictions["test"], data["subject_id"][split["test"]]),
                    "train_audit": audit,
                }
                pred_path = args.out_root / "predictions" / protocol / model_name / f"seed_{seed}.npz"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    pred_path,
                    sample_id=data["sample_id"],
                    subject_id=data["subject_id"],
                    target=data["target"],
                    train_index=split["train"],
                    val_index=split["val"],
                    test_index=split["test"],
                    train_prediction=predictions["train"],
                    val_prediction=predictions["val"],
                    test_prediction=predictions["test"],
                )
                result["prediction_path"] = str(pred_path)
                results.append(result)
                print(
                    f"completed protocol={protocol} model={model_name} seed={seed} "
                    f"test_rmse={_fmt(result['test']['rmse'])} test_raw_r={_fmt(result['test']['raw_r'])}",
                    flush=True,
                )
            if args.limit_runs and run_count >= args.limit_runs:
                break
        if args.limit_runs and run_count >= args.limit_runs:
            break
    output = {
        "script": Path(__file__).name,
        "models": list(model_names),
        "protocols": list(protocols),
        "seeds": seeds,
        "train_rule": "pretrain + finetune; train-only token normalization",
        "run_count": len(results),
        "results": results,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_json = args.out_root / "eql_caf_matrix_report.json"
    out_md = args.out_root / "eql_caf_matrix_report.md"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output, out_md)
    print(f"run_count={len(results)}")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    return 0


def _load_packed_dataset(path: Path, *, target_label: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as loaded:
        label_names = tuple(str(value) for value in loaded["label_names"].tolist())
        labels = loaded["labels"].astype(np.float32)
        if target_label in label_names:
            target = labels[:, label_names.index(target_label)]
        elif labels.shape[1] == 1:
            target = labels[:, 0]
        else:
            raise ValueError(f"target label {target_label!r} not found in packed label_names {label_names}")
        tokens = np.stack([loaded[f"{modality}_tokens"].astype(np.float32) for modality in ("eeg", "wear", "video", "audio")], axis=1)
        return {
            "sample_id": loaded["sample_id"].astype(str),
            "subject_id": loaded["subject_id"].astype(str),
            "target": target.astype(np.float32),
            "tokens": tokens,
            "token_mask": loaded["token_mask"].astype(bool),
            "quality_features": loaded["quality_features"].astype(np.float32),
            "row_count": int(tokens.shape[0]),
        }


def _fit_model(
    *,
    data: dict[str, Any],
    split: dict[str, np.ndarray],
    model_name: str,
    seed: int,
    hidden_dim: int,
    num_heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    modality_dropout: float,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    x_mean, x_std = _fit_token_normalization(data["tokens"], data["token_mask"], split["train"])
    y_mean = float(data["target"][split["train"]].mean())
    y_std = float(data["target"][split["train"]].std()) or 1.0
    dev = torch.device(device)
    module = make_eql_caf_variant(
        model_name,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_tokens=data["tokens"].shape[2],
        quality_feature_dim=data["quality_features"].shape[-1],
    ).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    train_idx = split["train"]
    val_idx = split["val"]
    for epoch in range(max(1, epochs)):
        module.train()
        losses = []
        for batch in _batches(train_idx, batch_size, rng):
            tokens, mask, quality, target = _batch_tensors(data, batch, x_mean, x_std, y_mean, y_std, dev)
            if modality_dropout > 0:
                mask = _apply_modality_dropout(mask, probability=modality_dropout)
            prediction = module(tokens, mask, quality)["prediction"]
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        module.eval()
        with torch.no_grad():
            tokens, mask, quality, target = _batch_tensors(data, val_idx, x_mean, x_std, y_mean, y_std, dev)
            val_loss = float(torch.mean((module(tokens, mask, quality)["prediction"] - target) ** 2).detach().cpu().item())
        row = {"epoch": float(epoch + 1), "train_loss": float(np.mean(losses)) if losses else math.nan, "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    return {
        "module": module,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epoch_count": int(len(history)),
        "initial_train_loss": history[0]["train_loss"] if history else math.nan,
        "final_train_loss": history[-1]["train_loss"] if history else math.nan,
        "normalization": "train_only",
        "modality_dropout": float(modality_dropout),
        "history": history,
    }


def _predict(model: dict[str, Any], data: dict[str, Any], *, indices: np.ndarray, device: str) -> np.ndarray:
    module = model["module"]
    dev = torch.device(device)
    values = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            batch = indices[start : start + 1024]
            tokens, mask, quality, _target = _batch_tensors(data, batch, model["x_mean"], model["x_std"], model["y_mean"], model["y_std"], dev)
            pred = module(tokens, mask, quality)["prediction"].detach().cpu().numpy()
            values.append((pred * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, :, None], tokens[indices], np.nan)
    mean = np.nanmean(available, axis=(0, 1, 2), keepdims=True)
    std = np.nanstd(available, axis=(0, 1, 2), keepdims=True)
    return np.where(np.isfinite(mean), mean, 0.0).astype(np.float32), np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)


def _batch_tensors(data: dict[str, Any], indices: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: float, y_std: float, device: torch.device):
    tokens = ((data["tokens"][indices].astype(np.float32) - x_mean) / x_std).astype(np.float32)
    target = ((data["target"][indices] - y_mean) / y_std).astype(np.float32)
    return (
        torch.as_tensor(tokens, dtype=torch.float32, device=device),
        torch.as_tensor(data["token_mask"][indices], dtype=torch.bool, device=device),
        torch.as_tensor(data["quality_features"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(target, dtype=torch.float32, device=device),
    )


def _apply_modality_dropout(mask: torch.Tensor, *, probability: float) -> torch.Tensor:
    if probability <= 0:
        return mask
    dropped = mask.clone()
    keep = torch.rand((mask.shape[0], 3), device=mask.device) >= probability
    dropped[:, 1:, :] = dropped[:, 1:, :] & keep[:, :, None]
    dropped[:, 0, :] = mask[:, 0, :]
    return dropped


def _load_split(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    split = {name: _load_indices(path / f"{name}.json", n_rows) for name in ("pretrain", "finetune", "val", "test")}
    split["train"] = np.asarray(split["pretrain"].tolist() + split["finetune"].tolist(), dtype=np.int64)
    return split


def _load_indices(path: Path, n_rows: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= n_rows):
        raise ValueError(f"{path} contains out-of-range indices")
    return values


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    rows = ["# EQL-CAF Matrix Report", "", "| protocol | model | seed | test RMSE | test raw r | test centered r |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for result in output["results"]:
        rows.append(
            f"| {result['protocol']} | {result['model']} | {result['seed']} | "
            f"{_fmt(result['test']['rmse'])} | {_fmt(result['test']['raw_r'])} | {_fmt(result['test']['within_subject_centered_r'])} |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
