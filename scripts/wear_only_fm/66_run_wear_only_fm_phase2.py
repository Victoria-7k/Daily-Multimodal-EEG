#!/usr/bin/env python3
"""Run Wear-only FM Phase 2, the 36-run wearable-only fatigue matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered
from daily_multimodal.split_paths import resolve_protocol_split_root


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_WEAR_FM_ROOT = DEFAULT_ROOT / "outputs/wear_fm"
DEFAULT_PROTOCOLS = ("cross_day", "date_in_order", "cross_subject")
DEFAULT_ROUTES = ("Wphysio", "Wdeep", "Wmoment_frozen", "W3FM_frozen")
DEFAULT_SEEDS = (240729, 240730, 240731)
LABEL_NAMES = (
    "inspired",
    "alert",
    "determined",
    "attentive",
    "active",
    "hostile",
    "nervous",
    "upset",
    "afraid",
    "ashamed",
    "fatigue",
)


@dataclass(frozen=True)
class Dataset:
    sample_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    event_id: np.ndarray
    target: np.ndarray
    complete_mask: np.ndarray
    physio: np.ndarray
    deep: np.ndarray
    ppg: np.ndarray
    acc: np.ndarray
    gsr: np.ndarray


class WearHead(torch.nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


class W3FMModel(torch.nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj_ppg = _projector(512, dropout)
        self.proj_acc = _projector(1024, dropout)
        self.proj_gsr = _projector(768, dropout)
        self.gate = torch.nn.Linear(256, 1)
        self.head = WearHead(input_dim=256, hidden_dim=hidden_dim, dropout=dropout)

    def encode(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ppg, acc, gsr = inputs
        tokens = torch.stack([self.proj_ppg(ppg), self.proj_acc(acc), self.proj_gsr(gsr)], dim=1)
        weights = torch.softmax(self.gate(tokens).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(inputs))


def _projector(input_dim: int, dropout: float) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--wear-fm-root", type=Path, default=DEFAULT_WEAR_FM_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--selection-metric", choices=("rmse", "raw_r", "centered_r"), default="rmse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT / "outputs/wear_fm/phase2")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    dataset = _load_dataset(args)
    protocols = _split_csv(args.protocols)
    routes = _split_csv(args.routes)
    seeds = [int(value) for value in _split_csv(args.seeds)]
    _validate_routes(routes)
    _write_preflight(dataset, args.out_root / "phase2_preflight.json")
    results: list[dict[str, Any]] = []
    started = time.time()
    run_number = 0
    for protocol in protocols:
        split = _load_split(resolve_protocol_split_root(args.splits_root, protocol), dataset.sample_id.shape[0])
        split = _filter_split(split, dataset.complete_mask)
        for seed in seeds:
            for route in routes:
                run_number += 1
                if args.limit_runs and run_number > args.limit_runs:
                    break
                run_dir = args.out_root / "runs" / protocol / route / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                if args.skip_existing and metrics_path.is_file():
                    results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    print(f"skipped existing protocol={protocol} route={route} seed={seed}", flush=True)
                    continue
                print(f"starting protocol={protocol} route={route} seed={seed}", flush=True)
                result = _run_one(args, dataset, split, protocol=protocol, route=route, seed=seed, run_dir=run_dir)
                results.append(result)
                print(
                    f"completed protocol={protocol} route={route} seed={seed} "
                    f"rmse={_fmt(result['test']['rmse'])} raw_r={_fmt(result['test']['raw_r'])} "
                    f"centered_r={_fmt(result['test']['within_subject_centered_r'])}",
                    flush=True,
                )
            if args.limit_runs and run_number >= args.limit_runs:
                break
        if args.limit_runs and run_number >= args.limit_runs:
            break
    output = {
        "script": Path(__file__).name,
        "stage": "wear_only_fm_phase2",
        "target_label": args.target_label,
        "protocols": list(protocols),
        "routes": list(routes),
        "seeds": seeds,
        "run_count": len(results),
        "elapsed_seconds": float(time.time() - started),
        "mask_count": int(dataset.complete_mask.sum()),
        "results": results,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    report_json = args.out_root / "wear_only_fm_phase2_report.json"
    report_md = args.out_root / "wear_only_fm_phase2_report.md"
    report_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output, report_md)
    print(f"run_count={len(results)}")
    print(f"out_json={report_json}")
    print(f"out_md={report_md}")
    return 0


def _load_dataset(args: argparse.Namespace) -> Dataset:
    rows = _load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([_norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    day_id = np.asarray([str(row.get("day_id", "")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    target = np.asarray([_target_value(row, args.target_label) for row in rows], dtype=np.float32)
    complete_mask = np.load(args.wear_fm_root / "staged_inputs/wear_complete_mask.npy").astype(bool)
    if complete_mask.shape != (len(rows),):
        raise ValueError(f"invalid wear_complete_mask shape: {complete_mask.shape}")
    physio = _load_wear_npz(args.embeddings_root / "wear/wear_physio_preprocessed_eeg23win_embeddings.npz", sample_id, complete_mask, "Wphysio")
    deep = _load_wear_npz(args.embeddings_root / "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz", sample_id, complete_mask, "Wdeep")
    ppg = _load_array(args.wear_fm_root / "embeddings/ppg_papagei_s_512d.npy", (len(rows), 512), "PPG")
    acc = _load_array(args.wear_fm_root / "embeddings/acc_harnet10.npy", (len(rows), 1024), "ACC")
    gsr = _load_array(args.wear_fm_root / "embeddings/gsr_normwear_768d.npy", (len(rows), 768), "GSR")
    for name, path in (
        ("PPG", args.wear_fm_root / "embeddings/ppg_papagei_s_valid_mask.npy"),
        ("ACC", args.wear_fm_root / "embeddings/acc_harnet10_valid_mask.npy"),
        ("GSR", args.wear_fm_root / "embeddings/gsr_normwear_valid_mask.npy"),
    ):
        mask = np.load(path).astype(bool)
        if not np.array_equal(mask, complete_mask):
            raise ValueError(f"{name} embedding mask does not equal wear_complete_mask")
    return Dataset(sample_id, subject_id, day_id, event_id, target, complete_mask, physio, deep, ppg, acc, gsr)


def _load_wear_npz(path: Path, sample_id: np.ndarray, complete_mask: np.ndarray, route: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as loaded:
        ids = loaded["sample_id"].astype(str)
        if not np.array_equal(ids, sample_id):
            raise ValueError(f"{route} sample_id does not match canonical index")
        emb = loaded["wear_emb"].astype(np.float32)
        mask = loaded["wear_mask"].astype(bool) if "wear_mask" in loaded.files else loaded["modality_mask"][:, 1].astype(bool)
    if emb.shape != (len(sample_id), 256):
        raise ValueError(f"{route} embedding shape {emb.shape} is not {(len(sample_id), 256)}")
    if int((complete_mask & ~mask).sum()) != 0:
        raise ValueError(f"{route} is missing rows inside wear_complete_mask")
    if not np.isfinite(emb[complete_mask]).all():
        raise ValueError(f"{route} has non-finite values inside wear_complete_mask")
    return emb


def _load_wmoment(args: argparse.Namespace, sample_id: np.ndarray, complete_mask: np.ndarray, protocol: str, seed: int) -> tuple[np.ndarray, str]:
    path = args.embeddings_root / "wear_tokens" / protocol / "wear_moment_frozen_v1" / f"seed_{seed}.npz"
    with np.load(path, allow_pickle=True) as loaded:
        ids = loaded["sample_id"].astype(str)
        if not np.array_equal(ids, sample_id):
            raise ValueError(f"Wmoment sample_id mismatch for {path}")
        emb = loaded["wear_emb"].astype(np.float32)
        mask = loaded["wear_mask"].astype(bool) if "wear_mask" in loaded.files else loaded["modality_mask"][:, 1].astype(bool)
    if emb.shape != (len(sample_id), 256):
        raise ValueError(f"Wmoment embedding shape {emb.shape} is invalid")
    if int((complete_mask & ~mask).sum()) != 0:
        raise ValueError(f"Wmoment is missing rows inside wear_complete_mask for {protocol} seed {seed}")
    return emb, str(path)


def _run_one(
    args: argparse.Namespace,
    dataset: Dataset,
    split: dict[str, np.ndarray],
    *,
    protocol: str,
    route: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if route == "Wphysio":
        model, audit = _fit_vector_model(args, dataset.physio, dataset, split, seed=seed)
        predictions = {name: _predict_vector_model(model, dataset.physio, indices=indices, device=args.device) for name, indices in _eval_splits(split).items()}
        source = str(args.embeddings_root / "wear/wear_physio_preprocessed_eeg23win_embeddings.npz")
    elif route == "Wdeep":
        model, audit = _fit_vector_model(args, dataset.deep, dataset, split, seed=seed)
        predictions = {name: _predict_vector_model(model, dataset.deep, indices=indices, device=args.device) for name, indices in _eval_splits(split).items()}
        source = str(args.embeddings_root / "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz")
    elif route == "Wmoment_frozen":
        emb, source = _load_wmoment(args, dataset.sample_id, dataset.complete_mask, protocol, seed)
        model, audit = _fit_vector_model(args, emb, dataset, split, seed=seed)
        predictions = {name: _predict_vector_model(model, emb, indices=indices, device=args.device) for name, indices in _eval_splits(split).items()}
    elif route == "W3FM_frozen":
        model, audit = _fit_w3fm_model(args, dataset, split, seed=seed)
        predictions = {name: _predict_w3fm_model(model, dataset, indices=indices, device=args.device) for name, indices in _eval_splits(split).items()}
        source = str(args.wear_fm_root / "embeddings/embedding_manifest.json")
    else:
        raise ValueError(f"unsupported route: {route}")
    result = {
        "protocol": protocol,
        "route": route,
        "seed": int(seed),
        "source": source,
        "target_label": args.target_label,
        "mask_count": int(dataset.complete_mask.sum()),
        "split_counts": {name: int(len(indices)) for name, indices in _eval_splits(split).items()},
        "train": _metric_aliases(evaluate_regression_with_centered(dataset.target[split["train"]], predictions["train"], dataset.subject_id[split["train"]])),
        "val": _metric_aliases(evaluate_regression_with_centered(dataset.target[split["val"]], predictions["val"], dataset.subject_id[split["val"]])),
        "test": _metric_aliases(evaluate_regression_with_centered(dataset.target[split["test"]], predictions["test"], dataset.subject_id[split["test"]])),
        "train_audit": audit,
    }
    pred_npz = run_dir / "predictions.npz"
    np.savez_compressed(
        pred_npz,
        sample_id=dataset.sample_id,
        subject_id=dataset.subject_id,
        day_id=dataset.day_id,
        event_id=dataset.event_id,
        target=dataset.target,
        train_index=split["train"],
        val_index=split["val"],
        test_index=split["test"],
        train_prediction=predictions["train"],
        val_prediction=predictions["val"],
        test_prediction=predictions["test"],
    )
    _write_predictions_table(run_dir / "test_predictions.csv", dataset, split["test"], predictions["test"])
    _try_write_parquet(run_dir / "test_predictions.parquet", dataset, split["test"], predictions["test"])
    torch.save({"state_dict": audit.pop("_best_state"), "route": route, "protocol": protocol, "seed": int(seed)}, run_dir / "best_checkpoint.pt")
    (run_dir / "val_history.csv").write_text(_history_csv(audit["history"]), encoding="utf-8")
    result["prediction_path"] = str(pred_npz)
    result["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")
    result["test_predictions_csv"] = str(run_dir / "test_predictions.csv")
    result["test_predictions_parquet"] = str(run_dir / "test_predictions.parquet") if (run_dir / "test_predictions.parquet").is_file() else None
    result["metrics_path"] = str(run_dir / "metrics.json")
    result["config_path"] = str(run_dir / "config.json")
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_config_snapshot(args, protocol, route, seed, source), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fit_vector_model(args: argparse.Namespace, emb: np.ndarray, dataset: Dataset, split: dict[str, np.ndarray], *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    device = torch.device(args.device)
    train = split["train"]
    val = split["val"]
    x_mean = emb[train].mean(axis=0, keepdims=True).astype(np.float32)
    x_std = emb[train].std(axis=0, keepdims=True).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(dataset.target[train].mean())
    y_std = float(dataset.target[train].std()) or 1.0
    module = WearHead(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    audit = _train_loop(args, module, train, val, seed, lambda idx: ((emb[idx] - x_mean) / x_std).astype(np.float32), dataset.target, dataset.subject_id, y_mean, y_std)
    return {"module": module, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}, audit


def _fit_w3fm_model(args: argparse.Namespace, dataset: Dataset, split: dict[str, np.ndarray], *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    device = torch.device(args.device)
    train = split["train"]
    val = split["val"]
    means = tuple(arr[train].mean(axis=0, keepdims=True).astype(np.float32) for arr in (dataset.ppg, dataset.acc, dataset.gsr))
    stds = tuple(arr[train].std(axis=0, keepdims=True).astype(np.float32) for arr in (dataset.ppg, dataset.acc, dataset.gsr))
    stds = tuple(np.where(std < 1e-6, 1.0, std).astype(np.float32) for std in stds)
    y_mean = float(dataset.target[train].mean())
    y_std = float(dataset.target[train].std()) or 1.0
    module = W3FMModel(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    def inputs(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(((arr[indices] - mean) / std).astype(np.float32) for arr, mean, std in zip((dataset.ppg, dataset.acc, dataset.gsr), means, stds))

    audit = _train_loop(args, module, train, val, seed, inputs, dataset.target, dataset.subject_id, y_mean, y_std)
    return {"module": module, "means": means, "stds": stds, "y_mean": y_mean, "y_std": y_std}, audit


def _train_loop(
    args: argparse.Namespace,
    module: torch.nn.Module,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
    input_fn,
    target: np.ndarray,
    subject_id: np.ndarray,
    y_mean: float,
    y_std: float,
) -> dict[str, Any]:
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(max(1, args.epochs)):
        module.train()
        losses: list[float] = []
        shuffled = train.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            x = _to_torch(input_fn(batch), device)
            y = torch.as_tensor((target[batch] - y_mean) / y_std, dtype=torch.float32, device=device)
            pred = module(x)
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        module.eval()
        with torch.no_grad():
            val_pred = []
            for start in range(0, len(val), 1024):
                batch = val[start : start + 1024]
                pred = module(_to_torch(input_fn(batch), device)).detach().cpu().numpy()
                val_pred.append((pred * y_std + y_mean).astype(np.float32))
            val_values = np.concatenate(val_pred) if val_pred else np.zeros((0,), dtype=np.float32)
        val_metrics = _metric_aliases(evaluate_regression_with_centered(target[val], val_values, subject_id[val]))
        score = _selection_score(val_metrics, args.selection_metric)
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_raw_r": float(val_metrics["raw_r"]),
            "val_within_subject_centered_r": float(val_metrics["within_subject_centered_r"]),
            "val_prediction_std": float(np.std(val_values)) if len(val_values) else math.nan,
            "selection_score": float(score),
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no best state")
    module.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "best_selection_score": float(best_score),
        "selection_metric": str(args.selection_metric),
        "epoch_count": int(len(history)),
        "history": history,
        "trainable_params": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in module.parameters())),
        "_best_state": best_state,
    }


def _predict_vector_model(model: dict[str, Any], emb: np.ndarray, *, indices: np.ndarray, device: str) -> np.ndarray:
    module = model["module"]
    module.eval()
    values: list[np.ndarray] = []
    dev = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            batch = indices[start : start + 1024]
            x = (emb[batch] - model["x_mean"]) / model["x_std"]
            pred = module(torch.as_tensor(x.astype(np.float32), dtype=torch.float32, device=dev)).detach().cpu().numpy()
            values.append((pred * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _predict_w3fm_model(model: dict[str, Any], dataset: Dataset, *, indices: np.ndarray, device: str) -> np.ndarray:
    module = model["module"]
    module.eval()
    values: list[np.ndarray] = []
    dev = torch.device(device)
    arrays = (dataset.ppg, dataset.acc, dataset.gsr)
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            batch = indices[start : start + 1024]
            x = tuple(((arr[batch] - mean) / std).astype(np.float32) for arr, mean, std in zip(arrays, model["means"], model["stds"]))
            pred = module(_to_torch(x, dev)).detach().cpu().numpy()
            values.append((pred * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _to_torch(value: Any, device: torch.device) -> Any:
    if isinstance(value, tuple):
        return tuple(torch.as_tensor(item, dtype=torch.float32, device=device) for item in value)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _write_predictions_table(path: Path, dataset: Dataset, indices: np.ndarray, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "sample_id", "subject_id", "day_id", "event_id", "target", "prediction"])
        for row_id, pred in zip(indices.tolist(), prediction.tolist()):
            writer.writerow([row_id, dataset.sample_id[row_id], dataset.subject_id[row_id], dataset.day_id[row_id], dataset.event_id[row_id], float(dataset.target[row_id]), float(pred)])


def _try_write_parquet(path: Path, dataset: Dataset, indices: np.ndarray, prediction: np.ndarray) -> None:
    try:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "row_id": indices.astype(np.int64),
                "sample_id": dataset.sample_id[indices],
                "subject_id": dataset.subject_id[indices],
                "day_id": dataset.day_id[indices],
                "event_id": dataset.event_id[indices],
                "target": dataset.target[indices].astype(np.float32),
                "prediction": prediction.astype(np.float32),
            }
        )
        frame.to_parquet(path, index=False)
    except Exception:
        return


def _write_preflight(dataset: Dataset, path: Path) -> None:
    payload = {
        "row_count": int(dataset.sample_id.shape[0]),
        "wear_complete_count": int(dataset.complete_mask.sum()),
        "target_finite": bool(np.isfinite(dataset.target).all()),
        "physio_shape": list(dataset.physio.shape),
        "deep_shape": list(dataset.deep.shape),
        "ppg_shape": list(dataset.ppg.shape),
        "acc_shape": list(dataset.acc.shape),
        "gsr_shape": list(dataset.gsr.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Wear-only FM Phase 2 Report",
        "",
        f"- run_count: `{output['run_count']}`",
        f"- mask_count: `{output['mask_count']}`",
        "",
        "| protocol | route | seed | test RMSE | test MAE | test raw r | test centered r | per-subject r mean | best epoch | trainable params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        test = row["test"]
        audit = row["train_audit"]
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['seed']} | {_fmt(test['rmse'])} | {_fmt(test['mae'])} | "
            f"{_fmt(test['raw_r'])} | {_fmt(test['within_subject_centered_r'])} | {_fmt(test.get('per_subject_r_mean'))} | "
            f"{audit['best_epoch']} | {audit['trainable_params']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _history_csv(history: list[dict[str, Any]]) -> str:
    fields = [
        "epoch",
        "train_loss",
        "val_rmse",
        "val_mae",
        "val_raw_r",
        "val_within_subject_centered_r",
        "val_prediction_std",
        "selection_score",
    ]
    rows = [",".join(fields)]
    rows.extend(",".join(str(item.get(field, "")) for field in fields) for item in history)
    return "\n".join(rows) + "\n"


def _config_snapshot(args: argparse.Namespace, protocol: str, route: str, seed: int, source: str) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "route": route,
        "seed": int(seed),
        "source": source,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "hidden_dim": int(args.hidden_dim),
        "patience": int(args.patience),
        "selection_metric": str(args.selection_metric),
        "train_supervision": "pretrain_plus_finetune_train_val_early_stop_test_once",
        "mask_policy": "wear_complete_mask intersected with split indices for all routes",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _target_value(row: dict[str, Any], target_label: str) -> float:
    if "label_columns" in row and target_label in row["label_columns"]:
        return float(row["label_columns"][target_label])
    label_names = tuple(str(value) for value in row.get("label_names") or LABEL_NAMES)
    if target_label not in label_names:
        raise ValueError(f"target label {target_label!r} missing from row")
    return float(row["labels"][label_names.index(target_label)])


def _load_array(path: Path, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.load(path).astype(np.float32)
    if values.shape != shape:
        raise ValueError(f"{name} shape {values.shape} is not {shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


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


def _filter_split(split: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out = {name: indices[mask[indices]].astype(np.int64) for name, indices in split.items()}
    for name in ("train", "val", "test"):
        if len(out[name]) == 0:
            raise ValueError(f"{name} split is empty after wear_complete_mask")
    return out


def _eval_splits(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: split[name] for name in ("train", "val", "test")}


def _metric_aliases(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["pearson_r"] = result.get("raw_r")
    result["per_subject_r_mean"] = result.get("per_subject_r", {}).get("mean")
    result["per_subject_r_std"] = result.get("per_subject_r", {}).get("std")
    return result


def _selection_score(metrics: dict[str, Any], metric: str) -> float:
    if metric == "rmse":
        return -float(metrics["rmse"])
    if metric == "raw_r":
        return float(metrics["raw_r"])
    if metric == "centered_r":
        return float(metrics["within_subject_centered_r"])
    raise ValueError(f"unsupported selection metric: {metric}")


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _validate_routes(routes: tuple[str, ...]) -> None:
    unknown = set(routes) - set(DEFAULT_ROUTES)
    if unknown:
        raise ValueError(f"unsupported routes: {sorted(unknown)}")


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
