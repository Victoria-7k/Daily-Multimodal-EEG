#!/usr/bin/env python3
"""Run the EEGPT-aligned raw/centered multi-task loss experiment matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered


LABEL_NAMES = [
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
]
DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_EXPERIMENTS = (
    "cross_day:A1_Wphysio_no_audio",
    "within_subject_day:A1_Wphysio_no_audio",
    "within_subject_day:B0_Wdeep_no_audio",
)


@dataclass(frozen=True)
class Branch:
    name: str
    modality: str
    filename: str
    emb_key: str
    mask_key: str
    modality_index: int


BRANCHES = {
    "eeg": Branch("eeg", "eeg", "eeg/eeg_eegpt_eeg23win_embeddings.npz", "eeg_emb", "eeg_mask", 0),
    "wear_physio": Branch("wear_physio", "wear", "wear/wear_physio_preprocessed_eeg23win_embeddings.npz", "wear_emb", "wear_mask", 1),
    "wear_deep": Branch("wear_deep", "wear", "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz", "wear_emb", "wear_mask", 1),
    "video_B0": Branch("video_B0", "video", "video/video_B0_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A1": Branch("video_A1", "video", "video/video_A1_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A2": Branch("video_A2", "video", "video/video_A2_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "audio": Branch("audio", "audio", "audio/audio_opensmile_eeg23win_embeddings.npz", "audio_emb", "audio_mask", 3),
}

EXPERIMENT_BRANCHES = {
    "B0_Wphysio_full": ("eeg", "wear_physio", "video_B0", "audio"),
    "B0_Wphysio_no_audio": ("eeg", "wear_physio", "video_B0"),
    "B0_Wphysio_no_video": ("eeg", "wear_physio", "audio"),
    "B0_Wphysio_bio_only": ("eeg", "wear_physio"),
    "B0_Wdeep_full": ("eeg", "wear_deep", "video_B0", "audio"),
    "B0_Wdeep_no_audio": ("eeg", "wear_deep", "video_B0"),
    "B0_Wdeep_no_video": ("eeg", "wear_deep", "audio"),
    "B0_Wdeep_bio_only": ("eeg", "wear_deep"),
    "A1_Wphysio_full": ("eeg", "wear_physio", "video_A1", "audio"),
    "A1_Wphysio_no_audio": ("eeg", "wear_physio", "video_A1"),
    "A1_Wdeep_full": ("eeg", "wear_deep", "video_A1", "audio"),
    "A1_Wdeep_no_audio": ("eeg", "wear_deep", "video_A1"),
    "A2_Wphysio_full": ("eeg", "wear_physio", "video_A2", "audio"),
    "A2_Wphysio_no_audio": ("eeg", "wear_physio", "video_A2"),
    "A2_Wdeep_full": ("eeg", "wear_deep", "video_A2", "audio"),
    "A2_Wdeep_no_audio": ("eeg", "wear_deep", "video_A2"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS), help="protocol:experiment pairs")
    parser.add_argument("--loss-modes", default="raw_centered_mse,raw_centered_corr")
    parser.add_argument("--lambdas", default="0.1,0.3,0.5,1.0")
    parser.add_argument("--no-raw-baseline", action="store_true")
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=240729)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subject-balanced-batches", action="store_true")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    rows = _load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([_norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    target = np.asarray([_target_value(row, args.target_label) for row in rows], dtype=np.float32)
    branch_data = _load_all_branches(args.embeddings_root, sample_id)
    requested = _parse_experiments(args.experiments)
    loss_modes = [value.strip() for value in args.loss_modes.split(",") if value.strip()]
    lambdas = [float(value.strip()) for value in args.lambdas.split(",") if value.strip()]
    for mode in loss_modes:
        if mode not in {"raw", "raw_centered_mse", "raw_centered_corr"}:
            raise ValueError(f"unsupported centered loss mode: {mode}")
    results: list[dict[str, Any]] = []
    run_number = 0
    for protocol, experiment in requested:
        branches = EXPERIMENT_BRANCHES[experiment]
        split = _load_split(args.splits_root / protocol, len(rows))
        tokens, token_mask, branch_report = _build_tokens(branch_data, branches)
        run_specs: list[tuple[str, float]] = []
        if not args.no_raw_baseline and "raw" not in loss_modes:
            run_specs.append(("raw", 0.0))
        for mode in loss_modes:
            if mode == "raw":
                run_specs.append(("raw", 0.0))
            else:
                run_specs.extend((mode, value) for value in lambdas)
        for loss_mode, centered_lambda in run_specs:
            run_seed = int(args.seed) + run_number
            run_number += 1
            print(
                f"starting protocol={protocol} experiment={experiment} "
                f"loss_mode={loss_mode} lambda={centered_lambda} seed={run_seed}",
                flush=True,
            )
            model, train_audit = _fit_model(
                tokens=tokens,
                token_mask=token_mask,
                target=target,
                subjects=subject_id,
                train_idx=split["train"],
                val_idx=split["val"],
                loss_mode=loss_mode,
                centered_lambda=centered_lambda,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                patience=args.patience,
                seed=run_seed,
                device=args.device,
                subject_balanced_batches=args.subject_balanced_batches,
            )
            predictions = {
                name: _predict(model, tokens, token_mask, indices=indices, device=args.device)
                for name, indices in split.items()
                if name in {"train", "val", "test"}
            }
            test_metrics = _metric_aliases(
                evaluate_regression_with_centered(
                    target[split["test"]], predictions["test"], subject_id[split["test"]]
                )
            )
            result = {
                "protocol": protocol,
                "experiment": experiment,
                "branches": list(branches),
                "enabled_modalities": [BRANCHES[name].modality for name in branches],
                "loss_mode": loss_mode,
                "centered_lambda": float(centered_lambda),
                "seed": run_seed,
                "row_count": len(rows),
                "target_label": args.target_label,
                "split_counts": {name: int(len(values)) for name, values in split.items()},
                "mask_coverage_by_split": _mask_coverage(token_mask, branches, split),
                "train": _metric_aliases(evaluate_regression_with_centered(target[split["train"]], predictions["train"], subject_id[split["train"]])),
                "val": _metric_aliases(evaluate_regression_with_centered(target[split["val"]], predictions["val"], subject_id[split["val"]])),
                "test": test_metrics,
                "train_raw_loss": train_audit.get("best_train_raw_loss"),
                "train_centered_loss": train_audit.get("best_train_centered_loss"),
                "batch_centered_subject_count_mean": train_audit.get("batch_centered_subject_count_mean"),
                "train_audit": train_audit,
                "branch_report": branch_report,
            }
            if args.predictions_dir:
                pred_path = args.predictions_dir / protocol / experiment / f"{loss_mode}_lambda_{centered_lambda:g}.npz"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    pred_path,
                    train_index=split["train"],
                    val_index=split["val"],
                    test_index=split["test"],
                    train_prediction=predictions["train"],
                    val_prediction=predictions["val"],
                    test_prediction=predictions["test"],
                    target=target,
                    sample_id=sample_id,
                    subject_id=subject_id,
                    event_id=event_id,
                )
                result["prediction_path"] = str(pred_path)
            results.append(result)
            print(
                f"completed protocol={protocol} experiment={experiment} loss_mode={loss_mode} "
                f"lambda={centered_lambda} rmse={_fmt(test_metrics['rmse'])} "
                f"raw_r={_fmt(test_metrics['raw_r'])} centered_r={_fmt(test_metrics['within_subject_centered_r'])}",
                flush=True,
            )

    output = {
        "stage": 2,
        "target_label": args.target_label,
        "root": str(args.root),
        "splits_root": str(args.splits_root),
        "embeddings_root": str(args.embeddings_root),
        "runtime": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "patience": args.patience,
            "seed": args.seed,
            "device": args.device,
            "subject_balanced_batches": args.subject_balanced_batches,
            "train_rule": "pretrain + finetune from splits_new",
            "normalization": "train_only",
        },
        "run_count": len(results),
        "results": results,
    }
    _write_json(output, args.out_json)
    _write_markdown(output, args.out_md)
    print(f"run_count={len(results)}")
    print(f"out_json={args.out_json}")
    print(f"out_md={args.out_md}")
    return 0


def _fit_model(
    *,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    loss_mode: str,
    centered_lambda: float,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    seed: int,
    device: str,
    subject_balanced_batches: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    x_mean, x_std = _fit_token_normalization(tokens, token_mask, train_idx)
    y_mean = float(target[train_idx].mean())
    y_std = float(target[train_idx].std()) or 1.0
    dev = torch.device(device)
    module = AttentionRegressor(modality_count=tokens.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train = torch.as_tensor(_normalize_tokens(tokens[train_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_train = torch.as_tensor(token_mask[train_idx], dtype=torch.bool, device=dev)
    y_train = torch.as_tensor((target[train_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    train_subjects = subjects[train_idx]
    x_val = torch.as_tensor(_normalize_tokens(tokens[val_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_val = torch.as_tensor(token_mask[val_idx], dtype=torch.bool, device=dev)
    y_val = torch.as_tensor((target[val_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    val_subjects = subjects[val_idx]
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    epoch_audits: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(max(1, epochs)):
        module.train()
        batch_losses: list[float] = []
        batch_raw_losses: list[float] = []
        batch_centered_losses: list[float] = []
        batch_subject_counts: list[int] = []
        for batch in _make_batches(train_subjects, batch_size, rng, subject_balanced_batches):
            prediction = module(x_train[batch], m_train[batch])
            raw_loss, centered_loss, eligible_subject_count = _loss_components(
                prediction, y_train[batch], train_subjects[batch], loss_mode
            )
            loss = raw_loss + float(centered_lambda) * centered_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
            batch_raw_losses.append(float(raw_loss.detach().cpu().item()))
            batch_centered_losses.append(float(centered_loss.detach().cpu().item()))
            batch_subject_counts.append(eligible_subject_count)
        module.eval()
        with torch.no_grad():
            val_prediction = module(x_val, m_val)
            val_raw, val_centered, val_subject_count = _loss_components(val_prediction, y_val, val_subjects, loss_mode)
            val_loss = float((val_raw + float(centered_lambda) * val_centered).detach().cpu().item())
        audit = {
            "train_loss": float(np.mean(batch_losses)) if batch_losses else math.nan,
            "train_raw_loss": float(np.mean(batch_raw_losses)) if batch_raw_losses else math.nan,
            "train_centered_loss": float(np.mean(batch_centered_losses)) if batch_centered_losses else math.nan,
            "batch_centered_subject_count_mean": float(np.mean(batch_subject_counts)) if batch_subject_counts else 0.0,
            "val_loss": val_loss,
            "val_raw_loss": float(val_raw.detach().cpu().item()),
            "val_centered_loss": float(val_centered.detach().cpu().item()),
            "val_centered_subject_count": int(val_subject_count),
        }
        epoch_audits.append(audit)
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
    best_audit = epoch_audits[max(0, best_epoch - 1)] if epoch_audits else {}
    module.eval()
    return {
        "module": module,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "initial_train_loss": epoch_audits[0]["train_loss"] if epoch_audits else math.nan,
        "final_train_loss": epoch_audits[-1]["train_loss"] if epoch_audits else math.nan,
        "best_train_raw_loss": best_audit.get("train_raw_loss"),
        "best_train_centered_loss": best_audit.get("train_centered_loss"),
        "batch_centered_subject_count_mean": best_audit.get("batch_centered_subject_count_mean", 0.0),
        "normalization": "train_only",
        "train_count": int(len(train_idx)),
        "loss_mode": loss_mode,
        "centered_lambda": float(centered_lambda),
        "epoch_count": len(epoch_audits),
    }


class AttentionRegressor(torch.nn.Module):
    def __init__(self, *, modality_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = torch.nn.Linear(256, hidden_dim)
        self.modality_embedding = torch.nn.Parameter(torch.zeros(1, modality_count, hidden_dim))
        torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, 1, dropout=dropout, batch_first=True)
        self.query = torch.nn.Parameter(torch.zeros(hidden_dim))
        torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(tokens) + self.modality_embedding
        attended, _ = self.self_attention(x, x, x, key_padding_mask=~mask, need_weights=False)
        attended = self.dropout(attended)
        scores = torch.matmul(attended, self.query).masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).reshape(-1)


def _loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    subjects: np.ndarray,
    loss_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    raw_loss = torch.mean((prediction - target) ** 2)
    if loss_mode == "raw":
        return raw_loss, torch.zeros_like(raw_loss), 0
    subject_values = np.asarray(subjects).reshape(-1)
    centered_prediction: list[torch.Tensor] = []
    centered_target: list[torch.Tensor] = []
    eligible = 0
    for subject in np.unique(subject_values):
        indices = np.flatnonzero(subject_values == subject)
        if len(indices) < 2:
            continue
        index = torch.as_tensor(indices, dtype=torch.long, device=prediction.device)
        centered_prediction.append(prediction[index] - prediction[index].mean())
        centered_target.append(target[index] - target[index].mean())
        eligible += 1
    if not centered_prediction:
        return raw_loss, torch.zeros_like(raw_loss), eligible
    pred = torch.cat(centered_prediction)
    truth = torch.cat(centered_target)
    if loss_mode == "raw_centered_mse":
        centered_loss = torch.mean((pred - truth) ** 2)
    elif loss_mode == "raw_centered_corr":
        denominator = torch.sqrt(torch.sum(pred * pred) * torch.sum(truth * truth)).clamp_min(1e-8)
        centered_loss = 1.0 - torch.sum(pred * truth) / denominator
    else:
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    return raw_loss, centered_loss, eligible


def _make_batches(subjects: np.ndarray, batch_size: int, rng: np.random.Generator, balanced: bool) -> list[np.ndarray]:
    if not balanced:
        order = rng.permutation(len(subjects))
        return [order[start : start + max(1, batch_size)] for start in range(0, len(order), max(1, batch_size))]
    queues = [list(rng.permutation(np.flatnonzero(subjects == subject)).tolist()) for subject in np.unique(subjects)]
    order: list[int] = []
    while any(queues):
        subject_order = rng.permutation(len(queues)).tolist()
        for queue_index in subject_order:
            if queues[queue_index]:
                order.append(queues[queue_index].pop())
    return [np.asarray(order[start : start + max(1, batch_size)], dtype=np.int64) for start in range(0, len(order), max(1, batch_size))]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _target_value(row: dict[str, Any], target: str) -> float:
    labels = row.get("labels")
    if isinstance(labels, dict):
        return float(labels[target])
    if isinstance(labels, list):
        return float(labels[LABEL_NAMES.index(target)])
    return float(row[target])


def _load_all_branches(embedding_root: Path, sample_id: np.ndarray) -> dict[str, dict[str, Any]]:
    result = {}
    for name, branch in BRANCHES.items():
        path = embedding_root / branch.filename
        with np.load(path, allow_pickle=True) as loaded:
            loaded_ids = loaded["sample_id"].astype(str)
            if not np.array_equal(loaded_ids, sample_id):
                raise ValueError(f"{name} sample_id order does not match canonical index")
            emb_key = branch.emb_key if branch.emb_key in loaded.files else "face_emb"
            embedding = loaded[emb_key].astype(np.float32)
            mask = loaded[branch.mask_key].astype(bool) if branch.mask_key in loaded.files else loaded["modality_mask"][:, branch.modality_index].astype(bool)
            if embedding.shape != (len(sample_id), 256) or mask.shape != (len(sample_id),):
                raise ValueError(f"invalid shape for {name}: {embedding.shape}, {mask.shape}")
        result[name] = {"embedding": embedding, "mask": mask, "path": str(path), "mask_sum": int(mask.sum())}
    return result


def _build_tokens(data: dict[str, dict[str, Any]], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tokens = np.stack([data[name]["embedding"] for name in names], axis=1).astype(np.float32)
    masks = np.stack([data[name]["mask"] for name in names], axis=1).astype(bool)
    report = {name: {"path": data[name]["path"], "mask_sum": data[name]["mask_sum"], "modality": BRANCHES[name].modality} for name in names}
    return tokens, masks, report


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


def _predict(model: dict[str, Any], tokens: np.ndarray, mask: np.ndarray, *, indices: np.ndarray, device: str) -> np.ndarray:
    module: AttentionRegressor = model["module"]
    device_obj = torch.device(device)
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            chunk = indices[start : start + 1024]
            x = torch.as_tensor(_normalize_tokens(tokens[chunk], model["x_mean"], model["x_std"]), dtype=torch.float32, device=device_obj)
            m = torch.as_tensor(mask[chunk], dtype=torch.bool, device=device_obj)
            values.append((module(x, m).detach().cpu().numpy() * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, None], tokens[indices], np.nan)
    mean = np.nanmean(available, axis=(0, 1), keepdims=True)
    std = np.nanstd(available, axis=(0, 1), keepdims=True)
    return np.where(np.isfinite(mean), mean, 0.0).astype(np.float32), np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)


def _normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def _mask_coverage(mask: np.ndarray, branches: tuple[str, ...], split: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            branch: {"valid": int(mask[indices, col].sum()), "total": int(len(indices)), "coverage": float(mask[indices, col].mean()) if len(indices) else 0.0}
            for col, branch in enumerate(branches)
        }
        for name, indices in split.items()
        if name in {"train", "val", "test"}
    }


def _metric_aliases(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["pearson_r"] = result.get("raw_r")
    result["per_subject_r_mean"] = result.get("per_subject_r", {}).get("mean")
    result["per_subject_r_std"] = result.get("per_subject_r", {}).get("std")
    return result


def _parse_experiments(value: str) -> list[tuple[str, str]]:
    items = []
    for raw in value.split(","):
        protocol, experiment = raw.strip().split(":", 1)
        if experiment not in EXPERIMENT_BRANCHES:
            raise ValueError(f"unsupported experiment: {experiment}")
        items.append((protocol, experiment))
    return items


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# EEGPT Raw + Centered Multi-Task Loss",
        "",
        f"target_label: `{output['target_label']}`",
        f"run_count: `{output['run_count']}`",
        "",
        "| protocol | experiment | loss | lambda | RMSE | MAE | raw r | centered r | per-subject r mean | best epoch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        test = row["test"]
        lines.append(
            f"| {row['protocol']} | {row['experiment']} | {row['loss_mode']} | {row['centered_lambda']:.3g} | "
            f"{_fmt(test['rmse'])} | {_fmt(test['mae'])} | {_fmt(test['raw_r'])} | "
            f"{_fmt(test['within_subject_centered_r'])} | {_fmt(test['per_subject_r_mean'])} | {row['train_audit']['best_epoch']} |"
        )
    lines.extend(["", "`centered_loss` is computed within each training batch for subjects with at least two samples; validation selection uses the same composite objective.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
