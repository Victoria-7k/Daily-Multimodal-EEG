"""Screen temporal, subject-day residual, and high-fatigue-day sampling ideas."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_BASELINE_ROOT = Path(
    "outputs/predictions/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw"
)
DEFAULT_OUT_ROOT = Path("outputs/server_sync/fatigue_calibration_20260816/phase4_next3_screen")

ROUTES = (
    ("cross_day", "B0_Wphysio_no_audio"),
    ("within_subject_day", "A2_Wdeep_full"),
)

VARIANTS = (
    "temporal_gru",
    "subject_day_mean_residual",
    "high_fatigue_day_oversample",
)


@dataclass
class SequenceBatch:
    x: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    raw_y: torch.Tensor
    valid: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--routes", default=",".join(f"{p}:{e}" for p, e in ROUTES))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seed", type=int, default=240800)
    parser.add_argument("--eeg-token-seed", type=int, default=240800)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    source = load_fusion_source()
    rows = source._load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([source._norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    window_id = np.asarray([int(row.get("event_window_id", row.get("window_id", 0))) for row in rows], dtype=np.int64)
    target = np.asarray([source._target_value(row, "fatigue") for row in rows], dtype=np.float32)
    requested = parse_routes(args.routes)
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unsupported = sorted(set(variants) - set(VARIANTS))
    if unsupported:
        raise ValueError(f"unsupported variants: {unsupported}")
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    run_count = 0
    branch_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for protocol, experiment in requested:
        split = source._load_split(args.splits_root / protocol, len(rows))
        branches = source._replace_eeg_branch(source.EXPERIMENT_BRANCHES[experiment], "eeg_eegpt_partial_ft_v1")
        branch_data = source._load_all_branches(
            args.embeddings_root,
            sample_id,
            protocol=protocol,
            eeg_seed=args.eeg_token_seed,
            eeg_token_root="eeg_encoder_256d_tokens",
            required_names=branches,
            cache=branch_cache,
        )
        tokens, token_mask, branch_report = source._build_tokens(branch_data, branches)
        sequences = {
            split_name: make_sequences(split_indices, subject_id, event_id, window_id)
            for split_name, split_indices in split.items()
            if split_name in {"train", "val", "test"}
        }
        leakage = sequence_leakage_report(sequences)
        baseline_npz = load_npz(args.baseline_root / protocol / "eeg_eegpt_partial_ft_v1" / experiment / "raw_lambda_0.npz")
        baseline_metrics = {name: metrics_for(*split_arrays(baseline_npz, name)) for name in ("val", "test")}
        for variant in variants:
            if args.limit_runs is not None and run_count >= args.limit_runs:
                break
            run_seed = int(args.seed) + run_count
            run_id = f"{variant}__{protocol}__{experiment}__seed_{run_seed}"
            print(f"starting run_id={run_id}", flush=True)
            model, train_audit = fit_variant(
                tokens=tokens,
                token_mask=token_mask,
                target=target,
                sequences=sequences,
                variant=variant,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                patience=args.patience,
                seed=run_seed,
                device=args.device,
            )
            predictions = {
                split_name: predict_split(model, tokens, token_mask, seqs, device=args.device)
                for split_name, seqs in sequences.items()
            }
            pred_dir = out_root / "predictions" / run_id / protocol / "eeg_eegpt_partial_ft_v1" / experiment
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"{variant}.npz"
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
            result: dict[str, Any] = {
                "run_id": run_id,
                "variant": variant,
                "protocol": protocol,
                "experiment": experiment,
                "seed": run_seed,
                "prediction_path": str(pred_path),
                "train_audit": train_audit,
                "sequence_preflight": leakage,
                "branch_report": branch_report,
            }
            for split_name in ("val", "test"):
                pred_metrics = metrics_for(predictions[split_name], target[split[split_name]], subject_id[split[split_name]])
                result[split_name] = pred_metrics
                deltas = delta_metrics(pred_metrics, baseline_metrics[split_name])
                record = {
                    "run_id": run_id,
                    "variant": variant,
                    "protocol": protocol,
                    "experiment": experiment,
                    "seed": run_seed,
                    "split": split_name,
                    **{f"{split_name}_{key}": value for key, value in pred_metrics.items()},
                    **{f"{split_name}_{key}": value for key, value in deltas.items()},
                    "prediction_path": str(pred_path),
                }
                metrics_rows.append(record)
            manifest.append(result)
            print(
                f"completed run_id={run_id} val_rmse={result['val']['rmse']:.4f} "
                f"val_raw_r={result['val']['raw_r']:.4f} val_centered_r={result['val']['within_subject_centered_r']:.4f} "
                f"val_std_ratio={result['val']['pred_std_over_true_std']:.4f}",
                flush=True,
            )
            run_count += 1
    write_json(out_root / "metrics_val_test.json", {"runs": manifest, "rows": metrics_rows})
    write_csv(out_root / "metrics_val_test.csv", metrics_rows)
    write_summary(out_root / "paired_delta_summary.md", metrics_rows)
    write_json(out_root / "sequence_preflight.json", {"runs": [{k: row[k] for k in ("run_id", "protocol", "experiment", "variant", "sequence_preflight")} for row in manifest]})
    print(json.dumps({"out_root": str(out_root), "run_count": run_count}, indent=2))
    return 0


def load_fusion_source() -> Any:
    path = Path(__file__).resolve().parent / "32_run_eegpt_centered_loss.py"
    spec = importlib.util.spec_from_file_location("run_eegpt_centered_loss_phase4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_routes(value: str) -> list[tuple[str, str]]:
    routes = []
    for item in value.split(","):
        protocol, experiment = item.strip().split(":", 1)
        routes.append((protocol, experiment))
    return routes


def make_sequences(indices: np.ndarray, subjects: np.ndarray, events: np.ndarray, windows: np.ndarray) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    index_set = np.asarray(indices, dtype=np.int64)
    for idx in index_set:
        key = f"{subjects[idx]}::{day_key(events[idx])}"
        groups.setdefault(key, []).append(int(idx))
    sequences = []
    for values in groups.values():
        ordered = sorted(values, key=lambda idx: (str(events[idx]), int(windows[idx]), int(idx)))
        sequences.append(np.asarray(ordered, dtype=np.int64))
    sequences.sort(key=lambda seq: int(seq[0]))
    return sequences


def day_key(event_id: str) -> str:
    for part in str(event_id).split("_"):
        if part.startswith("day-"):
            return part
    return str(event_id)


def sequence_leakage_report(sequences: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    sets = {name: set(int(v) for seq in seqs for v in seq.tolist()) for name, seqs in sequences.items()}
    overlaps = {
        f"{left}_{right}": len(sets[left] & sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {
        "sequence_counts": {name: len(seqs) for name, seqs in sequences.items()},
        "window_counts": {name: sum(len(seq) for seq in seqs) for name, seqs in sequences.items()},
        "index_overlaps": overlaps,
        "ok": all(value == 0 for value in overlaps.values()),
    }


class SequenceFusionModel(torch.nn.Module):
    def __init__(self, *, modality_count: int, hidden_dim: int, dropout: float, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.input_projection = torch.nn.Linear(256, hidden_dim)
        self.modality_embedding = torch.nn.Parameter(torch.zeros(1, 1, modality_count, hidden_dim))
        torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.modality_attention = torch.nn.MultiheadAttention(hidden_dim, 1, dropout=dropout, batch_first=True)
        self.query = torch.nn.Parameter(torch.zeros(hidden_dim))
        torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.dropout = torch.nn.Dropout(dropout)
        self.window_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.temporal_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.day_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.residual_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def encode_windows(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, modalities, dim = x.shape
        flat_x = x.reshape(batch * length, modalities, dim)
        flat_mask = mask.reshape(batch * length, modalities)
        projected = self.input_projection(flat_x) + self.modality_embedding.reshape(1, modalities, -1)
        attended, _ = self.modality_attention(projected, projected, projected, key_padding_mask=~flat_mask, need_weights=False)
        attended = self.dropout(attended)
        scores = torch.matmul(attended, self.query).masked_fill(~flat_mask, -1.0e9)
        weights = torch.softmax(scores, dim=1) * flat_mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        return pooled.reshape(batch, length, -1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_windows(x, mask)
        if self.variant == "temporal_gru":
            recurrent, _ = self.gru(encoded)
            pred = self.temporal_head(recurrent).squeeze(-1)
            return {"prediction": pred}
        if self.variant == "subject_day_mean_residual":
            weights = valid.to(dtype=encoded.dtype).unsqueeze(-1)
            context = torch.sum(encoded * weights, dim=1) / weights.sum(dim=1).clamp_min(1.0)
            day = self.day_head(context).squeeze(-1)
            expanded = context.unsqueeze(1).expand_as(encoded)
            residual = self.residual_head(torch.cat([encoded, expanded], dim=-1)).squeeze(-1)
            residual = residual - (residual * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1)
            return {"prediction": day.unsqueeze(1) + residual, "day": day, "residual": residual}
        pred = self.window_head(encoded).squeeze(-1)
        return {"prediction": pred}


def fit_variant(
    *,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    target: np.ndarray,
    sequences: dict[str, list[np.ndarray]],
    variant: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything(seed)
    train_idx = np.concatenate(sequences["train"])
    x_mean, x_std = fit_token_normalization(tokens, token_mask, train_idx)
    y_mean = float(target[train_idx].mean())
    y_std = float(target[train_idx].std()) or 1.0
    dev = torch.device(device)
    model = SequenceFusionModel(modality_count=tokens.shape[1], hidden_dim=hidden_dim, dropout=dropout, variant=variant).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    high_weights = sequence_sampling_weights(sequences["train"], target) if variant == "high_fatigue_day_oversample" else None
    for epoch in range(max(1, epochs)):
        model.train()
        train_losses = []
        for batch_sequences in iter_sequence_batches(sequences["train"], batch_size, rng, weights=high_weights):
            batch = make_batch(batch_sequences, tokens, token_mask, target, x_mean, x_std, y_mean, y_std, dev)
            outputs = model(batch.x, batch.mask, batch.valid)
            loss = variant_loss(outputs, batch, variant)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch_sequences in iter_sequence_batches(sequences["val"], batch_size, rng, shuffle=False):
                batch = make_batch(batch_sequences, tokens, token_mask, target, x_mean, x_std, y_mean, y_std, dev)
                val_losses.append(float(variant_loss(model(batch.x, batch.mask, batch.valid), batch, variant).detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)) if train_losses else math.nan, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "module": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, {
        "variant": variant,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "epoch_count": len(history),
        "normalization": "train_only",
        "history": history,
    }


def variant_loss(outputs: dict[str, torch.Tensor], batch: SequenceBatch, variant: str) -> torch.Tensor:
    pred = outputs["prediction"]
    residual = (pred - batch.y) * batch.valid
    raw_loss = torch.sum(residual * residual) / batch.valid.sum().clamp_min(1)
    if variant == "subject_day_mean_residual":
        day_target = torch.sum(batch.y * batch.valid, dim=1) / batch.valid.sum(dim=1).clamp_min(1)
        day_loss = torch.mean((outputs["day"] - day_target) ** 2)
        centered_target = batch.y - day_target.unsqueeze(1)
        centered_residual = (outputs["residual"] - centered_target) * batch.valid
        centered_loss = torch.sum(centered_residual * centered_residual) / batch.valid.sum().clamp_min(1)
        return raw_loss + 0.2 * day_loss + 0.2 * centered_loss
    if variant == "high_fatigue_day_oversample":
        high = (batch.raw_y >= 4.0).to(dtype=batch.y.dtype)
        weights = torch.where(high > 0, torch.full_like(batch.y, 2.0), torch.ones_like(batch.y))
        return torch.sum(weights * batch.valid * residual * residual) / torch.sum(weights * batch.valid).clamp_min(1)
    return raw_loss


def fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, None], tokens[indices], np.nan)
    mean = np.nanmean(available, axis=(0, 1), keepdims=True)
    std = np.nanstd(available, axis=(0, 1), keepdims=True)
    return np.where(np.isfinite(mean), mean, 0.0).astype(np.float32), np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)


def normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def sequence_sampling_weights(sequences: list[np.ndarray], target: np.ndarray) -> np.ndarray:
    weights = []
    for seq in sequences:
        raw = target[seq]
        if float(np.max(raw)) >= 4.0:
            weights.append(4.0)
        elif float(np.mean(raw)) >= 3.5:
            weights.append(2.0)
        else:
            weights.append(1.0)
    values = np.asarray(weights, dtype=np.float64)
    return values / values.sum()


def iter_sequence_batches(
    sequences: list[np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
    *,
    weights: np.ndarray | None = None,
    shuffle: bool = True,
) -> list[list[np.ndarray]]:
    if not sequences:
        return []
    if weights is not None:
        count = len(sequences)
        order = rng.choice(np.arange(count), size=count, replace=True, p=weights)
    else:
        order = np.arange(len(sequences))
        if shuffle:
            rng.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        batches.append([sequences[int(idx)] for idx in order[start : start + batch_size]])
    return batches


def make_batch(
    sequences: list[np.ndarray],
    tokens: np.ndarray,
    token_mask: np.ndarray,
    target: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    device: torch.device,
) -> SequenceBatch:
    max_len = max(len(seq) for seq in sequences)
    batch = len(sequences)
    modality_count = tokens.shape[1]
    x = np.zeros((batch, max_len, modality_count, tokens.shape[2]), dtype=np.float32)
    m = np.zeros((batch, max_len, modality_count), dtype=bool)
    y = np.zeros((batch, max_len), dtype=np.float32)
    raw_y = np.zeros((batch, max_len), dtype=np.float32)
    valid = np.zeros((batch, max_len), dtype=bool)
    for row, seq in enumerate(sequences):
        length = len(seq)
        x[row, :length] = normalize_tokens(tokens[seq], x_mean, x_std)
        m[row, :length] = token_mask[seq]
        raw = target[seq].astype(np.float32)
        raw_y[row, :length] = raw
        y[row, :length] = (raw - y_mean) / y_std
        valid[row, :length] = True
    return SequenceBatch(
        x=torch.as_tensor(x, dtype=torch.float32, device=device),
        mask=torch.as_tensor(m, dtype=torch.bool, device=device),
        y=torch.as_tensor(y, dtype=torch.float32, device=device),
        raw_y=torch.as_tensor(raw_y, dtype=torch.float32, device=device),
        valid=torch.as_tensor(valid, dtype=torch.float32, device=device),
    )


def predict_split(model: dict[str, Any], tokens: np.ndarray, token_mask: np.ndarray, sequences: list[np.ndarray], *, device: str) -> np.ndarray:
    module: SequenceFusionModel = model["module"]
    dev = torch.device(device)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_sequences in iter_sequence_batches(sequences, 16, np.random.default_rng(1), shuffle=False):
            batch = make_batch(
                batch_sequences,
                tokens,
                token_mask,
                np.zeros(len(tokens), dtype=np.float32),
                model["x_mean"],
                model["x_std"],
                0.0,
                1.0,
                dev,
            )
            outputs = module(batch.x, batch.mask, batch.valid)
            pred = outputs["prediction"].detach().cpu().numpy()
            for row, seq in enumerate(batch_sequences):
                predictions.append((pred[row, : len(seq)] * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(predictions) if predictions else np.zeros((0,), dtype=np.float32)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(openable_path(path), allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def openable_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def split_arrays(data: dict[str, np.ndarray], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = data[f"{split}_index"].astype(np.int64)
    pred = data[f"{split}_prediction"].astype(np.float64)
    target = data["target"][index].astype(np.float64)
    subject = data["subject_id"][index].astype(str)
    return pred, target, subject


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def centered_r(pred: np.ndarray, target: np.ndarray, subject: np.ndarray) -> float:
    pred_centered = pred.astype(np.float64).copy()
    target_centered = target.astype(np.float64).copy()
    for sid in np.unique(subject):
        mask = subject == sid
        pred_centered[mask] -= float(np.mean(pred_centered[mask]))
        target_centered[mask] -= float(np.mean(target_centered[mask]))
    return pearson(pred_centered, target_centered)


def label_masks(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = target <= 2.0
    high = target >= 4.0
    return low, high


def metrics_for(pred: np.ndarray, target: np.ndarray, subject: np.ndarray) -> dict[str, float]:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    err = pred - target
    low, high = label_masks(target)
    pred_std = float(np.std(pred))
    true_std = float(np.std(target))
    return {
        "count": int(len(target)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "raw_r": pearson(pred, target),
        "within_subject_centered_r": centered_r(pred, target, subject),
        "pred_std": pred_std,
        "true_std": true_std,
        "pred_std_over_true_std": float(pred_std / true_std) if true_std else float("nan"),
        "low_fatigue_bias": float(np.mean(err[low])) if np.any(low) else float("nan"),
        "high_fatigue_bias": float(np.mean(err[high])) if np.any(high) else float("nan"),
        "high_fatigue_mae": float(np.mean(np.abs(err[high]))) if np.any(high) else float("nan"),
    }


def delta_metrics(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    high_base = abs(float(baseline["high_fatigue_bias"]))
    high_new = abs(float(candidate["high_fatigue_bias"]))
    high_mae_base = float(baseline["high_fatigue_mae"])
    high_mae_new = float(candidate["high_fatigue_mae"])
    return {
        "rmse_delta": float(candidate["rmse"] - baseline["rmse"]),
        "mae_delta": float(candidate["mae"] - baseline["mae"]),
        "raw_r_delta": float(candidate["raw_r"] - baseline["raw_r"]),
        "centered_r_delta": float(candidate["within_subject_centered_r"] - baseline["within_subject_centered_r"]),
        "std_ratio_delta": float(candidate["pred_std_over_true_std"] - baseline["pred_std_over_true_std"]),
        "high_abs_bias_reduction": float((high_base - high_new) / high_base) if high_base else float("nan"),
        "high_mae_reduction": float((high_mae_base - high_mae_new) / high_mae_base) if high_mae_base else float("nan"),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    val_rows = [row for row in rows if row["split"] == "val"]
    keys = [
        "variant",
        "protocol",
        "experiment",
        "val_rmse_delta",
        "val_mae_delta",
        "val_raw_r_delta",
        "val_centered_r_delta",
        "val_std_ratio_delta",
        "val_high_abs_bias_reduction",
        "val_high_mae_reduction",
    ]
    lines = [
        "# Next-3 Calibration Screen",
        "",
        "Small screen for temporal GRU, subject-day mean+residual, and high-fatigue-day oversampling. Values are val deltas against the same-route Phase 0 raw baseline.",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in val_rows:
        lines.append("| " + " | ".join(format_cell(row.get(key, "")) for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "nan"
    return str(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
