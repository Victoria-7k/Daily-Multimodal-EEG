"""Event-supervised EEGPT fine-tuning for single- and multi-label affect tasks."""

from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from daily_multimodal.training.centered_metrics import safe_pearsonr, within_subject_centered_arrays
from daily_multimodal.training.eeg_encoder_matrix import (
    EEGAlignedDataset,
    SplitProtocol,
    _fit_channel_normalization,
    _torch_eeg_batch,
    build_torch_encoder,
)


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
class EEGSupervisedRuntime:
    epochs: int = 80
    hidden_dim: int = 128
    batch_size: int = 256
    fallback_batch_size: int = 64
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 15
    grad_clip: float = 1.0
    partial_last_n_blocks: int = 2
    device: str = "cuda"
    amp: bool = True
    torch_threads: int = 4


def load_event_metadata(index_path: Path | str, expected_rows: int) -> dict[str, np.ndarray]:
    rows = []
    with Path(index_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != expected_rows:
        raise ValueError(f"index row count {len(rows)} != EEG row count {expected_rows}")
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    if np.any(event_id == ""):
        raise ValueError("canonical index contains empty event_id")
    return {
        "event_id": event_id,
        "subject_id": np.asarray([str(row.get("subject_id", "")) for row in rows], dtype=str),
        "day_id": np.asarray([str(row.get("day_id", "")) for row in rows], dtype=str),
        "sample_id": np.asarray([str(row.get("sample_id", "")) for row in rows], dtype=str),
    }


def target_matrix(dataset: EEGAlignedDataset, label_names: Sequence[str]) -> np.ndarray:
    labels = np.asarray(dataset.y)
    if labels.ndim != 2:
        raise ValueError(f"multi-emotion EEG training requires 2D y, got {labels.shape}")
    indices = []
    for label in label_names:
        if label not in LABEL_NAMES:
            raise ValueError(f"unknown target label: {label}")
        indices.append(LABEL_NAMES.index(label))
    if labels.shape[1] <= max(indices):
        raise ValueError(f"target columns {labels.shape[1]} cannot provide {tuple(label_names)}")
    values = np.asarray(labels[:, indices], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("target matrix contains non-finite values")
    return values


def event_equal_window_weights(event_id: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(event_id.astype(str), return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float32)


def aggregate_events(
    target: np.ndarray,
    prediction: np.ndarray,
    event_id: np.ndarray,
    subject_id: np.ndarray,
    day_id: np.ndarray,
) -> dict[str, np.ndarray]:
    unique, inverse = np.unique(event_id.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    y_sum = np.zeros((len(unique), target.shape[1]), dtype=np.float64)
    p_sum = np.zeros_like(y_sum)
    np.add.at(y_sum, inverse, target)
    np.add.at(p_sum, inverse, prediction)
    event_subject = np.empty(len(unique), dtype=object)
    event_day = np.empty(len(unique), dtype=object)
    for idx in range(len(unique)):
        members = np.flatnonzero(inverse == idx)
        subjects = np.unique(subject_id[members].astype(str))
        days = np.unique(day_id[members].astype(str))
        if len(subjects) != 1 or len(days) != 1:
            raise ValueError(f"event {unique[idx]} crosses subject/day")
        if not np.allclose(target[members], target[members[0]], atol=1e-6):
            raise ValueError(f"event {unique[idx]} has inconsistent targets")
        event_subject[idx] = subjects[0]
        event_day[idx] = days[0]
    return {
        "event_id": unique.astype(str),
        "subject_id": event_subject.astype(str),
        "day_id": event_day.astype(str),
        "target": (y_sum / counts[:, None]).astype(np.float32),
        "prediction": (p_sum / counts[:, None]).astype(np.float32),
        "window_count": counts.astype(np.int64),
    }


def event_macro_standardized_rmse(
    target_standardized: np.ndarray, prediction_standardized: np.ndarray, event_id: np.ndarray
) -> float:
    dummy = np.asarray(["x"] * len(event_id), dtype=str)
    arrays = aggregate_events(target_standardized, prediction_standardized, event_id, dummy, dummy)
    error = arrays["prediction"] - arrays["target"]
    return float(np.mean(np.sqrt(np.mean(error * error, axis=0))))


def evaluate_events(arrays: dict[str, np.ndarray], label_names: Sequence[str], train_y_std: np.ndarray) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(label_names):
        true = arrays["target"][:, index]
        pred = arrays["prediction"][:, index]
        error = pred - true
        centered_true, centered_pred = within_subject_centered_arrays(true, pred, arrays["subject_id"])
        per_label[label] = {
            "count": int(len(true)),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "standardized_rmse": float(np.sqrt(np.mean(error * error)) / float(train_y_std[0, index])),
            "mae": float(np.mean(np.abs(error))),
            "raw_r": safe_pearsonr(true, pred),
            "within_subject_centered_r": safe_pearsonr(centered_true, centered_pred),
            "label_std": float(np.std(true)),
            "prediction_std": float(np.std(pred)),
        }
    metrics = ("rmse", "standardized_rmse", "mae", "raw_r", "within_subject_centered_r")
    summary = {
        metric: float(np.mean([row[metric] for row in per_label.values() if row[metric] is not None]))
        for metric in metrics
    }
    return {"per_label": per_label, "summary": summary}


def run_eeg_supervised(
    *,
    dataset: EEGAlignedDataset,
    split: SplitProtocol,
    metadata: dict[str, np.ndarray],
    label_names: Sequence[str],
    protocol: str,
    seed: int,
    checkpoint: Path | str,
    runtime: EEGSupervisedRuntime,
    input_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(max(1, int(runtime.torch_threads)))
    if runtime.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    try:
        return _run_once(
            dataset=dataset,
            split=split,
            metadata=metadata,
            label_names=tuple(label_names),
            protocol=protocol,
            seed=seed,
            checkpoint=checkpoint,
            runtime=runtime,
            batch_size=runtime.batch_size,
            accumulation=1,
            torch=torch,
            input_cache=input_cache,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() and "oom" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        fallback = max(1, int(runtime.fallback_batch_size))
        accumulation = max(1, math.ceil(runtime.batch_size / fallback))
        result = _run_once(
            dataset=dataset,
            split=split,
            metadata=metadata,
            label_names=tuple(label_names),
            protocol=protocol,
            seed=seed,
            checkpoint=checkpoint,
            runtime=runtime,
            batch_size=fallback,
            accumulation=accumulation,
            torch=torch,
            input_cache=input_cache,
        )
        result["oom_recovered"] = True
        return result


def _run_once(
    *,
    dataset: EEGAlignedDataset,
    split: SplitProtocol,
    metadata: dict[str, np.ndarray],
    label_names: tuple[str, ...],
    protocol: str,
    seed: int,
    checkpoint: Path | str,
    runtime: EEGSupervisedRuntime,
    batch_size: int,
    accumulation: int,
    torch: Any,
    input_cache: dict[str, Any] | None,
) -> dict[str, Any]:
    started = time.time()
    _seed_everything(seed, torch)
    device = torch.device(runtime.device)
    encoder, backend_report = build_torch_encoder(
        profile="eegpt_partial_ft_v1",
        n_channels=dataset.channel_count,
        n_times=dataset.sample_count,
        sample_rate_hz=200.0,
        cbramod_checkpoint=None,
        eegpt_checkpoint=checkpoint,
        allow_cbramod_download=False,
        torch=torch,
    )
    trainability = configure_strict_last_blocks(
        encoder, last_n_blocks=int(runtime.partial_last_n_blocks)
    )
    model = EEGSupervisedRegressor(
        encoder=encoder,
        label_names=label_names,
        hidden_dim=runtime.hidden_dim,
        dropout=runtime.dropout,
        torch=torch,
    ).to(device)
    if input_cache is None:
        channel_mean, channel_std = _fit_channel_normalization(dataset.x, split.train)
        cached_x = None
    else:
        channel_mean = input_cache["channel_mean"]
        channel_std = input_cache["channel_std"]
        cached_x = input_cache["tensor"]
    targets = target_matrix(dataset, label_names)
    y_mean = targets[split.train].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = targets[split.train].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(y_std >= 1e-6, y_std, 1.0).astype(np.float32)
    weights = event_equal_window_weights(metadata["event_id"][split.train])
    encoder_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder_params if name.startswith("encoder.") else head_params).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": runtime.encoder_learning_rate},
            {"params": head_params, "lr": runtime.head_learning_rate},
        ],
        weight_decay=runtime.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(runtime.amp and device.type == "cuda"))
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(max(1, runtime.epochs)):
        model.train()
        order = rng.permutation(len(split.train))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        for offset, positions in enumerate(batches):
            indices = split.train[positions]
            batch_x = (
                cached_x[torch.as_tensor(indices, dtype=torch.long, device=device)]
                if cached_x is not None
                else _torch_eeg_batch(dataset.x, indices, channel_mean, channel_std, torch=torch, device=device)
            )
            batch_y = torch.as_tensor(
                (targets[indices] - y_mean) / y_std, dtype=torch.float32, device=device
            )
            batch_w = torch.as_tensor(weights[positions], dtype=torch.float32, device=device)
            with torch.cuda.amp.autocast(enabled=bool(runtime.amp and device.type == "cuda")):
                prediction = model(batch_x)
                per_window = torch.mean((prediction - batch_y) ** 2, dim=1)
                loss = (torch.sum(per_window * batch_w) / torch.clamp(torch.sum(batch_w), min=1e-8)) / accumulation
            scaler.scale(loss).backward()
            boundary = (offset + 1) % accumulation == 0 or offset + 1 == len(batches)
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()) * accumulation)
        val_standardized = _predict_standardized(
            model, dataset.x, split.val, channel_mean, channel_std, batch_size=512, torch=torch, device=device,
            cached_x=cached_x,
        )
        val_target = ((targets[split.val] - y_mean) / y_std).astype(np.float32)
        val_metric = event_macro_standardized_rmse(
            val_target, val_standardized, metadata["event_id"][split.val]
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_event_macro_standardized_rmse": val_metric,
        })
        if val_metric < best_val:
            best_val = val_metric
            best_epoch = epoch + 1
            stale = 0
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= runtime.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    predictions = {}
    event_predictions = {}
    metrics = {}
    for leaf in ("val", "test"):
        standardized = _predict_standardized(
            model, dataset.x, getattr(split, leaf), channel_mean, channel_std, batch_size=512, torch=torch, device=device,
            cached_x=cached_x,
        )
        prediction = (standardized * y_std + y_mean).astype(np.float32)
        predictions[leaf] = prediction
        arrays = aggregate_events(
            targets[getattr(split, leaf)],
            prediction,
            metadata["event_id"][getattr(split, leaf)],
            metadata["subject_id"][getattr(split, leaf)],
            metadata["day_id"][getattr(split, leaf)],
        )
        event_predictions[leaf] = arrays
        metrics[leaf] = evaluate_events(arrays, label_names, y_std)
    embeddings = _extract_embeddings(
        model, dataset.x, np.arange(dataset.row_count), channel_mean, channel_std, torch=torch, device=device,
        cached_x=cached_x,
    )
    return {
        "status": "ok",
        "protocol": protocol,
        "label_names": list(label_names),
        "seed": int(seed),
        "split_root": str(split.source_root),
        "split_counts": {leaf: int(len(getattr(split, leaf))) for leaf in ("train", "val", "test")},
        "backend": backend_report.get("backend"),
        "backend_report": backend_report,
        "trainability": trainability,
        "train_audit": {
            "selection_metric": "validation event-level macro standardized RMSE",
            "best_epoch": best_epoch,
            "best_val_event_macro_standardized_rmse": best_val,
            "epoch_count": len(history),
            "history": history,
            "normalization": "train_only_channel_and_per_label_target_mean_std",
            "supervision_unit": "event_equal_replicated_window_mse",
            "train_event_count": int(np.unique(metadata["event_id"][split.train]).size),
            "event_weight_sum": float(weights.sum()),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(accumulation),
            "input_cache": "protocol_normalized_full_gpu_tensor" if cached_x is not None else "streamed_memmap_batches",
            "partial_last_n_blocks": int(runtime.partial_last_n_blocks),
            "target_mean": y_mean.reshape(-1).tolist(),
            "target_std": y_std.reshape(-1).tolist(),
        },
        "val": metrics["val"],
        "test": metrics["test"],
        "event_predictions": event_predictions,
        "embeddings": embeddings,
        "duration_seconds": float(time.time() - started),
    }


class EEGSupervisedRegressor:  # pragma: no cover - constructed only with torch runtime
    def __new__(
        cls, *, encoder: Any, label_names: tuple[str, ...], hidden_dim: int, dropout: float, torch: Any
    ) -> Any:
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.projection = torch.nn.Sequential(
                    torch.nn.LazyLinear(256), torch.nn.ReLU(), torch.nn.Dropout(dropout)
                )
                self.shared = torch.nn.Sequential(
                    torch.nn.LayerNorm(256),
                    torch.nn.Linear(256, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                )
                head_hidden = max(1, hidden_dim // 2)
                self.heads = torch.nn.ModuleDict({
                    label: torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, head_hidden),
                        torch.nn.GELU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(head_hidden, 1),
                    )
                    for label in label_names
                })

            def pooled_feature(self, x: Any) -> Any:
                output = self.encoder(x)
                if isinstance(output, dict):
                    output = output.get("features", output.get("encoder_output", output.get("cls_token")))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if output.ndim > 2:
                    output = output.reshape(output.shape[0], -1, output.shape[-1]).mean(dim=1)
                if output.ndim == 1:
                    output = output.unsqueeze(0)
                return output

            def embedding(self, x: Any) -> Any:
                return self.projection(self.pooled_feature(x))

            def forward(self, x: Any) -> Any:
                shared = self.shared(self.embedding(x))
                return torch.cat([self.heads[label](shared) for label in label_names], dim=1)

        return Module()


def configure_strict_last_blocks(encoder: Any, *, last_n_blocks: int = 2) -> dict[str, Any]:
    """Train exactly the last N transformer blocks and the encoder's final norm.

    Projection and task heads live outside ``encoder`` in
    :class:`EEGSupervisedRegressor` and remain trainable by construction.  This
    deliberately avoids matching every in-block ``attn.proj``/``norm`` name.
    """
    named = list(encoder.named_parameters())
    block_pattern = re.compile(r"(?:^|\.)(?:blocks|layers|encoder_layers)\.(\d+)(?:\.|$)")
    block_indices = sorted(
        {
            int(match.group(1))
            for name, _ in named
            for match in [block_pattern.search(name)]
            if match is not None
        }
    )
    keep = set(block_indices[-max(0, int(last_n_blocks)) :])
    trainable = []
    frozen = []
    for name, parameter in named:
        match = block_pattern.search(name)
        in_kept_block = match is not None and int(match.group(1)) in keep
        is_final_norm = bool(re.search(r"(?:^|\.)norm\.(?:weight|bias)$", name)) and match is None
        parameter.requires_grad = bool(in_kept_block or is_final_norm)
        (trainable if parameter.requires_grad else frozen).append(name)
    return {
        "strategy": "strict_last_blocks_plus_final_norm",
        "parameter_count": len(named),
        "trainable_count": len(trainable),
        "frozen_count": len(frozen),
        "kept_block_indices": sorted(keep),
        "trainable_names": trainable,
        "last_n_blocks": int(last_n_blocks),
    }


def prepare_protocol_input_cache(
    dataset: EEGAlignedDataset,
    split: SplitProtocol,
    *,
    device: str,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Normalize from train rows and cache the full EEG tensor on the selected device."""
    import torch

    dev = torch.device(device)
    channel_mean, channel_std = _fit_channel_normalization(dataset.x, split.train)
    tensor = torch.empty(
        (dataset.row_count, dataset.channel_count, dataset.sample_count),
        dtype=torch.float32,
        device=dev,
    )
    for start in range(0, dataset.row_count, chunk_size):
        indices = np.arange(start, min(dataset.row_count, start + chunk_size), dtype=np.int64)
        tensor[start : start + len(indices)] = _torch_eeg_batch(
            dataset.x, indices, channel_mean, channel_std, torch=torch, device=dev
        )
    return {
        "tensor": tensor,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "row_count": dataset.row_count,
        "source": "train_only_normalized_full_gpu_tensor",
    }


def _predict_standardized(
    model: Any,
    x: np.ndarray,
    indices: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    *,
    batch_size: int,
    torch: Any,
    device: Any,
    cached_x: Any | None = None,
) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch = (
                cached_x[torch.as_tensor(chunk, dtype=torch.long, device=device)]
                if cached_x is not None
                else _torch_eeg_batch(x, chunk, channel_mean, channel_std, torch=torch, device=device)
            )
            values.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(values, axis=0)


def _extract_embeddings(
    model: Any,
    x: np.ndarray,
    indices: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    *,
    torch: Any,
    device: Any,
    cached_x: Any | None = None,
) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            chunk = indices[start : start + 512]
            batch = (
                cached_x[torch.as_tensor(chunk, dtype=torch.long, device=device)]
                if cached_x is not None
                else _torch_eeg_batch(x, chunk, channel_mean, channel_std, torch=torch, device=device)
            )
            values.append(model.embedding(batch).detach().cpu().numpy().astype(np.float32))
    result = np.concatenate(values, axis=0)
    if result.shape != (len(indices), 256) or not np.isfinite(result).all():
        raise ValueError(f"invalid exported EEG embedding shape/values: {result.shape}")
    return result


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
