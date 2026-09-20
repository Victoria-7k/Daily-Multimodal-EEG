"""Matched 0814/0906 structures on one fixed token bag and 11 EMA labels."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES, load_window_split, target_value
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig, DailyAffectRegressionModel
from daily_multimodal.daily_affect.samplers import shuffled_batches
from daily_multimodal.daily_affect.training import (
    DailyAffectBagDataset,
    _seed_everything,
    apply_modality_dropout,
    difficulty_lambda_for_epoch,
    fit_token_normalization,
    normalize_tokens,
)
from daily_multimodal.training.multihead_regression import evaluate_event_level


EEG_BRANCH = "eeg_eegpt_partial_ft_multitask_11label_v1"
EEG_PROFILE = "eegpt_partial_ft_multitask_11label_v1"
ROUTE_ID = f"A1_Wphysio_no_audio__{EEG_BRANCH}"
EMBEDDING_SEED = 240800
STATIC_POLICIES = (
    "uniform", "last_10s", "last_30s", "last_60s", "first_30s",
    "kernel_short", "kernel_medium", "kernel_long",
)
OTHER_MODELS = (
    "state_uniform", "prior_uniform", "prior_ordD_uniform", "global_kernel_no_prior",
    "dynamic_kernel_no_prior", "dynamic_kernel_prior_uniform", "dynamic_kernel",
    "dynamic_fixed_short", "dynamic_fixed_medium", "dynamic_fixed_long",
)


def conditions() -> dict[str, tuple[str, str]]:
    result = {"window_attention_regression_full_mean": ("window_replicated", "uniform")}
    result.update({f"bag_static_reg__temporal_{policy}": ("bag_static", policy) for policy in STATIC_POLICIES})
    for model_id in OTHER_MODELS:
        name = "prior_regD_uniform_reg" if model_id == "prior_ordD_uniform" else f"{model_id}_reg"
        result[name] = (model_id, "uniform")
    return result


class SharedTwoLayerElevenHeads(nn.Module):
    """The same two-linear-layer trunk and 11 two-linear-layer heads on every row."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        head_dim = max(1, hidden_dim // 2)
        self.heads = nn.ModuleDict({
            label: nn.Sequential(
                nn.Linear(hidden_dim, head_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_dim, 1),
            )
            for label in LABEL_NAMES
        })

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        shared = self.shared(representation)
        return torch.cat([self.heads[label](shared) for label in LABEL_NAMES], dim=-1)


class MultiLabelGaussianProbe(nn.Module):
    """Uncertainty probe for the ordinal-difficulty structural variant."""

    def __init__(self, hidden_dim: int, modality_count: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2 * len(LABEL_NAMES)) for _ in range(modality_count)])

    def forward(self, adapted: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = mask.bool()
        modality_valid = valid.any(dim=1)
        counts = valid.sum(dim=1).clamp_min(1).to(adapted.dtype)
        pooled = (adapted * valid[:, :, :, None]).sum(dim=1) / counts[:, :, None]
        parameters = torch.stack([head(pooled[:, index]) for index, head in enumerate(self.heads)], dim=1)
        parameters = parameters.reshape(adapted.shape[0], adapted.shape[2], len(LABEL_NAMES), 2)
        mean, log_variance = parameters[..., 0], parameters[..., 1].clamp(-8.0, 8.0)
        difficulty = torch.sigmoid(log_variance).mean(dim=-1)
        difficulty = torch.where(modality_valid, difficulty, torch.ones_like(difficulty))
        return {
            "probe_mean": mean,
            "probe_log_variance": log_variance,
            "modality_difficulty": difficulty,
            "valid_modality_mask": modality_valid,
        }


class MultiEmotionStructureModel(DailyAffectRegressionModel):
    def __init__(self, config: DailyAffectRegressionConfig) -> None:
        super().__init__(config)
        self.head = SharedTwoLayerElevenHeads(config.hidden_dim, config.dropout)
        if self.uses_regression_difficulty:
            self.probe = MultiLabelGaussianProbe(config.hidden_dim, config.modality_count)


def event_targets(dataset: DailyAffectBagDataset, index_rows: list[dict[str, Any]]) -> np.ndarray:
    """Use the bag's sample IDs to prove all 23 windows carry one 11-label EMA observation."""

    by_sample = {str(row["sample_id"]): row for row in index_rows}
    if len(by_sample) != len(index_rows):
        raise ValueError("canonical index has duplicate sample_id")
    values = np.zeros((dataset.row_count, len(LABEL_NAMES)), dtype=np.float32)
    for bag_index, sample_ids in enumerate(dataset.sample_id_matrix):
        if len(set(sample_ids.tolist())) != 23:
            raise ValueError(f"bag {bag_index} does not contain 23 distinct windows")
        first = by_sample[str(sample_ids[0])]
        target = np.asarray([target_value(first, label) for label in LABEL_NAMES], dtype=np.float32)
        for sample_id in sample_ids[1:]:
            row = by_sample[str(sample_id)]
            if str(row.get("event_id")) != str(dataset.event_id[bag_index]):
                raise ValueError(f"bag {bag_index} event_id mismatch")
            if not np.allclose([target_value(row, label) for label in LABEL_NAMES], target, atol=1e-6):
                raise ValueError(f"bag {bag_index} has inconsistent 11-label target")
        if not np.isclose(target[LABEL_NAMES.index("fatigue")], dataset.label[bag_index], atol=1e-6):
            raise ValueError(f"bag {bag_index} does not match its fatigue label")
        values[bag_index] = target
    if not np.isfinite(values).all():
        raise ValueError("non-finite multi-emotion targets")
    return values


def audit_bag(dataset: DailyAffectBagDataset, targets: np.ndarray, protocol: str) -> dict[str, Any]:
    split = dataset.split_indices()
    if dataset.route_id != ROUTE_ID:
        raise ValueError(f"expected {ROUTE_ID}, got {dataset.route_id}")
    if dataset.tokens.shape != (1253, 23, 4, 256) or targets.shape != (1253, 11):
        raise ValueError(f"unexpected bag/target shapes: {dataset.tokens.shape}, {targets.shape}")
    if not np.isfinite(dataset.tokens).all() or not dataset.modality_mask[:, :, 0].all():
        raise ValueError("invalid common EEG token")
    if dataset.modality_mask[:, :, 3].any() or np.any(dataset.tokens[:, :, 3]):
        raise ValueError("no_audio bag unexpectedly contains audio")
    if len(set(dataset.event_id.tolist())) != dataset.row_count:
        raise ValueError("duplicate event ids")
    leaves = {key: set(split[key].tolist()) for key in ("train", "val", "test")}
    if any(leaves[a] & leaves[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("event split overlap")
    if sum(len(values) for values in leaves.values()) != dataset.row_count:
        raise ValueError("event split does not cover all bags")
    subject_days = {}
    for key in leaves:
        subject_days[key] = {(str(dataset.subject_id[i]), str(dataset.day_id[i])) for i in leaves[key]}
    if protocol == "within_subject_day" and any(
        subject_days[a] & subject_days[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("repaired within_subject_day has subject-day overlap")
    sources = json.loads(dataset.source_npz_json)
    expected = {EEG_BRANCH, "wear_physio", "video_A1"}
    expected_suffix = f"/multitask_11label/{protocol}/seed_{EMBEDDING_SEED}.npz"
    if set(sources) != expected or expected_suffix not in sources[EEG_BRANCH].replace("\\", "/"):
        raise ValueError(f"unexpected fixed embedding provenance: {sources}")
    return {
        "protocol": protocol, "route_id": dataset.route_id, "embedding_seed": EMBEDDING_SEED,
        "event_count": dataset.row_count, "split_counts": {key: len(leaves[key]) for key in leaves},
        "subject_day_counts": {key: len(subject_days[key]) for key in leaves},
        "source_npz": sources, "supervision_boundary": dataset.supervision_boundary,
        "label_names": list(LABEL_NAMES),
    }


def audit_multitask_eeg_token(path: Path, index_rows: list[dict[str, Any]], split_root: Path, protocol: str) -> dict[str, Any]:
    """Verify the fixed 11-label-supervised token and its repaired split provenance."""

    install_numpy_core_pickle_aliases()
    expected_sample_ids = np.asarray([str(row["sample_id"]) for row in index_rows])
    leaf = load_window_split(split_root / protocol, len(index_rows))
    expected = {
        "train_index": np.flatnonzero(np.isin(leaf, ("pretrain", "finetune"))),
        "val_index": np.flatnonzero(leaf == "val"),
        "test_index": np.flatnonzero(leaf == "test"),
    }
    with np.load(path, allow_pickle=True) as token:
        if not np.array_equal(token["sample_id"].astype(str), expected_sample_ids):
            raise ValueError(f"fixed EEG token sample order mismatch: {path}")
        for key, indices in expected.items():
            if not np.array_equal(np.sort(token[key].astype(np.int64)), indices):
                raise ValueError(f"fixed EEG token {key} differs from canonical split: {path}")
        profile = np.unique(token["encoder_profile"].astype(str)).tolist()
        if profile != [EEG_PROFILE]:
            raise ValueError(f"unexpected EEG token profile: {profile}")
        if str(token["protocol"][0]) != protocol or int(token["seed"][0]) != EMBEDDING_SEED:
            raise ValueError(f"unexpected EEG token protocol/seed: {path}")
        if str(token["target_label"][0]) != "all_11_labels":
            raise ValueError(f"unexpected EEG token target_label: {path}")
        if tuple(token["target_labels"].astype(str).tolist()) != LABEL_NAMES:
            raise ValueError(f"unexpected EEG token target label order: {path}")
        if str(token["train_supervision"][0]) != "multitask_event_supervised_eegpt_partial_ft":
            raise ValueError(f"unexpected EEG token supervision: {path}")
        if token["eeg_emb"].shape != (len(index_rows), 256) or not np.isfinite(token["eeg_emb"]).all():
            raise ValueError(f"invalid EEG token values: {path}")
    return {
        "eeg_token_path": str(path), "eeg_profile": EEG_PROFILE,
        "sample_order_match": True, "split_indices_match": True,
        "target_labels_match": True, "train_supervision": "multitask_event_supervised_eegpt_partial_ft",
    }


def _batch(dataset: DailyAffectBagDataset, targets: np.ndarray, indices: np.ndarray, x_mean: np.ndarray,
           x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device: torch.device
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = normalize_tokens(dataset.tokens[indices], x_mean, x_std)
    return (
        torch.as_tensor(tokens, dtype=torch.float32, device=device),
        torch.as_tensor(dataset.modality_mask[indices], dtype=torch.bool, device=device),
        torch.as_tensor((targets[indices] - y_mean) / y_std, dtype=torch.float32, device=device),
    )


def _predict(model: MultiEmotionStructureModel, dataset: DailyAffectBagDataset, indices: np.ndarray,
             x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray,
             device: torch.device, batch_size: int = 128) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x, mask, _ = _batch(dataset, np.zeros((dataset.row_count, len(LABEL_NAMES)), dtype=np.float32),
                                  indices[start:start + batch_size], x_mean, x_std, y_mean, y_std, device)
            chunks.append((model(x, mask)["prediction"].cpu().numpy() * y_std + y_mean).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def run_condition(*, dataset: DailyAffectBagDataset, targets: np.ndarray, protocol: str, condition_id: str,
                  model_id: str, temporal_policy: str, seed: int, out_dir: Path, epochs: int = 80,
                  batch_size: int = 64, hidden_dim: int = 128, learning_rate: float = 1e-3,
                  weight_decay: float = 1e-4, dropout: float = 0.1, patience: int = 15,
                  modality_dropout_prob: float = 0.1, lambda_d: float = 0.25,
                  probe_loss_weight: float = 0.1, device: str = "cuda") -> dict[str, Any]:
    started = time.time()
    _seed_everything(seed)
    split = dataset.split_indices()
    train, val = split["train"], split["val"]
    x_mean, x_std = fit_token_normalization(dataset.tokens, dataset.modality_mask, train, scope="per_modality")
    y_mean = targets[train].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = targets[train].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(np.isfinite(y_std) & (y_std >= 1e-6), y_std, 1.0).astype(np.float32)
    cfg = DailyAffectRegressionConfig(
        model_id=model_id, hidden_dim=hidden_dim, dropout=dropout,
        adapter_mode="per_modality", temporal_policy=temporal_policy, lambda_d=lambda_d,
    )
    dev = torch.device(device)
    model = MultiEmotionStructureModel(cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_score, best_epoch, stale, best_state = math.inf, 0, 0, None
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        losses = []
        scheduled_lambda = difficulty_lambda_for_epoch(epoch, target=lambda_d, warmup_epochs=5, ramp_epochs=5)
        for batch_indices in shuffled_batches(train, batch_size, rng):
            x, mask, target = _batch(dataset, targets, batch_indices, x_mean, x_std, y_mean, y_std, dev)
            x, mask = apply_modality_dropout(x, mask, probability=modality_dropout_prob, rng=rng)
            output = model(x, mask, lambda_d_override=scheduled_lambda)
            if model.uses_window_regression:
                valid = output["window_mask"].to(target.dtype)
                error = (output["window_prediction"] - target[:, None, :]).square()
                head_loss = ((error * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]).mean()
            else:
                head_loss = (output["prediction"] - target).square().mean()
            if model.uses_regression_difficulty:
                valid = output["valid_modality_mask"][:, :, None]
                logvar = output["probe_log_variance"]
                residual = target[:, None, :] - output["probe_mean"]
                nll = 0.5 * (torch.exp(-logvar) * residual.square() + logvar)
                probe_loss = (nll * valid).sum() / valid.sum().clamp_min(1) / len(LABEL_NAMES)
            else:
                probe_loss = head_loss.new_zeros(())
            loss = head_loss + probe_loss_weight * probe_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_pred = _predict(model, dataset, val, x_mean, x_std, y_mean, y_std, dev)
        val_error = (val_pred - targets[val]) / y_std
        score = float(np.sqrt(np.mean(np.square(val_error), axis=0)).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_standardized_rmse": score})
        if score < best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    for leaf in ("val", "test"):
        indices = split[leaf]
        pred = _predict(model, dataset, indices, x_mean, x_std, y_mean, y_std, dev)
        predictions[leaf] = pred
        arrays = {"target": targets[indices], "prediction": pred, "subject_id": dataset.subject_id[indices]}
        metrics[leaf] = evaluate_event_level(arrays, y_std)
    np.savez_compressed(
        out_dir / "event_predictions.npz", label_names=np.asarray(LABEL_NAMES),
        **{f"{leaf}_{key}": value for leaf in ("val", "test") for key, value in {
            "event_id": dataset.event_id[split[leaf]], "subject_id": dataset.subject_id[split[leaf]],
            "day_id": dataset.day_id[split[leaf]], "target": targets[split[leaf]], "prediction": predictions[leaf],
        }.items()},
    )
    torch.save({
        "state_dict": best_state, "condition_id": condition_id, "model_id": model_id,
        "temporal_policy": temporal_policy, "label_names": LABEL_NAMES, "embedding_seed": EMBEDDING_SEED,
        "downstream_seed": seed, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
    }, out_dir / "best_checkpoint.pt")
    result = {
        "status": "ok", "protocol": protocol, "route_id": ROUTE_ID, "condition_id": condition_id,
        "model_id": model_id, "temporal_policy": temporal_policy, "normalization": "per_modality",
        "adapter_mode": "per_modality", "head_variant": "H1_shared2_11xhead2",
        "embedding_seed": EMBEDDING_SEED, "downstream_seed": seed, "bag_path": str(dataset.bag_path),
        "supervision_boundary": dataset.supervision_boundary, "selection_metric": "val_macro_standardized_rmse_min",
        "best_epoch": best_epoch, "best_val_macro_standardized_rmse": best_score,
        "epochs_ran": len(history), "duration_seconds": time.time() - started,
        "train_target_mean": y_mean.reshape(-1).tolist(), "train_target_std": y_std.reshape(-1).tolist(),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "val": metrics["val"], "test": metrics["test"], "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
