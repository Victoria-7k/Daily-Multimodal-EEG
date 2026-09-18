"""Event-aware multi-label fusion regressors for the multi-emotion route."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

import numpy as np
import torch

from daily_multimodal.training.centered_metrics import safe_pearsonr, within_subject_centered_arrays


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
POSITIVE_LABELS = LABEL_NAMES[:5]
NEGATIVE_LABELS = LABEL_NAMES[5:10]
HEAD_VARIANTS = ("E0_existing_11out", "H0_shared2_linear11", "H1_shared2_11xhead2")


class EventAwareMultiLabelRegressor(torch.nn.Module):
    def __init__(self, *, modality_count: int, hidden_dim: int, dropout: float, head_variant: str) -> None:
        super().__init__()
        if head_variant not in HEAD_VARIANTS:
            raise ValueError(f"unsupported head variant: {head_variant}")
        self.head_variant = head_variant
        self.input_projection = torch.nn.Linear(256, hidden_dim)
        self.modality_embedding = torch.nn.Parameter(torch.zeros(1, modality_count, hidden_dim))
        torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, 1, dropout=dropout, batch_first=True)
        self.query = torch.nn.Parameter(torch.zeros(hidden_dim))
        torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.dropout = torch.nn.Dropout(dropout)
        if head_variant == "E0_existing_11out":
            self.shared = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            self.output = torch.nn.Linear(hidden_dim, len(LABEL_NAMES))
            self.heads = None
        else:
            self.shared = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            if head_variant == "H0_shared2_linear11":
                self.output = torch.nn.Linear(hidden_dim, len(LABEL_NAMES))
                self.heads = None
            else:
                head_hidden = max(1, hidden_dim // 2)
                self.output = None
                self.heads = torch.nn.ModuleDict(
                    {
                        label: torch.nn.Sequential(
                            torch.nn.Linear(hidden_dim, head_hidden),
                            torch.nn.GELU(),
                            torch.nn.Dropout(dropout),
                            torch.nn.Linear(head_hidden, 1),
                        )
                        for label in LABEL_NAMES
                    }
                )

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(torch.all(mask.any(dim=1))):
            raise ValueError("every sample must have at least one available modality")
        hidden = self.input_projection(tokens) + self.modality_embedding[:, : tokens.shape[1], :]
        hidden = hidden.masked_fill(~mask[:, :, None], 0.0)
        attended, _ = self.self_attention(hidden, hidden, hidden, key_padding_mask=~mask, need_weights=False)
        attended = attended.masked_fill(~mask[:, :, None], 0.0)
        scores = torch.einsum("bmh,h->bm", attended, self.query).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return self.shared(self.dropout(torch.sum(attended * weights, dim=1)))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shared = self.encode(tokens, mask)
        if self.heads is None:
            assert self.output is not None
            return self.output(shared)
        return torch.cat([self.heads[label](shared) for label in LABEL_NAMES], dim=1)


class EventAwareScalarRegressor(torch.nn.Module):
    """ST-11 downstream model: two-layer trunk plus a two-layer scalar head."""

    def __init__(self, *, modality_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = torch.nn.Linear(256, hidden_dim)
        self.modality_embedding = torch.nn.Parameter(torch.zeros(1, modality_count, hidden_dim))
        torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.self_attention = torch.nn.MultiheadAttention(hidden_dim, 1, dropout=dropout, batch_first=True)
        self.query = torch.nn.Parameter(torch.zeros(hidden_dim))
        torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.shared = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(torch.all(mask.any(dim=1))):
            raise ValueError("every sample must have at least one available modality")
        hidden = self.input_projection(tokens) + self.modality_embedding[:, : tokens.shape[1], :]
        hidden = hidden.masked_fill(~mask[:, :, None], 0.0)
        attended, _ = self.self_attention(hidden, hidden, hidden, key_padding_mask=~mask, need_weights=False)
        attended = attended.masked_fill(~mask[:, :, None], 0.0)
        scores = torch.einsum("bmh,h->bm", attended, self.query).masked_fill(~mask, -1e9)
        pooled = torch.sum(attended * torch.softmax(scores, dim=1).unsqueeze(-1), dim=1)
        return self.head(self.shared(pooled)).reshape(-1, 1)


def fit_event_aware_model(
    *,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    targets: np.ndarray,
    event_id: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    head_variant: str,
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
    _seed_everything(seed)
    x_mean, x_std = fit_token_normalization(tokens, token_mask, train_idx)
    y_mean = targets[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = targets[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(np.isfinite(y_std) & (y_std >= 1e-6), y_std, 1.0).astype(np.float32)
    train_weights = event_equal_window_weights(event_id[train_idx])
    dev = torch.device(device)
    module = EventAwareMultiLabelRegressor(
        modality_count=tokens.shape[1], hidden_dim=hidden_dim, dropout=dropout, head_variant=head_variant
    ).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train = torch.as_tensor(normalize_tokens(tokens[train_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_train = torch.as_tensor(token_mask[train_idx], dtype=torch.bool, device=dev)
    y_train = torch.as_tensor((targets[train_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    w_train = torch.as_tensor(train_weights, dtype=torch.float32, device=dev)
    x_val = torch.as_tensor(normalize_tokens(tokens[val_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_val = torch.as_tensor(token_mask[val_idx], dtype=torch.bool, device=dev)
    y_val_np = ((targets[val_idx] - y_mean) / y_std).astype(np.float32)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        module.train()
        losses: list[float] = []
        for batch in make_batches(len(train_idx), batch_size, rng):
            prediction = module(x_train[batch], m_train[batch])
            per_window = torch.mean((prediction - y_train[batch]) ** 2, dim=1)
            loss = torch.sum(per_window * w_train[batch]) / torch.clamp(torch.sum(w_train[batch]), min=1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        module.eval()
        with torch.no_grad():
            val_pred = module(x_val, m_val).detach().cpu().numpy()
        val_loss = event_macro_standardized_rmse(y_val_np, val_pred, event_id[val_idx])
        history.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else math.nan, "val_event_macro_srmse": val_loss}
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in module.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    module.load_state_dict(best_state)
    bundle = {"module": module, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    audit = {
        "head_variant": head_variant,
        "best_epoch": best_epoch,
        "best_val_event_macro_standardized_rmse": best_val,
        "epoch_count": len(history),
        "history": history,
        "event_weight_sum": float(train_weights.sum()),
        "train_event_count": int(np.unique(event_id[train_idx]).size),
        "target_normalization": {
            label: {"mean": float(y_mean[0, idx]), "std": float(y_std[0, idx])}
            for idx, label in enumerate(LABEL_NAMES)
        },
    }
    return bundle, audit


def fit_event_aware_scalar(
    *,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    targets: np.ndarray,
    event_id: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
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
    _seed_everything(seed)
    values = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    x_mean, x_std = fit_token_normalization(tokens, token_mask, train_idx)
    y_mean = values[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = values[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(y_std >= 1e-6, y_std, 1.0).astype(np.float32)
    train_weights = event_equal_window_weights(event_id[train_idx])
    dev = torch.device(device)
    module = EventAwareScalarRegressor(
        modality_count=tokens.shape[1], hidden_dim=hidden_dim, dropout=dropout
    ).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train = torch.as_tensor(normalize_tokens(tokens[train_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_train = torch.as_tensor(token_mask[train_idx], dtype=torch.bool, device=dev)
    y_train = torch.as_tensor((values[train_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    w_train = torch.as_tensor(train_weights, dtype=torch.float32, device=dev)
    x_val = torch.as_tensor(normalize_tokens(tokens[val_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_val = torch.as_tensor(token_mask[val_idx], dtype=torch.bool, device=dev)
    y_val = ((values[val_idx] - y_mean) / y_std).astype(np.float32)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        module.train()
        losses = []
        for batch in make_batches(len(train_idx), batch_size, rng):
            prediction = module(x_train[batch], m_train[batch])
            per_window = torch.mean((prediction - y_train[batch]) ** 2, dim=1)
            loss = torch.sum(per_window * w_train[batch]) / torch.clamp(torch.sum(w_train[batch]), min=1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        module.eval()
        with torch.no_grad():
            val_prediction = module(x_val, m_val).detach().cpu().numpy()
        val_metric = event_macro_standardized_rmse(y_val, val_prediction, event_id[val_idx])
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_event_standardized_rmse": val_metric,
        })
        if val_metric < best_val:
            best_val = val_metric
            best_epoch = epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in module.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    module.load_state_dict(best_state)
    return {
        "module": module,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, {
        "head_variant": "ST_shared2_head2_scalar",
        "best_epoch": best_epoch,
        "best_val_event_standardized_rmse": best_val,
        "epoch_count": len(history),
        "history": history,
        "event_weight_sum": float(train_weights.sum()),
        "train_event_count": int(np.unique(event_id[train_idx]).size),
    }


def predict(bundle: dict[str, Any], tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray, device: str) -> np.ndarray:
    module: EventAwareMultiLabelRegressor = bundle["module"]
    dev = torch.device(device)
    output: list[np.ndarray] = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            chunk = indices[start : start + 1024]
            x = torch.as_tensor(normalize_tokens(tokens[chunk], bundle["x_mean"], bundle["x_std"]), dtype=torch.float32, device=dev)
            m = torch.as_tensor(mask[chunk], dtype=torch.bool, device=dev)
            pred = module(x, m).detach().cpu().numpy()
            output.append((pred * bundle["y_std"] + bundle["y_mean"]).astype(np.float32))
    return np.concatenate(output, axis=0) if output else np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)


def predict_scalar(bundle: dict[str, Any], tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray, device: str) -> np.ndarray:
    module: EventAwareScalarRegressor = bundle["module"]
    dev = torch.device(device)
    output = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            chunk = indices[start : start + 1024]
            x = torch.as_tensor(
                normalize_tokens(tokens[chunk], bundle["x_mean"], bundle["x_std"]),
                dtype=torch.float32,
                device=dev,
            )
            m = torch.as_tensor(mask[chunk], dtype=torch.bool, device=dev)
            pred = module(x, m).detach().cpu().numpy()
            output.append((pred * bundle["y_std"] + bundle["y_mean"]).astype(np.float32))
    return np.concatenate(output, axis=0) if output else np.zeros((0, 1), dtype=np.float32)


def evaluate_scalar_event_level(arrays: dict[str, np.ndarray], train_y_std: float) -> dict[str, Any]:
    true = arrays["target"][:, 0]
    pred = arrays["prediction"][:, 0]
    error = pred - true
    centered_true, centered_pred = within_subject_centered_arrays(true, pred, arrays["subject_id"])
    return {
        "count": int(len(true)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "standardized_rmse": float(np.sqrt(np.mean(error * error)) / float(train_y_std)),
        "mae": float(np.mean(np.abs(error))),
        "raw_r": safe_pearsonr(true, pred),
        "within_subject_centered_r": safe_pearsonr(centered_true, centered_pred),
        "label_std": float(np.std(true)),
        "prediction_std": float(np.std(pred)),
    }


def event_level_arrays(
    targets: np.ndarray,
    predictions: np.ndarray,
    event_id: np.ndarray,
    subject_id: np.ndarray,
    day_id: np.ndarray,
) -> dict[str, np.ndarray]:
    unique, inverse = np.unique(event_id.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    y_sum = np.zeros((len(unique), targets.shape[1]), dtype=np.float64)
    p_sum = np.zeros_like(y_sum)
    np.add.at(y_sum, inverse, targets)
    np.add.at(p_sum, inverse, predictions)
    y_event = (y_sum / counts[:, None]).astype(np.float32)
    p_event = (p_sum / counts[:, None]).astype(np.float32)
    subjects = np.empty(len(unique), dtype=object)
    days = np.empty(len(unique), dtype=object)
    for idx in range(len(unique)):
        members = np.flatnonzero(inverse == idx)
        sub_values = np.unique(subject_id[members].astype(str))
        day_values = np.unique(day_id[members].astype(str))
        if len(sub_values) != 1 or len(day_values) != 1:
            raise ValueError(f"event {unique[idx]} crosses subject/day")
        if not np.allclose(targets[members], targets[members[0]], atol=1e-6):
            raise ValueError(f"event {unique[idx]} has inconsistent labels")
        subjects[idx] = sub_values[0]
        days[idx] = day_values[0]
    return {
        "event_id": unique.astype(str),
        "subject_id": subjects.astype(str),
        "day_id": days.astype(str),
        "target": y_event,
        "prediction": p_event,
        "window_count": counts.astype(np.int64),
    }


def evaluate_event_level(arrays: dict[str, np.ndarray], train_y_std: np.ndarray) -> dict[str, Any]:
    per_label: dict[str, dict[str, Any]] = {}
    target = arrays["target"]
    prediction = arrays["prediction"]
    subjects = arrays["subject_id"]
    for idx, label in enumerate(LABEL_NAMES):
        true = target[:, idx]
        pred = prediction[:, idx]
        error = pred - true
        centered_true, centered_pred = within_subject_centered_arrays(true, pred, subjects)
        tail = true >= 3.0
        per_label[label] = {
            "count": int(len(true)),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "standardized_rmse": float(np.sqrt(np.mean(error * error)) / float(train_y_std[0, idx])),
            "mae": float(np.mean(np.abs(error))),
            "raw_r": safe_pearsonr(true, pred),
            "within_subject_centered_r": safe_pearsonr(centered_true, centered_pred),
            "label_std": float(np.std(true)),
            "prediction_std": float(np.std(pred)),
            "tail_count_y_ge_3": int(tail.sum()),
            "tail_mae_y_ge_3": float(np.mean(np.abs(error[tail]))) if bool(tail.any()) else None,
        }
    return {"per_label": per_label, "summary": summarize_metrics(per_label), "groups": group_summaries(per_label)}


def summarize_metrics(per_label: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {metric: mean_metric(per_label, metric) for metric in ("rmse", "standardized_rmse", "mae", "raw_r", "within_subject_centered_r")}


def group_summaries(per_label: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups = {"positive_activation": POSITIVE_LABELS, "negative_distress": NEGATIVE_LABELS, "fatigue": ("fatigue",)}
    return {
        name: {metric: mean_metric({label: per_label[label] for label in labels}, metric) for metric in ("standardized_rmse", "raw_r", "within_subject_centered_r")}
        for name, labels in groups.items()
    }


def mean_metric(per_label: dict[str, dict[str, Any]], metric: str) -> float | None:
    values = [row.get(metric) for row in per_label.values()]
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def event_equal_window_weights(event_id: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(event_id.astype(str), return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float32)


def event_macro_standardized_rmse(target: np.ndarray, prediction: np.ndarray, event_id: np.ndarray) -> float:
    dummy = np.asarray(["x"] * len(event_id))
    arrays = event_level_arrays(target, prediction, event_id, dummy, dummy)
    error = arrays["prediction"] - arrays["target"]
    return float(np.mean(np.sqrt(np.mean(error * error, axis=0))))


def fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, None], tokens[indices], np.nan)
    mean = np.nanmean(available, axis=(0, 1), keepdims=True)
    std = np.nanstd(available, axis=(0, 1), keepdims=True)
    return np.where(np.isfinite(mean), mean, 0.0).astype(np.float32), np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)


def normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def make_batches(n_rows: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_rows)
    return [order[start : start + max(1, batch_size)] for start in range(0, n_rows, max(1, batch_size))]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
