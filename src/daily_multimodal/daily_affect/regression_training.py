"""Training and artifacts for the scalar-regression Daily-affect route."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered

from .regression import DailyAffectRegressionConfig, DailyAffectRegressionModel
from .samplers import shuffled_batches
from .training import (
    DailyAffectBagDataset,
    _seed_everything,
    apply_modality_dropout,
    difficulty_lambda_for_epoch,
    fit_token_normalization,
    normalize_tokens,
)


def run_daily_affect_regression_run(
    *,
    dataset: DailyAffectBagDataset,
    protocol: str,
    condition_id: str,
    model_id: str,
    normalization: str,
    seed: int,
    run_dir: Path,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    hidden_dim: int = 128,
    patience: int = 15,
    lambda_d: float = 0.25,
    probe_loss_weight: float = 0.1,
    modality_dropout_prob: float = 0.1,
    probe_warmup_epochs: int = 5,
    difficulty_ramp_epochs: int = 5,
    temporal_policy: str = "uniform",
    device: str = "cuda",
    adapter_mode: str | None = None,
) -> dict[str, Any]:
    """Fit one leakage-safe, scalar-target condition and persist its evidence."""

    model, audit = fit_daily_affect_regression_model(
        dataset=dataset,
        model_id=model_id,
        normalization=normalization,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        hidden_dim=hidden_dim,
        patience=patience,
        lambda_d=lambda_d,
        probe_loss_weight=probe_loss_weight,
        modality_dropout_prob=modality_dropout_prob,
        probe_warmup_epochs=probe_warmup_epochs,
        difficulty_ramp_epochs=difficulty_ramp_epochs,
        temporal_policy=temporal_policy,
        device=device,
        adapter_mode=adapter_mode,
    )
    split = dataset.split_indices()
    predictions = {
        name: predict_daily_affect_regression_model(model, dataset, indices=indices, device=device, include_diagnostics=False)
        for name, indices in split.items()
        if name in {"train", "val", "test"}
    }
    metrics = {
        name: regression_metrics(dataset.label[split[name]], predictions[name]["event_prediction"], dataset.subject_id[split[name]])
        for name in ("train", "val", "test")
    }
    result: dict[str, Any] = {
        "protocol": protocol,
        "condition_id": condition_id,
        "family": "window" if model["module"].uses_window_regression else "ema_bag",
        "route_id": dataset.route_id,
        "model_id": model_id,
        "normalization": normalization,
        "adapter_mode": model["config"]["adapter_mode"],
        "objective_id": model["config"]["objective_id"],
        "supervision_unit": model["config"]["supervision_unit"],
        "routing_id": model["config"]["routing_id"],
        "seed": int(seed),
        "bag_path": str(dataset.bag_path),
        "target_label": dataset.target_label,
        "row_count": dataset.row_count,
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "supervision_boundary": dataset.supervision_boundary,
        "train": metrics["train"],
        "val": metrics["val"],
        "test": metrics["test"],
        "train_audit": audit,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions.npz"
    save_regression_predictions(prediction_path, dataset, split, predictions)
    test_table_path = run_dir / "test_predictions.csv"
    write_regression_prediction_table(test_table_path, dataset, split["test"], predictions["test"])
    state = audit.pop("_best_state")
    checkpoint_path = run_dir / "best_checkpoint.pt"
    torch.save(
        {
            "state_dict": state,
            "model_id": model_id,
            "normalization": normalization,
            "seed": int(seed),
            "config": model["config"],
            "x_mean": model["x_mean"],
            "x_std": model["x_std"],
            "target_mean": model["target_mean"],
            "target_std": model["target_std"],
        },
        checkpoint_path,
    )
    history_path = run_dir / "val_history.csv"
    history_path.write_text(regression_history_csv(audit["history"]), encoding="utf-8")
    result.update(
        {
            "prediction_path": str(prediction_path),
            "checkpoint_path": str(checkpoint_path),
            "test_predictions_csv": str(test_table_path),
            "metrics_path": str(run_dir / "metrics.json"),
            "config_path": str(run_dir / "config.json"),
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(regression_config_snapshot(model, dataset, protocol=protocol, condition_id=condition_id, seed=seed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def fit_daily_affect_regression_model(
    *,
    dataset: DailyAffectBagDataset,
    model_id: str,
    normalization: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    hidden_dim: int,
    patience: int,
    lambda_d: float,
    probe_loss_weight: float,
    modality_dropout_prob: float,
    probe_warmup_epochs: int,
    difficulty_ramp_epochs: int,
    temporal_policy: str,
    device: str,
    adapter_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if normalization not in {"shared", "per_modality"}:
        raise ValueError(f"unsupported normalization: {normalization}")
    adapter_mode = normalization if adapter_mode is None else adapter_mode
    if adapter_mode not in {"shared", "per_modality"}:
        raise ValueError(f"unsupported adapter_mode: {adapter_mode}")
    _seed_everything(seed)
    train = dataset.train_index
    val = dataset.val_index
    x_mean, x_std = fit_token_normalization(dataset.tokens, dataset.modality_mask, train, scope=normalization)
    target_mean = float(np.mean(dataset.label[train], dtype=np.float64))
    target_std = float(np.std(dataset.label[train], dtype=np.float64))
    target_std = target_std if math.isfinite(target_std) and target_std >= 1e-6 else 1.0
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    cfg = DailyAffectRegressionConfig(
        model_id=model_id,
        hidden_dim=hidden_dim,
        adapter_mode=adapter_mode,
        dropout=dropout,
        lambda_d=lambda_d,
        temporal_policy=temporal_policy,
    )
    module = DailyAffectRegressionModel(cfg).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(max(1, int(epochs))):
        module.train()
        losses: list[float] = []
        head_losses: list[float] = []
        probe_losses: list[float] = []
        scheduled_lambda_d = difficulty_lambda_for_epoch(
            epoch, target=lambda_d, warmup_epochs=probe_warmup_epochs, ramp_epochs=difficulty_ramp_epochs
        )
        for batch in shuffled_batches(train, batch_size, rng):
            tokens, mask, target = regression_batch_tensors(dataset, batch, x_mean, x_std, target_mean, target_std, dev)
            tokens, mask = apply_modality_dropout(tokens, mask, probability=modality_dropout_prob, rng=rng)
            outputs = module(
                tokens,
                mask,
                lambda_d_override=scheduled_lambda_d,
                detach_difficulty_override=cfg.detach_difficulty,
            )
            if module.uses_window_regression:
                head_loss = event_balanced_window_mse(outputs["window_prediction"], target, outputs["window_mask"])
            else:
                head_loss = torch.mean((outputs["prediction"] - target) ** 2)
            probe_loss = gaussian_probe_nll(outputs, target) if module.uses_regression_difficulty else head_loss.new_zeros(())
            loss = head_loss + float(probe_loss_weight) * probe_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite scalar-regression loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            head_losses.append(float(head_loss.detach().cpu().item()))
            probe_losses.append(float(probe_loss.detach().cpu().item()))
        module.eval()
        val_prediction = _predict_regression_module(module, dataset, val, x_mean, x_std, target_mean, target_std, dev, False)
        val_metrics = regression_metrics(dataset.label[val], val_prediction["event_prediction"], dataset.subject_id[val])
        val_rmse = float(val_metrics["rmse"]) if val_metrics["rmse"] is not None else float("inf")
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(np.mean(losses)) if losses else math.nan,
                "train_head_loss": float(np.mean(head_losses)) if head_losses else math.nan,
                "train_probe_nll": float(np.mean(probe_losses)) if probe_losses else math.nan,
                "routing_lambda_d": float(scheduled_lambda_d),
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_raw_r": val_metrics["raw_r"],
                "val_within_subject_centered_r": val_metrics["within_subject_centered_r"],
                "selection_score": -val_rmse,
            }
        )
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= max(1, int(patience)):
                break
    if best_state is None:
        raise RuntimeError("training produced no best scalar-regression state")
    module.load_state_dict(best_state)
    config = {
        "model_id": model_id,
        "normalization": normalization,
        "adapter_mode": adapter_mode,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "hidden_dim": int(hidden_dim),
        "patience": int(patience),
        "lambda_d": float(lambda_d),
        "probe_loss_weight": float(probe_loss_weight),
        "modality_dropout_prob": float(modality_dropout_prob),
        "probe_warmup_epochs": int(probe_warmup_epochs),
        "difficulty_ramp_epochs": int(difficulty_ramp_epochs),
        "selection_metric": "val_rmse_min",
        "objective_id": "scalar_mse__gaussian_probe_nll_0.1" if module.uses_regression_difficulty else "scalar_mse",
        "supervision_unit": "event_balanced_replicated_window_mse" if module.uses_window_regression else "ema_bag_event_mse",
        "routing_id": "regression_uncertainty__detach_true" if module.uses_regression_difficulty else "none",
        "temporal_policy": temporal_policy,
    }
    return (
        {"module": module, "x_mean": x_mean, "x_std": x_std, "target_mean": target_mean, "target_std": target_std, "config": config},
        {
            "best_epoch": int(best_epoch),
            "best_val_rmse": float(best_rmse),
            "selection_metric": "val_rmse_min",
            "epoch_count": int(len(history)),
            "initial_train_loss": history[0]["train_loss"] if history else math.nan,
            "final_train_loss": history[-1]["train_loss"] if history else math.nan,
            "trainable_params": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
            "total_params": int(sum(p.numel() for p in module.parameters())),
            "history": history,
            "_best_state": best_state,
        },
    )


def predict_daily_affect_regression_model(
    model: dict[str, Any], dataset: DailyAffectBagDataset, *, indices: np.ndarray, device: str, include_diagnostics: bool
) -> dict[str, np.ndarray]:
    module: DailyAffectRegressionModel = model["module"]
    module.eval()
    return _predict_regression_module(
        module,
        dataset,
        np.asarray(indices, dtype=np.int64),
        model["x_mean"],
        model["x_std"],
        float(model["target_mean"]),
        float(model["target_std"]),
        torch.device(device),
        include_diagnostics,
    )


def regression_batch_tensors(
    dataset: DailyAffectBagDataset,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = normalize_tokens(dataset.tokens[indices], x_mean, x_std)
    target = (dataset.label[indices].astype(np.float32) - float(target_mean)) / float(target_std)
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(dataset.modality_mask[indices], dtype=torch.bool, device=device),
        torch.as_tensor(target, dtype=torch.float32, device=device),
    )


def event_balanced_window_mse(window_prediction: torch.Tensor, target: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
    """Equal-weight each EMA event, then average its valid-window squared errors."""

    if not torch.any(window_mask):
        raise ValueError("window scalar supervision received a batch without valid modality windows")
    squared_error = (window_prediction - target[:, None]) ** 2
    weights = window_mask.to(dtype=squared_error.dtype)
    return torch.mean((squared_error * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))


def gaussian_probe_nll(outputs: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    valid = outputs["valid_modality_mask"]
    if not torch.any(valid):
        return target.new_zeros(())
    target_per_modality = target[:, None].expand_as(outputs["probe_mean"])
    log_variance = outputs["probe_log_variance"]
    nll = 0.5 * (torch.exp(-log_variance) * (target_per_modality - outputs["probe_mean"]) ** 2 + log_variance)
    return nll[valid].mean()


def _predict_regression_module(
    module: DailyAffectRegressionModel,
    dataset: DailyAffectBagDataset,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
    include_diagnostics: bool,
) -> dict[str, np.ndarray]:
    results: dict[str, list[np.ndarray]] = {"event_prediction": []}
    if module.uses_window_regression:
        results["window_prediction"] = []
    diagnostic_keys = (
        "modality_weights",
        "modality_difficulty",
        "valid_modality_mask",
        "temporal_weights",
        "kernel_mixture",
        "states",
        "probe_mean",
        "probe_log_variance",
    )
    if include_diagnostics:
        for key in diagnostic_keys:
            results[key] = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            tokens, mask, _target = regression_batch_tensors(dataset, batch, x_mean, x_std, target_mean, target_std, device)
            output = module(tokens, mask)
            results["event_prediction"].append(
                (output["prediction"].detach().cpu().numpy() * target_std + target_mean).astype(np.float32)
            )
            if module.uses_window_regression:
                results["window_prediction"].append(
                    (output["window_prediction"].detach().cpu().numpy() * target_std + target_mean).astype(np.float32)
                )
            if include_diagnostics:
                for key in diagnostic_keys:
                    results[key].append(output[key].detach().cpu().numpy())
    output_arrays: dict[str, np.ndarray] = {}
    for key, values in results.items():
        output_arrays[key] = np.concatenate(values, axis=0) if values else np.asarray([], dtype=np.float32)
    return output_arrays


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, subject_id: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_regression_with_centered(y_true, y_pred, subject_id)
    predictions = np.asarray(y_pred, dtype=np.float32)
    metrics.update(
        {
            "prediction_mean": float(np.mean(predictions)) if predictions.size else None,
            "prediction_std": float(np.std(predictions)) if predictions.size else None,
            "prediction_p10": float(np.quantile(predictions, 0.10)) if predictions.size else None,
            "prediction_p90": float(np.quantile(predictions, 0.90)) if predictions.size else None,
            "prediction_iqr": float(np.subtract(*np.quantile(predictions, [0.75, 0.25]))) if predictions.size else None,
        }
    )
    return metrics


def save_regression_predictions(
    path: Path,
    dataset: DailyAffectBagDataset,
    split: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "event_id": dataset.event_id,
        "subject_id": dataset.subject_id,
        "day_id": dataset.day_id,
        "label": dataset.label,
        "split": dataset.split,
        "pretrain_index": split["pretrain"],
        "finetune_index": split["finetune"],
        "train_index": split["train"],
        "val_index": split["val"],
        "test_index": split["test"],
    }
    for name, values in predictions.items():
        payload[f"{name}_event_prediction"] = values["event_prediction"]
        if "window_prediction" in values:
            payload[f"{name}_window_prediction"] = values["window_prediction"]
    np.savez_compressed(path, **payload)


def write_regression_prediction_table(
    path: Path, dataset: DailyAffectBagDataset, indices: np.ndarray, prediction: dict[str, np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True,)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bag_index", "event_id", "subject_id", "day_id", "label", "scalar_prediction"])
        for offset, index in enumerate(indices.tolist()):
            writer.writerow(
                [index, dataset.event_id[index], dataset.subject_id[index], dataset.day_id[index], float(dataset.label[index]), float(prediction["event_prediction"][offset])]
            )


def regression_history_csv(history: list[dict[str, Any]]) -> str:
    fields = [
        "epoch", "train_loss", "train_head_loss", "train_probe_nll", "routing_lambda_d", "val_rmse", "val_mae",
        "val_raw_r", "val_within_subject_centered_r", "selection_score",
    ]
    rows = [",".join(fields)]
    rows.extend(",".join("" if row.get(field) is None else str(row.get(field, "")) for field in fields) for row in history)
    return "\n".join(rows) + "\n"


def regression_config_snapshot(
    model: dict[str, Any], dataset: DailyAffectBagDataset, *, protocol: str, condition_id: str, seed: int
) -> dict[str, Any]:
    return {
        **model["config"],
        "protocol": protocol,
        "condition_id": condition_id,
        "seed": int(seed),
        "bag_path": str(dataset.bag_path),
        "route_id": dataset.route_id,
        "target_label": dataset.target_label,
        "row_count": dataset.row_count,
        "source_npz_json": dataset.source_npz_json,
        "supervision_boundary": dataset.supervision_boundary,
        "train_supervision": "pretrain_plus_finetune_train_val_early_stop_test_once",
        "target_scaling": {"fit": "train_only", "mean": float(model["target_mean"]), "std": float(model["target_std"])},
        "mask_contract": "EMA-level modality_mask with shape (N_ema,23,4)",
    }
