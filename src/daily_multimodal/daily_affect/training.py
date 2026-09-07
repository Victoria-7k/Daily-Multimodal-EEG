from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .diagnostics import DIAGNOSTIC_KEYS, stack_diagnostics
from .losses import (
    categorical_probe_loss,
    classification_loss,
    cumulative_probability_ordinal_loss,
    expected_score_huber_loss,
    ordinal_probe_loss,
    within_subject_ordinal_ranking_loss,
)
from .metrics import classification_metrics, metric_aliases
from .model import DailyAffectConfig, DailyAffectOrdinalModel
from .npz_compat import install_numpy_core_pickle_aliases
from .samplers import shuffled_batches
from .temperature_scaling import fit_probe_temperatures_grid


@dataclass(frozen=True)
class DailyAffectBagDataset:
    bag_path: Path
    tokens: np.ndarray
    modality_mask: np.ndarray
    label: np.ndarray
    label_zero_based: np.ndarray
    event_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    sample_id_matrix: np.ndarray
    split: np.ndarray
    route_id: str
    target_label: str
    source_npz_json: str
    supervision_boundary: str
    pretrain_index: np.ndarray
    finetune_index: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.tokens.shape[0])

    def split_indices(self) -> dict[str, np.ndarray]:
        return {
            "pretrain": self.pretrain_index,
            "finetune": self.finetune_index,
            "train": self.train_index,
            "val": self.val_index,
            "test": self.test_index,
        }


def load_bag_dataset(path: Path) -> DailyAffectBagDataset:
    install_numpy_core_pickle_aliases()
    with np.load(path, allow_pickle=True) as loaded:
        tokens = loaded["tokens"].astype(np.float32)
        mask = loaded["modality_mask"].astype(bool)
        if tokens.ndim != 4 or tokens.shape[1:] != (23, 4, 256):
            raise ValueError(f"{path} tokens must have shape (N,23,4,256), got {tokens.shape}")
        if mask.shape != tokens.shape[:3]:
            raise ValueError(f"{path} modality_mask must have shape {tokens.shape[:3]}, got {mask.shape}")
        split = loaded["split"].astype(str)
        if {"pretrain_index", "finetune_index", "train_index", "val_index", "test_index"}.issubset(set(loaded.files)):
            pretrain = loaded["pretrain_index"].astype(np.int64)
            finetune = loaded["finetune_index"].astype(np.int64)
            train = loaded["train_index"].astype(np.int64)
            val = loaded["val_index"].astype(np.int64)
            test = loaded["test_index"].astype(np.int64)
        else:
            pretrain = np.flatnonzero(split == "pretrain").astype(np.int64)
            finetune = np.flatnonzero(split == "finetune").astype(np.int64)
            train = np.asarray(pretrain.tolist() + finetune.tolist(), dtype=np.int64)
            val = np.flatnonzero(split == "val").astype(np.int64)
            test = np.flatnonzero(split == "test").astype(np.int64)
        for name, indices in {"train": train, "val": val, "test": test}.items():
            if len(indices) == 0:
                raise ValueError(f"{path} has empty {name} split")
        return DailyAffectBagDataset(
            bag_path=path,
            tokens=tokens,
            modality_mask=mask,
            label=loaded["label"].astype(np.float32),
            label_zero_based=loaded["label_zero_based"].astype(np.int64),
            event_id=loaded["event_id"].astype(str),
            subject_id=loaded["subject_id"].astype(str),
            day_id=loaded["day_id"].astype(str),
            sample_id_matrix=loaded["sample_id_matrix"].astype(str),
            split=split,
            route_id=str(loaded["route_id"].item()) if np.asarray(loaded["route_id"]).shape == () else str(loaded["route_id"]),
            target_label=str(loaded["target_label"].item()) if "target_label" in loaded.files else "fatigue",
            source_npz_json=str(loaded["source_npz_json"].item()) if "source_npz_json" in loaded.files else "{}",
            supervision_boundary=str(loaded["supervision_boundary"].item()) if "supervision_boundary" in loaded.files else "unknown",
            pretrain_index=pretrain,
            finetune_index=finetune,
            train_index=train,
            val_index=val,
            test_index=test,
        )


def run_daily_affect_run(
    *,
    dataset: DailyAffectBagDataset,
    protocol: str,
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
    beta_ord: float = 0.25,
    probe_loss_weight: float = 0.1,
    ordinal_loss_weight: float = 0.5,
    rank_loss_weight: float = 0.1,
    expected_score_loss_weight: float = 0.0,
    expected_score_huber_delta: float = 1.0,
    modality_dropout_prob: float = 0.1,
    probe_warmup_epochs: int = 5,
    difficulty_ramp_epochs: int = 5,
    probe_kind: str = "cumulative",
    difficulty_mode: str = "auto",
    detach_difficulty: bool = True,
    selection_metric: str = "qwk",
    class_balanced_loss: bool = True,
    calibrate_probe_temperature: bool = False,
    calibrate_temperature: bool | None = None,
    calibration_finetune_epochs: int = 5,
    temporal_policy: str = "uniform",
    device: str = "cuda",
    include_diagnostics: bool = True,
    adapter_mode: str | None = None,
    experiment_id: str = "default",
) -> dict[str, Any]:
    if calibrate_temperature is not None:
        # Compatibility for callers predating probe-specific calibration.
        calibrate_probe_temperature = bool(calibrate_temperature)
    model, audit = fit_daily_affect_model(
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
        beta_ord=beta_ord,
        probe_loss_weight=probe_loss_weight,
        ordinal_loss_weight=ordinal_loss_weight,
        rank_loss_weight=rank_loss_weight,
        expected_score_loss_weight=expected_score_loss_weight,
        expected_score_huber_delta=expected_score_huber_delta,
        modality_dropout_prob=modality_dropout_prob,
        probe_warmup_epochs=probe_warmup_epochs,
        difficulty_ramp_epochs=difficulty_ramp_epochs,
        probe_kind=probe_kind,
        difficulty_mode=difficulty_mode,
        detach_difficulty=detach_difficulty,
        selection_metric=selection_metric,
        class_balanced_loss=class_balanced_loss,
        temporal_policy=temporal_policy,
        device=device,
        adapter_mode=adapter_mode,
    )
    split = dataset.split_indices()
    predictions = {
        name: predict_daily_affect_model(model, dataset, indices=indices, device=device, include_diagnostics=False)
        for name, indices in split.items()
        if name in {"train", "val", "test"}
    }
    metrics = {
        name: metric_aliases(
            classification_metrics(
                dataset.label_zero_based[split[name]],
                predictions[name]["predicted_class"],
                expected_score=predictions[name]["expected_score"],
                subject_ids=dataset.subject_id[split[name]],
                probabilities=predictions[name]["probabilities"],
            )
        )
        for name in ("train", "val", "test")
    }
    calibration: dict[str, Any] | None = None
    if calibrate_probe_temperature:
        calibration, calibration_audit = calibrate_and_finetune_probe_routing(
            model=model,
            dataset=dataset,
            class_weights=class_weight_vector(dataset.label_zero_based[split["train"]]) if class_balanced_loss else None,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            rank_loss_weight=rank_loss_weight,
            ordinal_loss_weight=ordinal_loss_weight,
            expected_score_loss_weight=expected_score_loss_weight,
            expected_score_huber_delta=expected_score_huber_delta,
            selection_metric=selection_metric,
            epochs=calibration_finetune_epochs,
            device=device,
        )
        audit["calibration_finetune"] = calibration_audit
        audit["_best_state"] = {key: value.detach().cpu().clone() for key, value in model["module"].state_dict().items()}
        predictions = {
            name: predict_daily_affect_model(model, dataset, indices=indices, device=device, include_diagnostics=False)
            for name, indices in split.items()
            if name in {"train", "val", "test"}
        }
        metrics = {
            name: metric_aliases(
                classification_metrics(
                    dataset.label_zero_based[split[name]],
                    predictions[name]["predicted_class"],
                    expected_score=predictions[name]["expected_score"],
                    subject_ids=dataset.subject_id[split[name]],
                    probabilities=predictions[name]["probabilities"],
                )
            )
            for name in ("train", "val", "test")
        }
    result = {
        "protocol": protocol,
        "route_id": dataset.route_id,
        "model_id": model_id,
        "normalization": normalization,
        "adapter_mode": model["config"]["adapter_mode"],
        "objective_id": model["config"]["objective_id"],
        "supervision_unit": model["config"]["supervision_unit"],
        "routing_id": model["config"]["routing_id"],
        "experiment_id": str(experiment_id),
        "seed": int(seed),
        "bag_path": str(dataset.bag_path),
        "target_label": dataset.target_label,
        "row_count": dataset.row_count,
        "split_counts": {name: int(len(indices)) for name, indices in split.items()},
        "supervision_boundary": dataset.supervision_boundary,
        "train": metrics["train"],
        "val": metrics["val"],
        "test": metrics["test"],
        "probe_calibration": calibration,
        "train_audit": audit,
    }
    model["config"]["experiment_id"] = str(experiment_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.npz"
    save_predictions(pred_path, dataset, split, predictions)
    diagnostics_path = None
    if include_diagnostics:
        all_indices = np.arange(dataset.row_count, dtype=np.int64)
        diagnostics = predict_daily_affect_model(model, dataset, indices=all_indices, device=device, include_diagnostics=True)
        diagnostics_path = run_dir / "diagnostics.npz"
        save_diagnostics(diagnostics_path, diagnostics)
    _write_predictions_table(run_dir / "test_predictions.csv", dataset, split["test"], predictions["test"])
    state = audit.pop("_best_state")
    torch.save(
        {
            "state_dict": state,
            "model_id": model_id,
            "normalization": normalization,
            "seed": int(seed),
            "config": model["config"],
            "x_mean": model["x_mean"],
            "x_std": model["x_std"],
        },
        run_dir / "best_checkpoint.pt",
    )
    (run_dir / "val_history.csv").write_text(history_csv(audit["history"]), encoding="utf-8")
    result["prediction_path"] = str(pred_path)
    result["diagnostics_path"] = str(diagnostics_path) if diagnostics_path is not None else None
    result["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")
    result["test_predictions_csv"] = str(run_dir / "test_predictions.csv")
    result["metrics_path"] = str(run_dir / "metrics.json")
    result["config_path"] = str(run_dir / "config.json")
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(config_snapshot(model, dataset, protocol=protocol, seed=seed), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def fit_daily_affect_model(
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
    beta_ord: float,
    probe_loss_weight: float,
    ordinal_loss_weight: float,
    rank_loss_weight: float,
    expected_score_loss_weight: float,
    expected_score_huber_delta: float,
    modality_dropout_prob: float,
    probe_warmup_epochs: int,
    difficulty_ramp_epochs: int,
    probe_kind: str,
    difficulty_mode: str,
    detach_difficulty: bool,
    selection_metric: str,
    class_balanced_loss: bool,
    temporal_policy: str,
    device: str,
    adapter_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if normalization not in {"shared", "per_modality"}:
        raise ValueError(f"unsupported normalization: {normalization}")
    adapter_mode = normalization if adapter_mode is None else adapter_mode
    if adapter_mode not in {"shared", "per_modality"}:
        raise ValueError(f"unsupported adapter_mode: {adapter_mode}")
    if float(expected_score_loss_weight) < 0.0:
        raise ValueError("expected_score_loss_weight must be nonnegative")
    if float(expected_score_huber_delta) <= 0.0:
        raise ValueError("expected_score_huber_delta must be positive")
    _seed_everything(seed)
    train = dataset.train_index
    val = dataset.val_index
    x_mean, x_std = fit_token_normalization(dataset.tokens, dataset.modality_mask, train, scope=normalization)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    cfg = DailyAffectConfig(
        model_id=model_id,
        hidden_dim=hidden_dim,
        adapter_mode=adapter_mode,
        dropout=dropout,
        lambda_d=lambda_d,
        beta_ord=beta_ord,
        probe_kind=probe_kind,
        difficulty_mode=difficulty_mode,
        detach_difficulty=detach_difficulty,
        temporal_policy=temporal_policy,
    )
    module = DailyAffectOrdinalModel(cfg).to(dev)
    if module.uses_window_replicated_supervision and float(expected_score_loss_weight) > 0.0:
        raise ValueError("expected-score Huber supervision is event-level and cannot be applied to window_replicated")
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    class_weights = None
    if class_balanced_loss:
        class_weights = torch.as_tensor(class_weight_vector(dataset.label_zero_based[train]), dtype=torch.float32, device=dev)
    subject_codes = subject_code_vector(dataset.subject_id)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(max(1, int(epochs))):
        module.train()
        losses: list[float] = []
        head_losses: list[float] = []
        ordinal_losses: list[float] = []
        rank_losses: list[float] = []
        expected_score_losses: list[float] = []
        probe_losses: list[float] = []
        scheduled_lambda_d = difficulty_lambda_for_epoch(
            epoch,
            target=lambda_d,
            warmup_epochs=probe_warmup_epochs,
            ramp_epochs=difficulty_ramp_epochs,
        )
        for batch in shuffled_batches(train, batch_size, rng):
            tokens, mask, labels_zero, raw_labels = batch_tensors(dataset, batch, x_mean, x_std, dev)
            tokens, mask = apply_modality_dropout(tokens, mask, probability=modality_dropout_prob, rng=rng)
            outputs = module(
                tokens,
                mask,
                lambda_d_override=scheduled_lambda_d,
                detach_difficulty_override=detach_difficulty,
            )
            batch_subject_codes = torch.as_tensor(subject_codes[batch], dtype=torch.long, device=dev)
            loss_logits, loss_labels, loss_raw_labels, rank_scores, rank_subject_codes = supervision_tensors(
                outputs,
                labels_zero,
                raw_labels,
                batch_subject_codes,
            )
            ce_loss = classification_loss(loss_logits, loss_labels, class_weights=class_weights)
            ordinal_loss = cumulative_probability_ordinal_loss(loss_logits, loss_labels)
            head_loss = ce_loss + float(ordinal_loss_weight) * ordinal_loss
            rank_loss = within_subject_ordinal_ranking_loss(rank_scores, loss_raw_labels, rank_subject_codes)
            score_loss = expected_score_huber_loss(
                outputs["expected_score"], raw_labels, delta=expected_score_huber_delta
            )
            probe_loss = probe_loss_for_outputs(outputs, labels_zero, raw_labels, module.uses_probe, probe_kind)
            loss = (
                head_loss
                + float(rank_loss_weight) * rank_loss
                + float(expected_score_loss_weight) * score_loss
                + (float(probe_loss_weight) * probe_loss if module.uses_probe else 0.0)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            head_losses.append(float(head_loss.detach().cpu().item()))
            ordinal_losses.append(float(ordinal_loss.detach().cpu().item()))
            rank_losses.append(float(rank_loss.detach().cpu().item()))
            expected_score_losses.append(float(score_loss.detach().cpu().item()))
            probe_losses.append(float(probe_loss.detach().cpu().item()))
        module.eval()
        val_pred = _predict_module(module, dataset, val, x_mean, x_std, dev, include_diagnostics=False)
        val_metrics = metric_aliases(
            classification_metrics(
                dataset.label_zero_based[val],
                val_pred["predicted_class"],
                expected_score=val_pred["expected_score"],
                subject_ids=dataset.subject_id[val],
                probabilities=val_pred["probabilities"],
            )
        )
        score = selection_score(val_metrics, selection_metric)
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "train_head_loss": float(np.mean(head_losses)) if head_losses else math.nan,
            "train_ordinal_loss": float(np.mean(ordinal_losses)) if ordinal_losses else math.nan,
            "train_rank_loss": float(np.mean(rank_losses)) if rank_losses else math.nan,
            "train_expected_score_huber_loss": float(np.mean(expected_score_losses)) if expected_score_losses else math.nan,
            "train_probe_loss": float(np.mean(probe_losses)) if probe_losses else math.nan,
            "routing_lambda_d": float(scheduled_lambda_d),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_macro_f1": val_metrics.get("macro_f1"),
            "val_qwk": val_metrics.get("qwk"),
            "val_ordinal_mae": val_metrics.get("ordinal_mae"),
            "val_nll": val_metrics.get("nll"),
            "selection_score": float(score),
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no best state")
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
        "beta_ord": float(beta_ord),
        "probe_kind": probe_kind,
        "difficulty_mode": difficulty_mode,
        "detach_difficulty": bool(detach_difficulty),
        "probe_loss_weight": float(probe_loss_weight),
        "ordinal_loss_weight": float(ordinal_loss_weight),
        "rank_loss_weight": float(rank_loss_weight),
        "expected_score_loss_weight": float(expected_score_loss_weight),
        "expected_score_huber_delta": float(expected_score_huber_delta),
        "modality_dropout_prob": float(modality_dropout_prob),
        "probe_warmup_epochs": int(probe_warmup_epochs),
        "difficulty_ramp_epochs": int(difficulty_ramp_epochs),
        "selection_metric": selection_metric,
        "class_balanced_loss": bool(class_balanced_loss),
        "objective_id": objective_id(
            class_balanced_loss,
            ordinal_loss_weight,
            rank_loss_weight,
            expected_score_loss_weight,
            expected_score_huber_delta,
        ),
        "supervision_unit": "window_replicated" if module.uses_window_replicated_supervision else "ema_bag",
        "resolved_difficulty_mode": module.resolved_difficulty_mode,
        "routing_id": routing_id(probe_kind, module.resolved_difficulty_mode, beta_ord, detach_difficulty),
        "probe_calibrated": False,
        "temporal_policy": temporal_policy,
    }
    return {
        "module": module,
        "x_mean": x_mean,
        "x_std": x_std,
        "config": config,
    }, {
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "selection_metric": selection_metric,
        "epoch_count": int(len(history)),
        "initial_train_loss": history[0]["train_loss"] if history else math.nan,
        "final_train_loss": history[-1]["train_loss"] if history else math.nan,
        "trainable_params": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in module.parameters())),
        "history": history,
        "_best_state": best_state,
    }


def calibrate_and_finetune_probe_routing(
    *,
    model: dict[str, Any],
    dataset: DailyAffectBagDataset,
    class_weights: np.ndarray | None,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    rank_loss_weight: float,
    ordinal_loss_weight: float,
    expected_score_loss_weight: float,
    expected_score_huber_delta: float,
    selection_metric: str,
    epochs: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate cumulative modality Probes on validation data, then tune routing layers.

    Calibration is deliberately separate from the main-head temperature scaling
    used by older runs.  Each modality receives one scalar temperature shared by
    all four cumulative logits.  The adapter and Probe stay frozen during the
    short post-calibration tuning stage.
    """

    module: DailyAffectOrdinalModel = model["module"]
    if not module.uses_probe or not module.uses_prior_guidance:
        raise ValueError("probe temperature calibration requires a prior-guided model with an active modality Probe")
    if module.config.probe_kind != "cumulative":
        raise ValueError("probe temperature calibration requires probe_kind='cumulative'")
    dev = torch.device(device)
    val = dataset.val_index
    val_outputs = _predict_module(module, dataset, val, model["x_mean"], model["x_std"], dev, include_diagnostics=True)
    fit = fit_probe_temperatures_grid(
        val_outputs["probe_ordinal_logits"],
        dataset.label_zero_based[val],
        val_outputs["valid_modality_mask"],
    )
    temperatures = torch.as_tensor(fit["temperatures"], dtype=torch.float32, device=dev)
    module.set_probe_temperatures(temperatures)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable_modules = (module.state_filter, module.kernel, module.global_kernel, module.head)
    for trainable in trainable_modules:
        if trainable is not None:
            for parameter in trainable.parameters():
                parameter.requires_grad_(True)
    trainable_parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("probe calibration left no trainable routing or head parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    weights = None
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=dev)
    train = dataset.train_index
    subject_codes = subject_code_vector(dataset.subject_id)
    rng = np.random.default_rng(seed + 7919)
    best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    best_score = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(max(1, int(epochs))):
        module.train()
        epoch_losses: list[float] = []
        for batch in shuffled_batches(train, min(128, max(1, len(train))), rng):
            tokens, mask, labels_zero, raw_labels = batch_tensors(dataset, batch, model["x_mean"], model["x_std"], dev)
            outputs = module(tokens, mask)
            ce_loss = classification_loss(outputs["class_logits"], labels_zero, class_weights=weights)
            ordinal_loss = cumulative_probability_ordinal_loss(outputs["class_logits"], labels_zero)
            subject_batch = torch.as_tensor(subject_codes[batch], dtype=torch.long, device=dev)
            rank_loss = within_subject_ordinal_ranking_loss(outputs["expected_score"], raw_labels, subject_batch)
            score_loss = expected_score_huber_loss(
                outputs["expected_score"], raw_labels, delta=expected_score_huber_delta
            )
            loss = (
                ce_loss
                + float(ordinal_loss_weight) * ordinal_loss
                + float(rank_loss_weight) * rank_loss
                + float(expected_score_loss_weight) * score_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        module.eval()
        val_pred = _predict_module(module, dataset, val, model["x_mean"], model["x_std"], dev, include_diagnostics=False)
        val_metrics = metric_aliases(
            classification_metrics(
                dataset.label_zero_based[val],
                val_pred["predicted_class"],
                expected_score=val_pred["expected_score"],
                subject_ids=dataset.subject_id[val],
                probabilities=val_pred["probabilities"],
            )
        )
        score = selection_score(val_metrics, selection_metric)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "selection_score": score})
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    module.load_state_dict(best_state)
    model["config"]["probe_calibrated"] = True
    model["config"]["probe_temperatures"] = [float(value) for value in fit["temperatures"]]
    return (
        {
            **fit,
            "kind": "per_modality_cumulative_probe_temperature",
            "post_calibration_tuning": "state_filter_global_or_dynamic_kernel_and_head_only",
        },
        {
            "epochs": int(max(1, int(epochs))),
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in trainable_parameters)),
            "history": history,
        },
    )


def predict_daily_affect_model(
    model: dict[str, Any],
    dataset: DailyAffectBagDataset,
    *,
    indices: np.ndarray,
    device: str,
    include_diagnostics: bool,
) -> dict[str, Any]:
    module: DailyAffectOrdinalModel = model["module"]
    dev = torch.device(device)
    module.eval()
    return _predict_module(module, dataset, np.asarray(indices, dtype=np.int64), model["x_mean"], model["x_std"], dev, include_diagnostics=include_diagnostics)


def supervision_tensors(
    outputs: dict[str, torch.Tensor],
    labels_zero: torch.Tensor,
    raw_labels: torch.Tensor,
    subject_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select event- or copied-window targets while preserving an event-level evaluator."""

    if "window_class_logits" not in outputs:
        return outputs["class_logits"], labels_zero, raw_labels, outputs["expected_score"], subject_codes
    window_mask = outputs["window_mask"].reshape(-1)
    if not torch.any(window_mask):
        raise ValueError("window-replicated supervision received a batch without valid modality windows")
    sequence_len = outputs["window_class_logits"].shape[1]
    repeated_labels = labels_zero[:, None].expand(-1, sequence_len).reshape(-1)
    repeated_raw_labels = raw_labels[:, None].expand(-1, sequence_len).reshape(-1)
    repeated_subject_codes = subject_codes[:, None].expand(-1, sequence_len).reshape(-1)
    return (
        outputs["window_class_logits"].reshape(-1, 5)[window_mask],
        repeated_labels[window_mask],
        repeated_raw_labels[window_mask],
        outputs["window_expected_score"].reshape(-1)[window_mask],
        repeated_subject_codes[window_mask],
    )


def _predict_module(
    module: DailyAffectOrdinalModel,
    dataset: DailyAffectBagDataset,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
    *,
    include_diagnostics: bool,
) -> dict[str, Any]:
    logits: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    diagnostic_chunks: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            tokens, mask, _labels_zero, _raw = batch_tensors(dataset, batch, x_mean, x_std, device)
            outputs = module(tokens, mask)
            logits.append(outputs["class_logits"].detach().cpu().numpy().astype(np.float32))
            probs.append(outputs["probabilities"].detach().cpu().numpy().astype(np.float32))
            expected.append(outputs["expected_score"].detach().cpu().numpy().astype(np.float32))
            predicted.append(outputs["predicted_class"].detach().cpu().numpy().astype(np.int64))
            if include_diagnostics:
                diagnostic_chunks.append(
                    {
                        key: outputs[key].detach().cpu().numpy().astype(np.float32)
                        for key in DIAGNOSTIC_KEYS
                        if key in outputs
                    }
                )
    result: dict[str, Any] = {
        "logits": np.concatenate(logits, axis=0) if logits else np.zeros((0, 5), dtype=np.float32),
        "probabilities": np.concatenate(probs, axis=0) if probs else np.zeros((0, 5), dtype=np.float32),
        "expected_score": np.concatenate(expected, axis=0) if expected else np.zeros((0,), dtype=np.float32),
        "predicted_class": np.concatenate(predicted, axis=0) if predicted else np.zeros((0,), dtype=np.int64),
    }
    if include_diagnostics:
        result.update(stack_diagnostics(diagnostic_chunks))
    return result


def batch_tensors(
    dataset: DailyAffectBagDataset,
    indices: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = normalize_tokens(dataset.tokens[indices], x_mean, x_std)
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(dataset.modality_mask[indices], dtype=torch.bool, device=device),
        torch.as_tensor(dataset.label_zero_based[indices], dtype=torch.long, device=device),
        torch.as_tensor(dataset.label[indices], dtype=torch.float32, device=device),
    )


def apply_modality_dropout(
    tokens: torch.Tensor,
    modality_mask: torch.Tensor,
    *,
    probability: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly remove one whole valid modality from selected EMA bags.

    The mask is bag-wide for the selected modality, matching realistic sensor
    absence while preserving at least one usable modality per bag.
    """

    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"modality dropout probability must be in [0, 1], got {probability}")
    if value == 0.0 or tokens.shape[0] == 0:
        return tokens, modality_mask
    dropped_tokens = tokens.clone()
    dropped_mask = modality_mask.clone()
    active = modality_mask.any(dim=1)
    for row in range(tokens.shape[0]):
        choices = torch.nonzero(active[row], as_tuple=False).flatten().detach().cpu().numpy()
        if len(choices) <= 1 or float(rng.random()) >= value:
            continue
        modality = int(rng.choice(choices))
        dropped_mask[row, :, modality] = False
        dropped_tokens[row, :, modality, :] = 0.0
    return dropped_tokens, dropped_mask


def difficulty_lambda_for_epoch(
    epoch_zero_based: int,
    *,
    target: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    if epoch_zero_based < max(0, int(warmup_epochs)):
        return 0.0
    ramp = max(0, int(ramp_epochs))
    if ramp == 0:
        return float(target)
    progress = min(1.0, (epoch_zero_based - max(0, int(warmup_epochs)) + 1) / float(ramp))
    return float(target) * progress


def subject_code_vector(subject_ids: np.ndarray) -> np.ndarray:
    labels = np.asarray(subject_ids).astype(str)
    _, codes = np.unique(labels, return_inverse=True)
    return codes.astype(np.int64)


def probe_loss_for_outputs(
    outputs: dict[str, torch.Tensor],
    labels_zero_based: torch.Tensor,
    raw_labels: torch.Tensor,
    enabled: bool,
    probe_kind: str,
) -> torch.Tensor:
    if not enabled:
        return outputs["class_logits"].new_zeros(())
    valid = outputs["valid_modality_mask"]
    if probe_kind == "cumulative":
        return ordinal_probe_loss(outputs["probe_ordinal_logits"], raw_labels, valid)
    if probe_kind == "categorical":
        return categorical_probe_loss(outputs["probe_class_logits"], labels_zero_based, valid)
    raise ValueError(f"unsupported probe kind: {probe_kind}")


def objective_id(
    class_balanced_loss: bool,
    ordinal_loss_weight: float,
    rank_loss_weight: float,
    expected_score_loss_weight: float,
    expected_score_huber_delta: float,
) -> str:
    ce = "weighted_ce" if class_balanced_loss else "ce"
    return (
        f"{ce}__ord_{float(ordinal_loss_weight):g}__rank_{float(rank_loss_weight):g}"
        f"__scorehuber_{float(expected_score_loss_weight):g}_delta_{float(expected_score_huber_delta):g}"
    )


def routing_id(probe_kind: str, difficulty_mode: str, beta_ord: float, detach_difficulty: bool) -> str:
    return (
        f"probe_{probe_kind}__difficulty_{difficulty_mode}__beta_{float(beta_ord):g}"
        f"__detach_{str(bool(detach_difficulty)).lower()}"
    )


def fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray, *, scope: str) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, :, None].astype(bool), tokens[indices], np.nan)
    if scope == "shared":
        mean = np.nanmean(available, axis=(0, 1, 2), keepdims=True)
        std = np.nanstd(available, axis=(0, 1, 2), keepdims=True)
    elif scope == "per_modality":
        mean = np.nanmean(available, axis=(0, 1), keepdims=True)
        std = np.nanstd(available, axis=(0, 1), keepdims=True)
    else:
        raise ValueError(f"unsupported normalization scope: {scope}")
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def class_weight_vector(labels_zero_based: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(labels_zero_based, dtype=np.int64), minlength=5).astype(np.float32)
    nonzero = counts[counts > 0]
    mean = float(nonzero.mean()) if nonzero.size else 1.0
    weights = np.where(counts > 0, mean / np.maximum(counts, 1.0), 1.0)
    return np.clip(weights, 0.5, 3.0).astype(np.float32)


def selection_score(metrics: dict[str, Any], metric: str) -> float:
    if metric == "qwk":
        value = metrics.get("qwk")
        return float(value) if value is not None else float(metrics.get("macro_f1") or 0.0)
    if metric == "macro_f1":
        return float(metrics.get("macro_f1") or 0.0)
    if metric == "accuracy":
        return float(metrics.get("accuracy") or 0.0)
    if metric == "ordinal_mae":
        return -float(metrics.get("ordinal_mae") or 1e9)
    if metric == "nll":
        return -float(metrics.get("nll") or 1e9)
    raise ValueError(f"unsupported selection metric: {metric}")


def save_predictions(path: Path, dataset: DailyAffectBagDataset, split: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "event_id": dataset.event_id,
        "subject_id": dataset.subject_id,
        "day_id": dataset.day_id,
        "label": dataset.label,
        "label_zero_based": dataset.label_zero_based,
        "split": dataset.split,
        "pretrain_index": split["pretrain"],
        "finetune_index": split["finetune"],
        "train_index": split["train"],
        "val_index": split["val"],
        "test_index": split["test"],
    }
    for name, values in predictions.items():
        payload[f"{name}_logits"] = values["logits"]
        payload[f"{name}_probabilities"] = values["probabilities"]
        payload[f"{name}_expected_score"] = values["expected_score"]
        payload[f"{name}_predicted_class"] = values["predicted_class"]
    np.savez_compressed(path, **payload)


def save_diagnostics(path: Path, diagnostics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: diagnostics[key] for key in DIAGNOSTIC_KEYS if key in diagnostics}
    np.savez_compressed(path, **payload)


def _write_predictions_table(path: Path, dataset: DailyAffectBagDataset, indices: np.ndarray, prediction: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bag_index",
                "event_id",
                "subject_id",
                "day_id",
                "label",
                "predicted_label",
                "expected_score",
                "correct",
                "prob_1",
                "prob_2",
                "prob_3",
                "prob_4",
                "prob_5",
            ]
        )
        for offset, bag_index in enumerate(indices.tolist()):
            pred_zero = int(prediction["predicted_class"][offset])
            probs = prediction["probabilities"][offset].tolist()
            writer.writerow(
                [
                    bag_index,
                    dataset.event_id[bag_index],
                    dataset.subject_id[bag_index],
                    dataset.day_id[bag_index],
                    int(dataset.label_zero_based[bag_index]) + 1,
                    pred_zero + 1,
                    float(prediction["expected_score"][offset]),
                    int(pred_zero == int(dataset.label_zero_based[bag_index])),
                    *[float(value) for value in probs],
                ]
            )


def history_csv(history: list[dict[str, Any]]) -> str:
    fields = [
        "epoch",
        "train_loss",
        "train_head_loss",
        "train_ordinal_loss",
        "train_rank_loss",
        "train_expected_score_huber_loss",
        "train_probe_loss",
        "routing_lambda_d",
        "val_accuracy",
        "val_macro_f1",
        "val_qwk",
        "val_ordinal_mae",
        "val_nll",
        "selection_score",
    ]
    rows = [",".join(fields)]
    rows.extend(",".join("" if item.get(field) is None else str(item.get(field, "")) for field in fields) for item in history)
    return "\n".join(rows) + "\n"


def config_snapshot(model: dict[str, Any], dataset: DailyAffectBagDataset, *, protocol: str, seed: int) -> dict[str, Any]:
    return {
        **model["config"],
        "protocol": protocol,
        "seed": int(seed),
        "bag_path": str(dataset.bag_path),
        "route_id": dataset.route_id,
        "target_label": dataset.target_label,
        "row_count": dataset.row_count,
        "source_npz_json": dataset.source_npz_json,
        "supervision_boundary": dataset.supervision_boundary,
        "train_supervision": "pretrain_plus_finetune_train_val_early_stop_test_once",
        "mask_contract": "EMA-level modality_mask with shape (N_ema,23,4)",
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
