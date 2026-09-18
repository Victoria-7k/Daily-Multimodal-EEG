#!/usr/bin/env python3
"""Run fixed-token 11-label fusion for the EEG-aligned multimodal route.

This script is intentionally separate from the single-target fatigue and
calibration scripts.  It implements the first multi-label handoff step:
fixed/frozen modality tokens -> fusion encoder -> 11-label regression head.
"""

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
POSITIVE_LABELS = {"inspired", "alert", "determined", "attentive", "active"}
NEGATIVE_LABELS = {"hostile", "nervous", "upset", "afraid", "ashamed"}
DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = DEFAULT_ROOT / "outputs/splits"
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_PROTOCOLS = ("cross_day", "within_subject_day")


@dataclass(frozen=True)
class Branch:
    name: str
    modality: str
    filename: str
    emb_key: str
    mask_key: str
    modality_index: int


BRANCHES = {
    "eeg_eegpt_frozen_v1": Branch(
        "eeg_eegpt_frozen_v1",
        "eeg",
        "eeg/eeg_eegpt_eeg23win_embeddings.npz",
        "eeg_emb",
        "eeg_mask",
        0,
    ),
    "wear_physio": Branch(
        "wear_physio",
        "wear",
        "wear/wear_physio_preprocessed_eeg23win_embeddings.npz",
        "wear_emb",
        "wear_mask",
        1,
    ),
    "wear_deep": Branch(
        "wear_deep",
        "wear",
        "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz",
        "wear_emb",
        "wear_mask",
        1,
    ),
    "video_B0": Branch("video_B0", "video", "video/video_B0_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A1": Branch("video_A1", "video", "video/video_A1_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A2": Branch("video_A2", "video", "video/video_A2_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "audio": Branch("audio", "audio", "audio/audio_opensmile_eeg23win_embeddings.npz", "audio_emb", "audio_mask", 3),
}

EXPERIMENT_BRANCHES = {
    "B0_Wphysio_full": ("eeg_eegpt_frozen_v1", "wear_physio", "video_B0", "audio"),
    "B0_Wphysio_no_audio": ("eeg_eegpt_frozen_v1", "wear_physio", "video_B0"),
    "B0_Wdeep_full": ("eeg_eegpt_frozen_v1", "wear_deep", "video_B0", "audio"),
    "B0_Wdeep_no_audio": ("eeg_eegpt_frozen_v1", "wear_deep", "video_B0"),
    "A1_Wphysio_full": ("eeg_eegpt_frozen_v1", "wear_physio", "video_A1", "audio"),
    "A1_Wphysio_no_audio": ("eeg_eegpt_frozen_v1", "wear_physio", "video_A1"),
    "A1_Wdeep_full": ("eeg_eegpt_frozen_v1", "wear_deep", "video_A1", "audio"),
    "A1_Wdeep_no_audio": ("eeg_eegpt_frozen_v1", "wear_deep", "video_A1"),
    "A2_Wphysio_full": ("eeg_eegpt_frozen_v1", "wear_physio", "video_A2", "audio"),
    "A2_Wphysio_no_audio": ("eeg_eegpt_frozen_v1", "wear_physio", "video_A2"),
    "A2_Wdeep_full": ("eeg_eegpt_frozen_v1", "wear_deep", "video_A2", "audio"),
    "A2_Wdeep_no_audio": ("eeg_eegpt_frozen_v1", "wear_deep", "video_A2"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--experiments", default=",".join(EXPERIMENT_BRANCHES))
    parser.add_argument("--seeds", default="240800")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int, default=0, help="Optional smoke limit. 0 means all requested runs.")
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
    event_window_id = np.asarray([int(row.get("event_window_id", -1)) for row in rows], dtype=np.int64)
    targets = _load_targets(rows)
    label_audit = _audit_labels(targets, rows, event_window_id)
    if not label_audit["gate_passed"]:
        output = _output_header(args, label_audit, [])
        _write_json(output, args.out_json)
        _write_markdown(output, args.out_md)
        print("label_audit_gate_passed=false")
        print(f"out_json={args.out_json}")
        print(f"out_md={args.out_md}")
        return 2

    protocols = _split_csv(args.protocols)
    experiments = _split_csv(args.experiments)
    seeds = [int(value) for value in _split_csv(args.seeds)]
    for experiment in experiments:
        if experiment not in EXPERIMENT_BRANCHES:
            raise ValueError(f"unsupported experiment: {experiment}")

    branch_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    run_count = 0
    for protocol in protocols:
        split = _load_split(args.splits_root / protocol, len(rows))
        for experiment in experiments:
            branches = EXPERIMENT_BRANCHES[experiment]
            branch_data = _load_all_branches(args.embeddings_root, sample_id, branches, branch_cache)
            tokens, token_mask, branch_report = _build_tokens(branch_data, branches)
            for seed in seeds:
                run_count += 1
                if args.limit_runs and run_count > args.limit_runs:
                    break
                print(f"starting protocol={protocol} experiment={experiment} seed={seed}", flush=True)
                model, train_audit = _fit_model(
                    tokens=tokens,
                    token_mask=token_mask,
                    targets=targets,
                    train_idx=split["train"],
                    val_idx=split["val"],
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    patience=args.patience,
                    seed=seed,
                    device=args.device,
                )
                predictions = {
                    name: _predict(model, tokens, token_mask, indices=indices, device=args.device)
                    for name, indices in split.items()
                    if name in {"train", "val", "test"}
                }
                result = {
                    "protocol": protocol,
                    "experiment": experiment,
                    "route_name": "fixed_tokens_multitask_fusion_v1",
                    "seed": int(seed),
                    "branches": list(branches),
                    "enabled_modalities": [BRANCHES[name].modality for name in branches],
                    "row_count": len(rows),
                    "label_names": list(LABEL_NAMES),
                    "supervision_boundary": "fixed_frozen_tokens_train_fusion_and_11label_head_only",
                    "split_counts": {name: int(len(values)) for name, values in split.items()},
                    "mask_coverage_by_split": _mask_coverage(token_mask, branches, split),
                    "train": _evaluate_multilabel(targets[split["train"]], predictions["train"], subject_id[split["train"]]),
                    "val": _evaluate_multilabel(targets[split["val"]], predictions["val"], subject_id[split["val"]]),
                    "test": _evaluate_multilabel(targets[split["test"]], predictions["test"], subject_id[split["test"]]),
                    "train_audit": train_audit,
                    "branch_report": branch_report,
                }
                if args.predictions_dir:
                    pred_path = args.predictions_dir / protocol / experiment / f"seed_{seed}.npz"
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        pred_path,
                        train_index=split["train"],
                        val_index=split["val"],
                        test_index=split["test"],
                        train_prediction=predictions["train"],
                        val_prediction=predictions["val"],
                        test_prediction=predictions["test"],
                        target=targets,
                        label_names=np.asarray(LABEL_NAMES, dtype=object),
                        sample_id=sample_id,
                        subject_id=subject_id,
                        event_id=event_id,
                    )
                    result["prediction_path"] = str(pred_path)
                results.append(result)
                test_summary = result["test"]["summary"]
                print(
                    f"completed protocol={protocol} experiment={experiment} seed={seed} "
                    f"mean_rmse={_fmt(test_summary['mean_rmse'])} mean_raw_r={_fmt(test_summary['mean_raw_r'])} "
                    f"fatigue_raw_r={_fmt(result['test']['per_label']['fatigue']['raw_r'])}",
                    flush=True,
                )
            if args.limit_runs and run_count >= args.limit_runs:
                break
        if args.limit_runs and run_count >= args.limit_runs:
            break

    output = _output_header(args, label_audit, results)
    _write_json(output, args.out_json)
    _write_markdown(output, args.out_md)
    print(f"run_count={len(results)}")
    print(f"out_json={args.out_json}")
    print(f"out_md={args.out_md}")
    return 0


class MultiLabelAttentionRegressor(torch.nn.Module):
    def __init__(self, *, modality_count: int, hidden_dim: int, dropout: float, output_dim: int) -> None:
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
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(tokens) + self.modality_embedding[:, : tokens.shape[1], :]
        hidden = hidden.masked_fill(~mask[:, :, None], 0.0)
        attended, _ = self.self_attention(hidden, hidden, hidden, key_padding_mask=~mask, need_weights=False)
        attended = attended.masked_fill(~mask[:, :, None], 0.0)
        scores = torch.einsum("bmh,h->bm", attended, self.query)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(attended * weights, dim=1)
        return self.head(self.dropout(pooled))


def _fit_model(
    *,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    targets: np.ndarray,
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
    x_mean, x_std = _fit_token_normalization(tokens, token_mask, train_idx)
    y_mean = targets[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = targets[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(np.isfinite(y_std) & (y_std >= 1e-6), y_std, 1.0).astype(np.float32)
    dev = torch.device(device)
    module = MultiLabelAttentionRegressor(
        modality_count=tokens.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_dim=len(LABEL_NAMES),
    ).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x_train = torch.as_tensor(_normalize_tokens(tokens[train_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_train = torch.as_tensor(token_mask[train_idx], dtype=torch.bool, device=dev)
    y_train = torch.as_tensor((targets[train_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    x_val = torch.as_tensor(_normalize_tokens(tokens[val_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_val = torch.as_tensor(token_mask[val_idx], dtype=torch.bool, device=dev)
    y_val = torch.as_tensor((targets[val_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        module.train()
        losses: list[float] = []
        for batch in _make_batches(len(train_idx), batch_size, rng):
            prediction = module(x_train[batch], m_train[batch])
            loss = torch.mean((prediction - y_train[batch]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        module.eval()
        with torch.no_grad():
            val_prediction = module(x_val, m_val)
            val_loss = float(torch.mean((val_prediction - y_val) ** 2).detach().cpu().item())
        audit = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "val_loss": val_loss,
        }
        history.append(audit)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
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
        "final_train_loss": history[-1]["train_loss"] if history else math.nan,
        "final_val_loss": history[-1]["val_loss"] if history else math.nan,
        "epoch_count": len(history),
        "history": history,
        "target_normalization": {
            name: {"mean": float(y_mean[0, idx]), "std": float(y_std[0, idx])}
            for idx, name in enumerate(LABEL_NAMES)
        },
        "feature_normalization": "train_only_available_tokens",
        "loss": "mean_equal_weight_standardized_label_mse",
    }


def _predict(model: dict[str, Any], tokens: np.ndarray, mask: np.ndarray, *, indices: np.ndarray, device: str) -> np.ndarray:
    module: MultiLabelAttentionRegressor = model["module"]
    dev = torch.device(device)
    values = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            chunk = indices[start : start + 1024]
            x = torch.as_tensor(_normalize_tokens(tokens[chunk], model["x_mean"], model["x_std"]), dtype=torch.float32, device=dev)
            m = torch.as_tensor(mask[chunk], dtype=torch.bool, device=dev)
            pred = module(x, m).detach().cpu().numpy()
            values.append((pred * model["y_std"] + model["y_mean"]).astype(np.float32))
    return np.concatenate(values, axis=0) if values else np.zeros((0, len(LABEL_NAMES)), dtype=np.float32)


def _evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: np.ndarray) -> dict[str, Any]:
    per_label = {}
    for idx, label in enumerate(LABEL_NAMES):
        true = y_true[:, idx].astype(np.float32)
        pred = y_pred[:, idx].astype(np.float32)
        err = pred - true
        centered_true, centered_pred = within_subject_centered_arrays(true, pred, subject_ids)
        per_label[label] = {
            "count": int(len(true)),
            "rmse": float(np.sqrt(np.mean(err * err))) if len(true) else None,
            "mae": float(np.mean(np.abs(err))) if len(true) else None,
            "raw_r": safe_pearsonr(true, pred),
            "within_subject_centered_r": safe_pearsonr(centered_true, centered_pred),
            "test_label_std": float(np.std(true)) if len(true) else None,
            "prediction_std": float(np.std(pred)) if len(pred) else None,
        }
    summary = _summarize_metrics(per_label)
    return {"per_label": per_label, "summary": summary, "groups": _group_summaries(per_label)}


def _summarize_metrics(per_label: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_rmse": _mean_metric(per_label, "rmse"),
        "mean_mae": _mean_metric(per_label, "mae"),
        "mean_raw_r": _mean_metric(per_label, "raw_r"),
        "mean_centered_r": _mean_metric(per_label, "within_subject_centered_r"),
    }


def _group_summaries(per_label: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = {
        "positive_activation": [name for name in LABEL_NAMES if name in POSITIVE_LABELS],
        "negative_distress": [name for name in LABEL_NAMES if name in NEGATIVE_LABELS],
        "fatigue": ["fatigue"],
    }
    return {
        group: {
            "labels": labels,
            "mean_rmse": _mean_metric({name: per_label[name] for name in labels}, "rmse"),
            "mean_raw_r": _mean_metric({name: per_label[name] for name in labels}, "raw_r"),
            "mean_centered_r": _mean_metric({name: per_label[name] for name in labels}, "within_subject_centered_r"),
        }
        for group, labels in groups.items()
    }


def _mean_metric(per_label: dict[str, dict[str, Any]], metric: str) -> float | None:
    values = [row.get(metric) for row in per_label.values()]
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _audit_labels(targets: np.ndarray, rows: list[dict[str, Any]], event_window_id: np.ndarray) -> dict[str, Any]:
    per_label = {}
    low_signal = []
    for idx, label in enumerate(LABEL_NAMES):
        values = targets[:, idx].astype(np.float64)
        counts = {str(score): int(np.sum(np.rint(values) == score)) for score in range(1, 6)}
        std = float(np.std(values))
        min_tail = min(counts.values())
        flags = []
        if std < 0.25:
            flags.append("low_variance")
        if min_tail < 50:
            flags.append("sparse_extreme_tail")
        if flags:
            low_signal.append(label)
        per_label[label] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "std": std,
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "rounded_label_counts": counts,
            "flags": flags,
        }
    row_count_ok = len(rows) == 28819
    window_ok = set(event_window_id.tolist()) == set(range(23))
    finite_ok = bool(np.all(np.isfinite(targets)))
    complete_ok = targets.shape == (len(rows), len(LABEL_NAMES))
    gate_passed = row_count_ok and window_ok and finite_ok and complete_ok and not any(
        "low_variance" in row["flags"] for row in per_label.values()
    )
    return {
        "row_count": int(len(rows)),
        "event_count": int(len({str(row.get("event_id", "")) for row in rows})),
        "event_window_id_min": int(np.min(event_window_id)),
        "event_window_id_max": int(np.max(event_window_id)),
        "event_window_id_count": int(len(set(event_window_id.tolist()))),
        "label_count": int(targets.shape[1]),
        "finite_ok": finite_ok,
        "canonical_28819_ok": row_count_ok,
        "event_window_0_22_ok": window_ok,
        "complete_ok": complete_ok,
        "low_signal_labels": low_signal,
        "gate_passed": gate_passed,
        "gate_note": "sparse_extreme_tail labels are warnings; low_variance would stop the experiment",
        "per_label": per_label,
    }


def _load_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        labels = row.get("labels")
        if isinstance(labels, dict):
            values.append([float(labels[name]) for name in LABEL_NAMES])
        elif isinstance(labels, list):
            values.append([float(labels[idx]) for idx in range(len(LABEL_NAMES))])
        else:
            values.append([float(row[name]) for name in LABEL_NAMES])
    targets = np.asarray(values, dtype=np.float32)
    if targets.shape[1] != len(LABEL_NAMES):
        raise ValueError(f"expected {len(LABEL_NAMES)} labels, got {targets.shape}")
    return targets


def _load_all_branches(
    embedding_root: Path,
    sample_id: np.ndarray,
    names: tuple[str, ...],
    cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in names:
        if name in cache:
            result[name] = cache[name]
            continue
        branch = BRANCHES[name]
        path = embedding_root / branch.filename
        with np.load(path, allow_pickle=True) as loaded:
            loaded_ids = loaded["sample_id"].astype(str)
            if not np.array_equal(loaded_ids, sample_id):
                raise ValueError(f"{name} sample_id order does not match canonical index")
            emb_key = branch.emb_key if branch.emb_key in loaded.files else "face_emb"
            embedding = loaded[emb_key].astype(np.float32)
            if branch.mask_key in loaded.files:
                mask = loaded[branch.mask_key].astype(bool)
            else:
                mask = loaded["modality_mask"][:, branch.modality_index].astype(bool)
            if embedding.shape != (len(sample_id), 256) or mask.shape != (len(sample_id),):
                raise ValueError(f"invalid shape for {name}: {embedding.shape}, {mask.shape}")
            if not np.all(np.isfinite(embedding)):
                raise ValueError(f"{name} contains non-finite embedding values")
        result[name] = {"embedding": embedding, "mask": mask, "path": str(path), "mask_sum": int(mask.sum())}
        cache[name] = result[name]
    return result


def _build_tokens(data: dict[str, dict[str, Any]], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tokens = np.stack([data[name]["embedding"] for name in names], axis=1).astype(np.float32)
    masks = np.stack([data[name]["mask"] for name in names], axis=1).astype(bool)
    report = {
        name: {"path": data[name]["path"], "mask_sum": data[name]["mask_sum"], "modality": BRANCHES[name].modality}
        for name in names
    }
    return tokens, masks, report


def _fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, None], tokens[indices], np.nan)
    mean = np.nanmean(available, axis=(0, 1), keepdims=True)
    std = np.nanstd(available, axis=(0, 1), keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def _make_batches(n_rows: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n_rows)
    return [order[start : start + max(1, batch_size)] for start in range(0, len(order), max(1, batch_size))]


def _mask_coverage(mask: np.ndarray, branches: tuple[str, ...], split: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            branch: {
                "valid": int(mask[indices, col].sum()),
                "total": int(len(indices)),
                "coverage": float(mask[indices, col].mean()) if len(indices) else 0.0,
            }
            for col, branch in enumerate(branches)
        }
        for name, indices in split.items()
        if name in {"train", "val", "test"}
    }


def _load_split(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    split = {name: _load_indices(path / f"{name}.json", n_rows) for name in ("pretrain", "finetune", "val", "test")}
    split["train"] = np.asarray(split["pretrain"].tolist() + split["finetune"].tolist(), dtype=np.int64)
    _validate_no_overlap(path.name, split["train"], split["val"], split["test"])
    return split


def _load_indices(path: Path, n_rows: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= n_rows):
        raise ValueError(f"{path} contains out-of-range indices")
    return values


def _validate_no_overlap(protocol: str, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    pairs = (("train", train), ("val", val), ("test", test))
    for left_name, left in pairs:
        left_set = set(left.tolist())
        for right_name, right in pairs:
            if left_name >= right_name:
                continue
            overlap = left_set.intersection(right.tolist())
            if overlap:
                raise ValueError(f"{protocol} {left_name}/{right_name} index overlap: {len(overlap)}")


def _output_header(args: argparse.Namespace, label_audit: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "fixed_tokens_multitask_fusion_v1",
        "root": str(args.root),
        "splits_root": str(args.splits_root),
        "embeddings_root": str(args.embeddings_root),
        "label_names": list(LABEL_NAMES),
        "label_audit": label_audit,
        "runtime": {
            "protocols": _split_csv(args.protocols),
            "experiments": _split_csv(args.experiments),
            "seeds": [int(value) for value in _split_csv(args.seeds)],
            "epochs": int(args.epochs),
            "hidden_dim": int(args.hidden_dim),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "patience": int(args.patience),
            "device": args.device,
            "train_rule": "pretrain + finetune",
            "target_normalization": "per_label_train_only",
            "token_normalization": "train_only_available_tokens",
            "supervision_boundary": "fixed/frozen embeddings; train fusion encoder and 11-label head",
        },
        "run_count": len(results),
        "results": results,
        "route_winners": _route_winners(results),
    }


def _route_winners(results: list[dict[str, Any]]) -> dict[str, Any]:
    winners: dict[str, Any] = {}
    for protocol in sorted({row["protocol"] for row in results}):
        rows = [row for row in results if row["protocol"] == protocol]
        per_label = {}
        for label in LABEL_NAMES:
            label_rows = [
                {
                    "experiment": row["experiment"],
                    "seed": row["seed"],
                    **row["test"]["per_label"][label],
                }
                for row in rows
            ]
            per_label[label] = {
                "best_rmse": min(label_rows, key=lambda row: row["rmse"] if row["rmse"] is not None else float("inf")),
                "best_raw_r": max(label_rows, key=lambda row: row["raw_r"] if row["raw_r"] is not None else -float("inf")),
                "best_centered_r": max(
                    label_rows,
                    key=lambda row: row["within_subject_centered_r"]
                    if row["within_subject_centered_r"] is not None
                    else -float("inf"),
                ),
            }
        winners[protocol] = per_label
    return winners


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fixed Tokens Multi-label Fusion",
        "",
        f"stage: `{output['stage']}`",
        f"run_count: `{output['run_count']}`",
        f"label_audit_gate_passed: `{output['label_audit']['gate_passed']}`",
        "",
        "## Label Audit",
        "",
        "| label | n | mean | std | min | median | max | flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label in LABEL_NAMES:
        row = output["label_audit"]["per_label"][label]
        lines.append(
            f"| {label} | {row['n']} | {_fmt(row['mean'])} | {_fmt(row['std'])} | {_fmt(row['min'])} | "
            f"{_fmt(row['median'])} | {_fmt(row['max'])} | {', '.join(row['flags']) or 'OK'} |"
        )
    lines.extend(
        [
            "",
            "## Test Summary",
            "",
            "| protocol | experiment | seed | mean RMSE | mean raw r | mean centered r | fatigue RMSE | fatigue raw r | fatigue centered r | best epoch |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["results"]:
        summary = row["test"]["summary"]
        fatigue = row["test"]["per_label"]["fatigue"]
        lines.append(
            f"| {row['protocol']} | {row['experiment']} | {row['seed']} | {_fmt(summary['mean_rmse'])} | "
            f"{_fmt(summary['mean_raw_r'])} | {_fmt(summary['mean_centered_r'])} | {_fmt(fatigue['rmse'])} | "
            f"{_fmt(fatigue['raw_r'])} | {_fmt(fatigue['within_subject_centered_r'])} | {row['train_audit']['best_epoch']} |"
        )
    lines.extend(["", "## Per-label Test Metrics", ""])
    for row in output["results"]:
        lines.extend(
            [
                f"### {row['protocol']} / {row['experiment']} / seed {row['seed']}",
                "",
                "| label | RMSE | MAE | raw r | centered r | test label std | prediction std |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label in LABEL_NAMES:
            metric = row["test"]["per_label"][label]
            lines.append(
                f"| {label} | {_fmt(metric['rmse'])} | {_fmt(metric['mae'])} | {_fmt(metric['raw_r'])} | "
                f"{_fmt(metric['within_subject_centered_r'])} | {_fmt(metric['test_label_std'])} | {_fmt(metric['prediction_std'])} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
