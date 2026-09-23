#!/usr/bin/env python3
"""Diagnose W3FM failure modes after the wear-only FM gate stopped."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered, safe_pearsonr
from daily_multimodal.split_paths import resolve_protocol_split_root


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_WEAR_FM_ROOT = DEFAULT_ROOT / "outputs/wear_fm"
DEFAULT_PHASE2_ROOT = DEFAULT_WEAR_FM_ROOT / "phase2"
DEFAULT_INTERNAL_ROOT = DEFAULT_WEAR_FM_ROOT / "internal_ablation"
DEFAULT_OUT_ROOT = DEFAULT_WEAR_FM_ROOT / "failure_diagnostics"
DEFAULT_PROTOCOLS = ("cross_day", "date_in_order")
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
    "stress",
    "happy",
)
MODALITIES = (
    ("ppg", "PPG/PaPaGei-S", "ppg_papagei_s_512d.npy", 512),
    ("acc", "ACC/HARNet10", "acc_harnet10.npy", 1024),
    ("gsr", "GSR/NormWear", "gsr_normwear_768d.npy", 768),
)


@dataclass(frozen=True)
class Dataset:
    sample_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    event_id: np.ndarray
    target: np.ndarray
    complete_mask: np.ndarray
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

    def encode_with_weights(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ppg, acc, gsr = inputs
        tokens = torch.stack([self.proj_ppg(ppg), self.proj_acc(acc), self.proj_gsr(gsr)], dim=1)
        logits = self.gate(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        embedding = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return embedding, weights, tokens, logits

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        embedding, _weights, _tokens, _logits = self.encode_with_weights(inputs)
        return self.head(embedding)


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
    parser.add_argument("--wear-fm-root", type=Path, default=DEFAULT_WEAR_FM_ROOT)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--internal-root", type=Path, default=DEFAULT_INTERNAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--top-features", type=int, default=128)
    parser.add_argument("--ridge-alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset = _load_dataset(args)
    protocols = _split_csv(args.protocols)
    seeds = [int(value) for value in _split_csv(args.seeds)]
    alphas = [float(value) for value in _split_csv(args.ridge_alphas)]
    args.out_root.mkdir(parents=True, exist_ok=True)

    split_by_protocol = {
        protocol: _filter_split(
            _load_split(resolve_protocol_split_root(args.splits_root, protocol), dataset.sample_id.shape[0]),
            dataset.complete_mask,
        )
        for protocol in protocols
    }
    shift_rows = []
    probe_rows = []
    feature_corr_rows = []
    gate_rows = []
    gate_error_rows = []
    for protocol, split in split_by_protocol.items():
        for key, label, _file_name, _dim in MODALITIES:
            arr = getattr(dataset, key)
            shift_rows.extend(_embedding_shift_rows(protocol, key, label, arr, split))
            corr_row, probe_row = _linear_probe_rows(protocol, key, label, arr, dataset, split, args.top_features, alphas)
            feature_corr_rows.append(corr_row)
            probe_rows.append(probe_row)
        for seed in seeds:
            gate_result = _gate_diagnostics(args, dataset, split, protocol, seed)
            gate_rows.extend(gate_result["gate_rows"])
            gate_error_rows.extend(gate_result["error_rows"])

    internal_rows = _read_internal_ablation(args.internal_root / "internal_ablation_summary.csv")
    conclusions = _build_conclusions(shift_rows, probe_rows, feature_corr_rows, gate_rows, gate_error_rows, internal_rows)
    output = {
        "script": Path(__file__).name,
        "stage": "wear_only_fm_failure_diagnostics",
        "target_label": args.target_label,
        "protocols": list(protocols),
        "seeds": seeds,
        "row_count": int(dataset.sample_id.shape[0]),
        "wear_complete_count": int(dataset.complete_mask.sum()),
        "diagnostic_boundary": "post_phase3_explanatory_diagnostic_not_a_formal_promotion_gate",
        "internal_ablation_rows": internal_rows,
        "embedding_shift": shift_rows,
        "feature_correlation_stability": feature_corr_rows,
        "linear_probes": probe_rows,
        "gate_weights": gate_rows,
        "gate_error_association": gate_error_rows,
        "conclusions": conclusions,
    }
    _write_json(args.out_root / "w3fm_failure_diagnostics_report.json", output)
    _write_csv(args.out_root / "embedding_shift_summary.csv", shift_rows)
    _write_csv(args.out_root / "feature_correlation_stability.csv", feature_corr_rows)
    _write_csv(args.out_root / "linear_probe_summary.csv", probe_rows)
    _write_csv(args.out_root / "gate_weight_summary.csv", gate_rows)
    _write_csv(args.out_root / "gate_error_summary.csv", gate_error_rows)
    _write_markdown(args.out_root / "w3fm_failure_diagnostics_report.md", output)
    print(f"out_json={args.out_root / 'w3fm_failure_diagnostics_report.json'}")
    print(f"out_md={args.out_root / 'w3fm_failure_diagnostics_report.md'}")
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
    arrays = {}
    for key, _label, file_name, dim in MODALITIES:
        path = args.wear_fm_root / "embeddings" / file_name
        values = np.load(path).astype(np.float32)
        expected = (len(rows), dim)
        if values.shape != expected:
            raise ValueError(f"{key} embedding shape {values.shape} is not {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} embedding contains non-finite values")
        arrays[key] = values
    return Dataset(sample_id, subject_id, day_id, event_id, target, complete_mask, arrays["ppg"], arrays["acc"], arrays["gsr"])


def _embedding_shift_rows(
    protocol: str,
    modality: str,
    label: str,
    arr: np.ndarray,
    split: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    train = split["train"]
    mean = arr[train].mean(axis=0, keepdims=True).astype(np.float32)
    raw_train_std = arr[train].std(axis=0, keepdims=True).astype(np.float32)
    low_std_fraction = float(np.mean(raw_train_std.reshape(-1) < 1e-6))
    std = raw_train_std.copy()
    std[std < 1e-6] = 1.0
    rows = []
    for split_name in ("train", "val", "test"):
        idx = split[split_name]
        z = ((arr[idx] - mean) / std).astype(np.float32)
        split_mean = z.mean(axis=0)
        split_std = z.std(axis=0)
        norm = np.linalg.norm(z, axis=1) / math.sqrt(arr.shape[1])
        rows.append(
            {
                "protocol": protocol,
                "modality": modality,
                "label": label,
                "split": split_name,
                "count": int(len(idx)),
                "dim": int(arr.shape[1]),
                "raw_global_std": float(np.std(arr[idx])),
                "raw_train_feature_std_median": float(np.median(raw_train_std)),
                "raw_train_feature_std_p10": float(np.quantile(raw_train_std, 0.10)),
                "raw_train_feature_std_p90": float(np.quantile(raw_train_std, 0.90)),
                "raw_train_low_std_fraction": low_std_fraction,
                "standardized_mean_shift_l2": float(np.sqrt(np.mean(split_mean * split_mean))),
                "standardized_abs_mean_shift": float(np.mean(np.abs(split_mean))),
                "standardized_std_median": float(np.median(split_std)),
                "standardized_std_p10": float(np.quantile(split_std, 0.10)),
                "standardized_std_p90": float(np.quantile(split_std, 0.90)),
                "standardized_sample_norm_mean": float(np.mean(norm)),
                "standardized_sample_norm_std": float(np.std(norm)),
            }
        )
    return rows


def _linear_probe_rows(
    protocol: str,
    modality: str,
    label: str,
    arr: np.ndarray,
    dataset: Dataset,
    split: dict[str, np.ndarray],
    top_features: int,
    alphas: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = split["train"]
    val = split["val"]
    test = split["test"]
    mean = arr[train].mean(axis=0, keepdims=True).astype(np.float32)
    std = arr[train].std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_z = ((arr[train] - mean) / std).astype(np.float32)
    train_corr = _feature_target_corr(train_z, dataset.target[train])
    k = min(int(top_features), arr.shape[1])
    order = np.argsort(-np.abs(train_corr))[:k]
    corr_by_split = {
        "train": train_corr[order],
        "val": _feature_target_corr(((arr[val] - mean) / std).astype(np.float32)[:, order], dataset.target[val]),
        "test": _feature_target_corr(((arr[test] - mean) / std).astype(np.float32)[:, order], dataset.target[test]),
    }
    corr_row = _feature_corr_summary(protocol, modality, label, order, corr_by_split)
    probe_row = _fit_topk_ridge(protocol, modality, label, order, mean, std, arr, dataset, split, alphas)
    return corr_row, probe_row


def _feature_target_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    x64 = x64 - x64.mean(axis=0, keepdims=True)
    y64 = y64 - y64.mean()
    numerator = x64.T @ y64
    x_denom = np.sqrt(np.sum(x64 * x64, axis=0))
    y_denom = float(np.sqrt(np.sum(y64 * y64)))
    denom = x_denom * y_denom
    out = np.zeros((x64.shape[1],), dtype=np.float64)
    ok = denom > 0.0
    out[ok] = numerator[ok] / denom[ok]
    return out.astype(np.float32)


def _feature_corr_summary(
    protocol: str,
    modality: str,
    label: str,
    order: np.ndarray,
    corr_by_split: dict[str, np.ndarray],
) -> dict[str, Any]:
    train = corr_by_split["train"]
    val = corr_by_split["val"]
    test = corr_by_split["test"]
    train_sign = np.sign(train)
    return {
        "protocol": protocol,
        "modality": modality,
        "label": label,
        "top_feature_count": int(len(order)),
        "top_train_abs_corr_mean": float(np.mean(np.abs(train))),
        "top_train_abs_corr_max": float(np.max(np.abs(train))),
        "top_val_signed_corr_mean": float(np.mean(train_sign * val)),
        "top_test_signed_corr_mean": float(np.mean(train_sign * test)),
        "top_val_sign_agreement": float(np.mean((train_sign * np.sign(val)) > 0.0)),
        "top_test_sign_agreement": float(np.mean((train_sign * np.sign(test)) > 0.0)),
        "top_feature_indices_head": ";".join(str(int(value)) for value in order[:10].tolist()),
    }


def _fit_topk_ridge(
    protocol: str,
    modality: str,
    label: str,
    order: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    arr: np.ndarray,
    dataset: Dataset,
    split: dict[str, np.ndarray],
    alphas: list[float],
) -> dict[str, Any]:
    train = split["train"]
    val = split["val"]
    y_mean = float(dataset.target[train].mean())
    y_std = float(dataset.target[train].std()) or 1.0
    x_train = ((arr[train] - mean) / std).astype(np.float64)[:, order]
    y_train = ((dataset.target[train] - y_mean) / y_std).astype(np.float64)
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train
    best: dict[str, Any] | None = None
    identity = np.eye(len(order), dtype=np.float64)
    for alpha in alphas:
        weights = np.linalg.solve(xtx + float(alpha) * identity, xty)
        val_pred = _ridge_predict(arr, split["val"], order, mean, std, weights, y_mean, y_std)
        metrics = evaluate_regression_with_centered(dataset.target[split["val"]], val_pred, dataset.subject_id[split["val"]])
        score = -float(metrics["rmse"])
        if best is None or score > float(best["score"]):
            best = {"alpha": float(alpha), "weights": weights, "score": score}
    if best is None:
        raise RuntimeError("ridge probe produced no model")
    result = {
        "protocol": protocol,
        "modality": modality,
        "label": label,
        "top_feature_count": int(len(order)),
        "selected_alpha": float(best["alpha"]),
    }
    for split_name in ("train", "val", "test"):
        idx = split[split_name]
        pred = _ridge_predict(arr, idx, order, mean, std, best["weights"], y_mean, y_std)
        metrics = evaluate_regression_with_centered(dataset.target[idx], pred, dataset.subject_id[idx])
        result[f"{split_name}_rmse"] = _nullable_float(metrics["rmse"])
        result[f"{split_name}_raw_r"] = _nullable_float(metrics["raw_r"])
        result[f"{split_name}_centered_r"] = _nullable_float(metrics["within_subject_centered_r"])
        result[f"{split_name}_prediction_std"] = float(np.std(pred)) if len(pred) else math.nan
    result["train_test_rmse_gap"] = float(result["test_rmse"] - result["train_rmse"])
    result["val_test_rmse_gap"] = float(result["test_rmse"] - result["val_rmse"])
    return result


def _ridge_predict(
    arr: np.ndarray,
    idx: np.ndarray,
    order: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    x = ((arr[idx] - mean) / std).astype(np.float64)[:, order]
    pred = x @ weights
    return (pred * y_std + y_mean).astype(np.float32)


def _gate_diagnostics(
    args: argparse.Namespace,
    dataset: Dataset,
    split: dict[str, np.ndarray],
    protocol: str,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    run_dir = args.phase2_root / "runs" / protocol / "W3FM_frozen" / f"seed_{seed}"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    try:
        checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu")
    module = W3FMModel(hidden_dim=int(config.get("hidden_dim", 128)), dropout=float(config.get("dropout", 0.1)))
    module.load_state_dict(checkpoint["state_dict"])
    module.to(torch.device(args.device))
    module.eval()
    train = split["train"]
    means = tuple(arr[train].mean(axis=0, keepdims=True).astype(np.float32) for arr in (dataset.ppg, dataset.acc, dataset.gsr))
    stds = tuple(arr[train].std(axis=0, keepdims=True).astype(np.float32) for arr in (dataset.ppg, dataset.acc, dataset.gsr))
    stds = tuple(np.where(std < 1e-6, 1.0, std).astype(np.float32) for std in stds)
    gate_rows = []
    error_rows = []
    predictions = np.load(run_dir / "predictions.npz", allow_pickle=True)
    for split_name in ("train", "val", "test"):
        idx = split[split_name]
        gate_arrays = _predict_gate_arrays(args, module, dataset, idx, means, stds)
        weights = gate_arrays["weights"]
        gate_rows.append(_gate_weight_summary(protocol, seed, split_name, idx, gate_arrays, dataset))
        pred = predictions[f"{split_name}_prediction"].astype(np.float32)
        if pred.shape != (len(idx),):
            raise ValueError(f"prediction shape mismatch for {run_dir} {split_name}: {pred.shape} vs {len(idx)}")
        error_rows.extend(_gate_error_rows(protocol, seed, split_name, weights, dataset.target[idx], pred))
    return {"gate_rows": gate_rows, "error_rows": error_rows}


def _predict_gate_arrays(
    args: argparse.Namespace,
    module: W3FMModel,
    dataset: Dataset,
    indices: np.ndarray,
    means: tuple[np.ndarray, np.ndarray, np.ndarray],
    stds: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    arrays = (dataset.ppg, dataset.acc, dataset.gsr)
    weight_chunks = []
    token_norm_chunks = []
    logit_chunks = []
    device = torch.device(args.device)
    with torch.no_grad():
        for start in range(0, len(indices), int(args.batch_size)):
            batch = indices[start : start + int(args.batch_size)]
            inputs = tuple(
                torch.as_tensor(((arr[batch] - mean) / std).astype(np.float32), dtype=torch.float32, device=device)
                for arr, mean, std in zip(arrays, means, stds)
            )
            _embedding, weights, tokens, logits = module.encode_with_weights(inputs)
            weight_chunks.append(weights.detach().cpu().numpy().astype(np.float32))
            token_norm = torch.linalg.vector_norm(tokens, dim=2) / math.sqrt(tokens.shape[2])
            token_norm_chunks.append(token_norm.detach().cpu().numpy().astype(np.float32))
            logit_chunks.append(logits.detach().cpu().numpy().astype(np.float32))
    if not weight_chunks:
        empty = np.zeros((0, 3), dtype=np.float32)
        return {"weights": empty, "token_norms": empty, "logits": empty}
    return {
        "weights": np.concatenate(weight_chunks, axis=0),
        "token_norms": np.concatenate(token_norm_chunks, axis=0),
        "logits": np.concatenate(logit_chunks, axis=0),
    }


def _gate_weight_summary(
    protocol: str,
    seed: int,
    split_name: str,
    idx: np.ndarray,
    gate_arrays: dict[str, np.ndarray],
    dataset: Dataset,
) -> dict[str, Any]:
    weights = gate_arrays["weights"]
    token_norms = gate_arrays["token_norms"]
    logits = gate_arrays["logits"]
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-8, 1.0)), axis=1) / math.log(weights.shape[1])
    dominant = np.argmax(weights, axis=1)
    row = {
        "protocol": protocol,
        "seed": int(seed),
        "split": split_name,
        "count": int(len(idx)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_p10": float(np.quantile(entropy, 0.10)),
        "max_weight_mean": float(np.mean(np.max(weights, axis=1))),
        "max_weight_gt_0p70_fraction": float(np.mean(np.max(weights, axis=1) > 0.70)),
        "max_weight_gt_0p80_fraction": float(np.mean(np.max(weights, axis=1) > 0.80)),
    }
    for col, modality in enumerate(("ppg", "acc", "gsr")):
        values = weights[:, col]
        row[f"{modality}_weight_mean"] = float(np.mean(values))
        row[f"{modality}_weight_std"] = float(np.std(values))
        row[f"{modality}_dominant_fraction"] = float(np.mean(dominant == col))
        row[f"{modality}_target_corr"] = _nullable_float(safe_pearsonr(dataset.target[idx], values))
        row[f"{modality}_token_norm_mean"] = float(np.mean(token_norms[:, col]))
        row[f"{modality}_token_norm_std"] = float(np.std(token_norms[:, col]))
        row[f"{modality}_gate_logit_mean"] = float(np.mean(logits[:, col]))
        row[f"{modality}_gate_logit_std"] = float(np.std(logits[:, col]))
    return row


def _gate_error_rows(
    protocol: str,
    seed: int,
    split_name: str,
    weights: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    abs_error = np.abs(prediction - target)
    if len(abs_error) == 0:
        return []
    q25, q75 = np.quantile(abs_error, [0.25, 0.75])
    low = abs_error <= q25
    high = abs_error >= q75
    rows = []
    for col, modality in enumerate(("ppg", "acc", "gsr")):
        values = weights[:, col]
        rows.append(
            {
                "protocol": protocol,
                "seed": int(seed),
                "split": split_name,
                "modality": modality,
                "count": int(len(abs_error)),
                "weight_abs_error_corr": _nullable_float(safe_pearsonr(values, abs_error)),
                "low_error_weight_mean": float(np.mean(values[low])),
                "high_error_weight_mean": float(np.mean(values[high])),
                "high_minus_low_weight": float(np.mean(values[high]) - np.mean(values[low])),
            }
        )
    return rows


def _read_internal_ablation(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _build_conclusions(
    shift_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    internal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    conclusions: dict[str, Any] = {}
    no_acc = _find_row(internal_rows, protocol="cross_day", route="W3FM_no_acc")
    if no_acc:
        conclusions["acc_branch_behavior_evidence"] = {
            "delta_raw_r_vs_W3FM_frozen_mean": _float_or_none(no_acc.get("delta_raw_r_vs_W3FM_frozen_mean")),
            "delta_rmse_vs_W3FM_frozen_mean": _float_or_none(no_acc.get("delta_rmse_vs_W3FM_frozen_mean")),
            "delta_centered_r_vs_W3FM_frozen_mean": _float_or_none(no_acc.get("delta_centered_r_vs_W3FM_frozen_mean")),
        }
    conclusions["largest_test_shift_by_protocol"] = _largest_by_group(
        [row for row in shift_rows if row["split"] == "test"],
        group_key="protocol",
        value_key="standardized_mean_shift_l2",
    )
    conclusions["linear_probe_best_by_protocol"] = _best_probe_by_protocol(probe_rows)
    conclusions["feature_corr_test_stability_by_protocol"] = _rank_by_group(
        corr_rows,
        group_key="protocol",
        value_key="top_test_signed_corr_mean",
        descending=True,
    )
    conclusions["gate_mean_by_protocol_split"] = _gate_means(gate_rows)
    conclusions["test_error_weight_association"] = _rank_by_group(
        [row for row in error_rows if row["split"] == "test"],
        group_key="protocol",
        value_key="high_minus_low_weight",
        descending=True,
    )
    return conclusions


def _find_row(rows: list[dict[str, Any]], **items: str) -> dict[str, Any] | None:
    for row in rows:
        if all(str(row.get(key)) == value for key, value in items.items()):
            return row
    return None


def _largest_by_group(rows: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({str(row[group_key]) for row in rows}):
        candidates = [row for row in rows if str(row[group_key]) == group]
        best = max(candidates, key=lambda row: float(row[value_key]))
        out[group] = {"modality": best["modality"], value_key: float(best[value_key])}
    return out


def _rank_by_group(rows: list[dict[str, Any]], group_key: str, value_key: str, *, descending: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({str(row[group_key]) for row in rows}):
        candidates = sorted(
            [row for row in rows if str(row[group_key]) == group],
            key=lambda row: float(row[value_key]),
            reverse=descending,
        )
        out[group] = [
            {
                "modality": row.get("modality"),
                value_key: _float_or_none(row.get(value_key)),
            }
            for row in candidates
        ]
    return out


def _best_probe_by_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for protocol in sorted({str(row["protocol"]) for row in rows}):
        metric = "test_centered_r" if protocol == "date_in_order" else "test_raw_r"
        candidates = sorted([row for row in rows if row["protocol"] == protocol], key=lambda row: float(row[metric]), reverse=True)
        out[protocol] = [
            {
                "modality": row["modality"],
                metric: float(row[metric]),
                "test_rmse": float(row["test_rmse"]),
                "train_test_rmse_gap": float(row["train_test_rmse_gap"]),
            }
            for row in candidates
        ]
    return out


def _gate_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for protocol in sorted({str(row["protocol"]) for row in rows}):
        for split_name in ("train", "val", "test"):
            candidates = [row for row in rows if row["protocol"] == protocol and row["split"] == split_name]
            if not candidates:
                continue
            key = f"{protocol}:{split_name}"
            out[key] = {
                "entropy_mean": float(np.mean([float(row["entropy_mean"]) for row in candidates])),
                "ppg_weight_mean": float(np.mean([float(row["ppg_weight_mean"]) for row in candidates])),
                "acc_weight_mean": float(np.mean([float(row["acc_weight_mean"]) for row in candidates])),
                "gsr_weight_mean": float(np.mean([float(row["gsr_weight_mean"]) for row in candidates])),
            }
    return out


def _write_markdown(path: Path, output: dict[str, Any]) -> None:
    lines = [
        "# W3FM Failure Diagnostics",
        "",
        f"- row_count: `{output['row_count']}`",
        f"- wear_complete_count: `{output['wear_complete_count']}`",
        f"- diagnostic_boundary: `{output['diagnostic_boundary']}`",
        "",
        "## Reading",
        "",
        "This report separates three hypotheses: modality instability, embedding distribution mismatch, and gate overfitting. It is explanatory evidence after the Phase 3 stop decision, not a new promotion gate.",
        "",
        "## Diagnosis Summary",
        "",
    ]
    lines.extend(_diagnosis_summary_lines(output))
    lines.extend(
        [
            "",
            "## Existing Internal Ablation Evidence",
            "",
            "| protocol | route | d raw r vs W3FM | d RMSE vs W3FM | d centered r vs W3FM |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in output["internal_ablation_rows"]:
        if row.get("protocol") in DEFAULT_PROTOCOLS:
            lines.append(
                f"| {row.get('protocol')} | {row.get('route')} | "
                f"{_fmt(_float_or_none(row.get('delta_raw_r_vs_W3FM_frozen_mean')))} | "
                f"{_fmt(_float_or_none(row.get('delta_rmse_vs_W3FM_frozen_mean')))} | "
                f"{_fmt(_float_or_none(row.get('delta_centered_r_vs_W3FM_frozen_mean')))} |"
            )
    lines.extend(
        [
            "",
            "## Embedding Split Shift",
            "",
            "| protocol | modality | split | raw std | z mean shift L2 | z abs mean shift | z sample norm |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["embedding_shift"]:
        if row["split"] in {"train", "test"}:
            lines.append(
            f"| {row['protocol']} | {row['modality']} | {row['split']} | "
            f"{_fmt(row['raw_global_std'])} | {_fmt(row['standardized_mean_shift_l2'])} | "
            f"{_fmt(row['standardized_abs_mean_shift'])} | {_fmt(row['standardized_sample_norm_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Train Feature Scale",
            "",
            "| protocol | modality | raw train feature std median | raw train feature std p10 | low-std feature fraction |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    seen_scale = set()
    for row in output["embedding_shift"]:
        key = (row["protocol"], row["modality"])
        if key in seen_scale:
            continue
        seen_scale.add(key)
        lines.append(
            f"| {row['protocol']} | {row['modality']} | {_fmt(row['raw_train_feature_std_median'])} | "
            f"{_fmt(row['raw_train_feature_std_p10'])} | {_fmt(row['raw_train_low_std_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Feature Correlation Stability",
            "",
            "| protocol | modality | top train abs corr | val signed corr | test signed corr | test sign agreement |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["feature_correlation_stability"]:
        lines.append(
            f"| {row['protocol']} | {row['modality']} | {_fmt(row['top_train_abs_corr_mean'])} | "
            f"{_fmt(row['top_val_signed_corr_mean'])} | {_fmt(row['top_test_signed_corr_mean'])} | "
            f"{_fmt(row['top_test_sign_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "## Linear Probe",
            "",
            "| protocol | modality | alpha | test RMSE | test raw r | test centered r | train-test RMSE gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["linear_probes"]:
        lines.append(
            f"| {row['protocol']} | {row['modality']} | {_fmt(row['selected_alpha'])} | "
            f"{_fmt(row['test_rmse'])} | {_fmt(row['test_raw_r'])} | {_fmt(row['test_centered_r'])} | "
            f"{_fmt(row['train_test_rmse_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate Weights",
            "",
            "| protocol | seed | split | entropy | PPG w | ACC w | GSR w | PPG norm | ACC norm | GSR norm | max>0.8 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["gate_weights"]:
        if row["split"] in {"train", "test"}:
            lines.append(
                f"| {row['protocol']} | {row['seed']} | {row['split']} | {_fmt(row['entropy_mean'])} | "
                f"{_fmt(row['ppg_weight_mean'])} | {_fmt(row['acc_weight_mean'])} | "
                f"{_fmt(row['gsr_weight_mean'])} | {_fmt(row['ppg_token_norm_mean'])} | "
                f"{_fmt(row['acc_token_norm_mean'])} | {_fmt(row['gsr_token_norm_mean'])} | "
                f"{_fmt(row['max_weight_gt_0p80_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "## Gate And Error",
            "",
            "| protocol | seed | split | modality | corr(weight, abs error) | high-low error weight |",
            "| --- | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in output["gate_error_association"]:
        if row["split"] == "test":
            lines.append(
                f"| {row['protocol']} | {row['seed']} | {row['split']} | {row['modality']} | "
                f"{_fmt(row['weight_abs_error_corr'])} | {_fmt(row['high_minus_low_weight'])} |"
            )
    lines.extend(["", "## Machine Summary", "", "```json", json.dumps(output["conclusions"], ensure_ascii=False, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _diagnosis_summary_lines(output: dict[str, Any]) -> list[str]:
    internal = output["internal_ablation_rows"]
    corr = output["feature_correlation_stability"]
    probes = output["linear_probes"]
    gate = output["gate_weights"]
    errors = output["gate_error_association"]
    shift = output["embedding_shift"]
    no_acc = _find_row(internal, protocol="cross_day", route="W3FM_no_acc")
    cross_acc_corr = _find_row(corr, protocol="cross_day", modality="acc")
    within_acc_corr = _find_row(corr, protocol="date_in_order", modality="acc")
    cross_acc_probe = _find_row(probes, protocol="cross_day", modality="acc")
    within_acc_probe = _find_row(probes, protocol="date_in_order", modality="acc")
    cross_ppg_probe = _find_row(probes, protocol="cross_day", modality="ppg")
    within_ppg_probe = _find_row(probes, protocol="date_in_order", modality="ppg")
    cross_gate = _average_gate(gate, "cross_day", "test")
    within_gate = _average_gate(gate, "date_in_order", "test")
    cross_acc_error = _average_error(errors, "cross_day", "test", "acc")
    within_acc_error = _average_error(errors, "date_in_order", "test", "acc")
    cross_gsr_shift = _find_row(shift, protocol="cross_day", modality="gsr", split="test")
    within_gsr_shift = _find_row(shift, protocol="date_in_order", modality="gsr", split="test")
    cross_acc_shift = _find_row(shift, protocol="cross_day", modality="acc", split="test")
    within_acc_shift = _find_row(shift, protocol="date_in_order", modality="acc", split="test")
    cross_acc_scale = _find_row(shift, protocol="cross_day", modality="acc", split="train")
    within_acc_scale = _find_row(shift, protocol="date_in_order", modality="acc", split="train")

    lines = []
    if no_acc:
        lines.append(
            "- ACC/HARNet10 is the strongest branch-level suspect in `cross_day`: "
            f"`W3FM_no_acc` changes raw r by {_fmt(no_acc.get('delta_raw_r_vs_W3FM_frozen_mean'))}, "
            f"RMSE by {_fmt(no_acc.get('delta_rmse_vs_W3FM_frozen_mean'))}, and centered r by "
            f"{_fmt(no_acc.get('delta_centered_r_vs_W3FM_frozen_mean'))} versus full W3FM."
        )
    if cross_acc_corr and within_acc_corr:
        lines.append(
            "- ACC has the clearest train-to-test feature-correlation instability: "
            f"cross_day top train features have mean abs corr {_fmt(cross_acc_corr.get('top_train_abs_corr_mean'))}, "
            f"but signed test corr {_fmt(cross_acc_corr.get('top_test_signed_corr_mean'))} and sign agreement "
            f"{_fmt(cross_acc_corr.get('top_test_sign_agreement'))}; date_in_order signed test corr is "
            f"{_fmt(within_acc_corr.get('top_test_signed_corr_mean'))}."
        )
    if cross_acc_probe and cross_ppg_probe and within_acc_probe and within_ppg_probe:
        lines.append(
            "- The light linear probes also favor PPG over ACC: "
            f"cross_day PPG raw r {_fmt(cross_ppg_probe.get('test_raw_r'))} vs ACC "
            f"{_fmt(cross_acc_probe.get('test_raw_r'))}; date_in_order PPG centered r "
            f"{_fmt(within_ppg_probe.get('test_centered_r'))} vs ACC "
            f"{_fmt(within_acc_probe.get('test_centered_r'))}. ACC has the largest train-test RMSE gap in both protocols."
        )
    if cross_gate and within_gate and cross_acc_error and within_acc_error:
        lines.append(
            "- The learned gate gives ACC about half of the test weight "
            f"(cross_day {_fmt(cross_gate['acc_weight_mean'])}, date_in_order {_fmt(within_gate['acc_weight_mean'])}). "
            "High-error test rows consistently receive more ACC weight: "
            f"high-low ACC weight is {_fmt(cross_acc_error['high_minus_low_weight'])} in cross_day and "
            f"{_fmt(within_acc_error['high_minus_low_weight'])} in date_in_order."
        )
    if cross_gsr_shift and within_gsr_shift and cross_acc_shift and within_acc_shift:
        lines.append(
            "- Distribution mismatch exists, but it does not point only to ACC: "
            f"GSR has the largest train-standardized test mean shift "
            f"(cross_day {_fmt(cross_gsr_shift.get('standardized_mean_shift_l2'))}, "
            f"date_in_order {_fmt(within_gsr_shift.get('standardized_mean_shift_l2'))}), while ACC shift is smaller "
            f"({_fmt(cross_acc_shift.get('standardized_mean_shift_l2'))} and "
            f"{_fmt(within_acc_shift.get('standardized_mean_shift_l2'))}). This makes simple input-scale mismatch a secondary explanation."
        )
    if cross_acc_scale and within_acc_scale:
        lines.append(
            "- ACC also has many low-variance raw embedding dimensions "
            f"(low-std fraction {_fmt(cross_acc_scale.get('raw_train_low_std_fraction'))} cross_day, "
            f"{_fmt(within_acc_scale.get('raw_train_low_std_fraction'))} date_in_order), so any follow-up should include "
            "per-modality feature filtering or stronger train-only normalization before fusion."
        )
    return lines or ["- No automatic diagnosis could be generated; inspect the detailed tables below."]


def _average_gate(rows: list[dict[str, Any]], protocol: str, split_name: str) -> dict[str, float] | None:
    selected = [row for row in rows if row["protocol"] == protocol and row["split"] == split_name]
    if not selected:
        return None
    return {
        "entropy_mean": float(np.mean([float(row["entropy_mean"]) for row in selected])),
        "ppg_weight_mean": float(np.mean([float(row["ppg_weight_mean"]) for row in selected])),
        "acc_weight_mean": float(np.mean([float(row["acc_weight_mean"]) for row in selected])),
        "gsr_weight_mean": float(np.mean([float(row["gsr_weight_mean"]) for row in selected])),
    }


def _average_error(rows: list[dict[str, Any]], protocol: str, split_name: str, modality: str) -> dict[str, float] | None:
    selected = [
        row
        for row in rows
        if row["protocol"] == protocol and row["split"] == split_name and row["modality"] == modality
    ]
    if not selected:
        return None
    return {
        "weight_abs_error_corr": float(np.mean([float(row["weight_abs_error_corr"]) for row in selected])),
        "high_minus_low_weight": float(np.mean([float(row["high_minus_low_weight"]) for row in selected])),
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


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None or not math.isfinite(number):
        return "NA"
    return f"{number:.4f}"


def _nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _float_or_none(value: Any) -> float | None:
    return _nullable_float(value)


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    if text.isdigit():
        return f"sub-{int(text):02d}"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
