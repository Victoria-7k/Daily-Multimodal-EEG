from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


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

ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
EMB_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")


@dataclass(frozen=True)
class Branch:
    name: str
    modality: str
    path: Path
    emb_key: str
    mask_key: str
    modality_index: int


@dataclass(frozen=True)
class Experiment:
    name: str
    branches: tuple[str, ...]


BRANCHES = {
    "eeg": Branch(
        name="eeg",
        modality="eeg",
        path=EMB_ROOT / "eeg/eeg_statfft_eeg23win_embeddings.npz",
        emb_key="eeg_emb",
        mask_key="eeg_mask",
        modality_index=0,
    ),
    "wear_physio": Branch(
        name="wear_physio",
        modality="wear",
        path=EMB_ROOT / "wear/wear_physio_preprocessed_eeg23win_embeddings.npz",
        emb_key="wear_emb",
        mask_key="wear_mask",
        modality_index=1,
    ),
    "wear_deep": Branch(
        name="wear_deep",
        modality="wear",
        path=EMB_ROOT / "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz",
        emb_key="wear_emb",
        mask_key="wear_mask",
        modality_index=1,
    ),
    "video_B0": Branch(
        name="video_B0",
        modality="video",
        path=EMB_ROOT / "video/video_B0_2xroi_eeg23win_embeddings.npz",
        emb_key="video_emb",
        mask_key="video_mask",
        modality_index=2,
    ),
    "audio": Branch(
        name="audio",
        modality="audio",
        path=EMB_ROOT / "audio/audio_opensmile_eeg23win_embeddings.npz",
        emb_key="audio_emb",
        mask_key="audio_mask",
        modality_index=3,
    ),
}

EXPERIMENTS = [
    Experiment("B0_Wphysio_full", ("eeg", "wear_physio", "video_B0", "audio")),
    Experiment("B0_Wphysio_no_audio", ("eeg", "wear_physio", "video_B0")),
    Experiment("B0_Wphysio_no_video", ("eeg", "wear_physio", "audio")),
    Experiment("B0_Wphysio_bio_only", ("eeg", "wear_physio")),
    Experiment("B0_Wdeep_full", ("eeg", "wear_deep", "video_B0", "audio")),
    Experiment("B0_Wdeep_no_audio", ("eeg", "wear_deep", "video_B0")),
    Experiment("B0_Wdeep_no_video", ("eeg", "wear_deep", "audio")),
    Experiment("B0_Wdeep_bio_only", ("eeg", "wear_deep")),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the B0 EEG-aligned fusion matrix.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--splits-root", type=Path, default=SPLITS_ROOT)
    parser.add_argument("--protocol", choices=["cross_subject", "cross_day", "within_subject_day"], default="cross_subject")
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/b0_fusion_matrix")
    parser.add_argument("--intermediate-dir", type=Path, default=ROOT / "intermediate/b0_fusion_matrix")
    parser.add_argument("--summary-md", type=Path, default=ROOT / "reports/b0_fusion_matrix_summary.md")
    parser.add_argument("--summary-json", type=Path, default=ROOT / "reports/b0_fusion_matrix_summary.json")
    parser.add_argument("--experiments", help="Comma-separated experiment names. Defaults to the full 8-run B0 matrix.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=240729)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    _seed_everything(int(args.seed))
    out_dir = args.out_dir
    intermediate_dir = args.intermediate_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_index(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([row["sample_id"] for row in rows], dtype=str)
    subject_id = np.asarray([_norm_subject(row["subject_id"]) for row in rows], dtype=str)
    event_id = np.asarray([row["event_id"] for row in rows], dtype=str)
    target = np.asarray([row["labels"][LABEL_NAMES.index(args.target_label)] for row in rows], dtype=np.float32)
    split = _load_split(args.splits_root / args.protocol, n_rows=len(sample_id))
    split = _limit_split(split, max_train=args.max_train, max_val=args.max_val, max_test=args.max_test)

    selected = _selected_experiments(args.experiments)
    all_branch_data = _load_all_branches(sample_id)

    results = []
    runtime = {
        "protocol": args.protocol,
        "target_label": args.target_label,
        "supervised_train_rule": "pretrain + finetune from splits_new",
        "epochs": int(args.epochs),
        "hidden_dim": int(args.hidden_dim),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "torch_threads": int(args.torch_threads),
        "device": args.device,
        "split_counts": {name: int(len(values)) for name, values in split.items()},
    }
    (out_dir / "b0_fusion_runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    for offset, experiment in enumerate(selected):
        print(f"experiment={experiment.name} branches={','.join(experiment.branches)}", flush=True)
        tokens, token_mask, branch_report = _build_tokens(all_branch_data, experiment.branches)
        model, train_audit = _fit_model(
            tokens=tokens,
            token_mask=token_mask,
            target=target,
            train_idx=split["train"],
            val_idx=split["val"],
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            dropout=float(args.dropout),
            patience=int(args.patience),
            seed=int(args.seed) + offset,
            device=args.device,
        )
        predictions = {
            name: _predict(model, tokens, token_mask, indices=indices, device=args.device)
            for name, indices in split.items()
            if name in {"train", "val", "test"}
        }
        exp_dir = intermediate_dir / experiment.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model["state_dict"], exp_dir / "model.pt")
        np.savez_compressed(
            exp_dir / "predictions.npz",
            train_index=split["train"],
            val_index=split["val"],
            test_index=split["test"],
            train_prediction=predictions["train"],
            val_prediction=predictions["val"],
            test_prediction=predictions["test"],
            target=target,
            sample_id=sample_id,
            subject_id=subject_id,
        )
        _write_prediction_csv(
            exp_dir / "test_predictions.csv",
            indices=split["test"],
            sample_id=sample_id,
            subject_id=subject_id,
            event_id=event_id,
            target=target,
            prediction=predictions["test"],
        )
        test_metrics = _metrics(predictions["test"], target[split["test"]])
        result = {
            "experiment": experiment.name,
            "enabled_modalities": [BRANCHES[name].modality for name in experiment.branches],
            "branches": branch_report,
            "row_count": int(len(sample_id)),
            "protocol": args.protocol,
            "target_label": args.target_label,
            "split_counts": {name: int(len(values)) for name, values in split.items()},
            "mask_coverage_by_split": _mask_coverage_by_split(token_mask, experiment.branches, split),
            "train": _metrics(predictions["train"], target[split["train"]]),
            "val": _metrics(predictions["val"], target[split["val"]]),
            "test": test_metrics,
            "pooled_raw_pearson_r": test_metrics["pearson_r"],
            "within_subject_centered_r": _within_subject_centered_r(
                predictions["test"],
                target[split["test"]],
                subject_id[split["test"]],
            ),
            "per_subject_r": _per_subject_r(
                predictions["test"],
                target[split["test"]],
                subject_id[split["test"]],
            ),
            "train_audit": train_audit,
            "prediction_path": str(exp_dir / "predictions.npz"),
        }
        (out_dir / f"{experiment.name}_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results.append(result)
        print(
            "completed={name} rmse={rmse:.4f} mae={mae:.4f} r={r}".format(
                name=experiment.name,
                rmse=float(test_metrics["rmse"]),
                mae=float(test_metrics["mae"]),
                r=_fmt_float(test_metrics["pearson_r"]),
            ),
            flush=True,
        )

    summary = {
        "runtime": runtime,
        "experiment_count": len(results),
        "experiments": results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    (out_dir / "b0_fusion_matrix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "b0_fusion_matrix_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(f"summary_json={args.summary_json}")
    print(f"summary_md={args.summary_md}")
    return 0


def _load_index(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_split(path: Path, *, n_rows: int) -> dict[str, np.ndarray]:
    pretrain = _load_indices(path / "pretrain.json", n_rows=n_rows)
    finetune = _load_indices(path / "finetune.json", n_rows=n_rows)
    val = _load_indices(path / "val.json", n_rows=n_rows)
    test = _load_indices(path / "test.json", n_rows=n_rows)
    train = np.asarray(pretrain.tolist() + finetune.tolist(), dtype=np.int64)
    return {
        "pretrain": pretrain,
        "finetune": finetune,
        "train": train,
        "val": val,
        "test": test,
    }


def _load_indices(path: Path, *, n_rows: int) -> np.ndarray:
    values = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() >= n_rows):
        raise ValueError(f"{path} has out-of-range indices")
    return values


def _limit_split(
    split: dict[str, np.ndarray],
    *,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
) -> dict[str, np.ndarray]:
    limited = dict(split)
    if max_train is not None:
        limited["train"] = limited["train"][: max(0, int(max_train))]
    if max_val is not None:
        limited["val"] = limited["val"][: max(0, int(max_val))]
    if max_test is not None:
        limited["test"] = limited["test"][: max(0, int(max_test))]
    return limited


def _selected_experiments(raw: str | None) -> list[Experiment]:
    if not raw:
        return EXPERIMENTS
    wanted = {value.strip() for value in raw.split(",") if value.strip()}
    by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise ValueError(f"unknown experiments: {missing}")
    return [by_name[name] for name in [experiment.name for experiment in EXPERIMENTS] if name in wanted]


def _load_all_branches(sample_id: np.ndarray) -> dict[str, dict[str, Any]]:
    return {name: _load_branch(branch, sample_id=sample_id) for name, branch in BRANCHES.items()}


def _load_branch(branch: Branch, *, sample_id: np.ndarray) -> dict[str, Any]:
    with np.load(branch.path, allow_pickle=True) as loaded:
        loaded_sample_id = loaded["sample_id"].astype(str)
        if not np.array_equal(loaded_sample_id, sample_id):
            raise ValueError(f"{branch.name} sample_id order does not match canonical index")
        if branch.emb_key in loaded.files:
            emb = loaded[branch.emb_key].astype(np.float32)
        elif branch.modality == "video" and "face_emb" in loaded.files:
            emb = loaded["face_emb"].astype(np.float32)
        else:
            raise ValueError(f"{branch.path} missing {branch.emb_key}")
        mask = _load_mask(loaded, branch)
        if emb.shape != (len(sample_id), 256):
            raise ValueError(f"{branch.name} embedding shape {emb.shape} != ({len(sample_id)}, 256)")
        if mask.shape != (len(sample_id),):
            raise ValueError(f"{branch.name} mask shape {mask.shape} != ({len(sample_id)},)")
        return {
            "embedding": emb,
            "mask": mask.astype(bool),
            "path": str(branch.path),
            "mask_sum": int(mask.sum()),
        }


def _load_mask(loaded: Any, branch: Branch) -> np.ndarray:
    if branch.mask_key in loaded.files:
        return loaded[branch.mask_key].astype(np.int8)
    if "modality_mask" in loaded.files:
        modality_mask = loaded["modality_mask"].astype(np.int8)
        if modality_mask.ndim == 2 and modality_mask.shape[1] > branch.modality_index:
            return modality_mask[:, branch.modality_index]
    raise ValueError(f"{branch.path} missing {branch.mask_key} or modality_mask column {branch.modality_index}")


def _build_tokens(
    branch_data: dict[str, dict[str, Any]],
    branch_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tokens = []
    masks = []
    report = {}
    for name in branch_names:
        data = branch_data[name]
        tokens.append(data["embedding"])
        masks.append(data["mask"])
        report[name] = {"path": data["path"], "mask_sum": data["mask_sum"], "modality": BRANCHES[name].modality}
    return np.stack(tokens, axis=1).astype(np.float32), np.stack(masks, axis=1).astype(bool), report


class AttentionRegressor(torch.nn.Module):
    def __init__(self, *, modality_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = torch.nn.Linear(256, hidden_dim)
        self.modality_embedding = torch.nn.Parameter(torch.zeros(1, modality_count, hidden_dim))
        torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
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
        scores = torch.matmul(attended, self.query)
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(attended * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).reshape(-1)


def _fit_model(
    *,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    target: np.ndarray,
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
    y_mean = float(target[train_idx].mean())
    y_std = float(target[train_idx].std()) or 1.0
    dev = torch.device(device)
    module = AttentionRegressor(modality_count=tokens.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(dev)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    x_train = torch.as_tensor(_normalize_tokens(tokens[train_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_train = torch.as_tensor(token_mask[train_idx], dtype=torch.bool, device=dev)
    y_train = torch.as_tensor((target[train_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    x_val = torch.as_tensor(_normalize_tokens(tokens[val_idx], x_mean, x_std), dtype=torch.float32, device=dev)
    m_val = torch.as_tensor(token_mask[val_idx], dtype=torch.bool, device=dev)
    y_val = torch.as_tensor((target[val_idx] - y_mean) / y_std, dtype=torch.float32, device=dev)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    rng = np.random.default_rng(seed)
    initial_loss = None
    final_loss = None
    for epoch in range(max(1, epochs)):
        module.train()
        order = rng.permutation(len(train_idx))
        losses = []
        for start in range(0, len(order), max(1, batch_size)):
            batch = order[start : start + max(1, batch_size)]
            prediction = module(x_train[batch], m_train[batch])
            loss = loss_fn(prediction, y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(losses)) if losses else math.nan
        if initial_loss is None:
            initial_loss = train_loss
        module.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(module(x_val, m_val), y_val).detach().cpu().item())
        final_loss = train_loss
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
    module.eval()
    return {
        "module": module,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "state_dict": module.state_dict(),
    }, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "initial_train_loss": float(initial_loss or math.nan),
        "final_train_loss": float(final_loss or math.nan),
        "normalization": "train_only",
        "train_count": int(len(train_idx)),
    }


def _predict(
    model: dict[str, Any],
    tokens: np.ndarray,
    token_mask: np.ndarray,
    *,
    indices: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    dev = torch.device(device)
    module: AttentionRegressor = model["module"]
    preds = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            x = torch.as_tensor(
                _normalize_tokens(tokens[idx], model["x_mean"], model["x_std"]),
                dtype=torch.float32,
                device=dev,
            )
            mask = torch.as_tensor(token_mask[idx], dtype=torch.bool, device=dev)
            pred = module(x, mask).detach().cpu().numpy().astype(np.float32)
            preds.append(pred * float(model["y_std"]) + float(model["y_mean"]))
    return np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)


def _fit_token_normalization(tokens: np.ndarray, mask: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = np.where(mask[indices, :, None], tokens[indices], np.nan)
    mean = np.nanmean(available, axis=(0, 1), keepdims=True)
    std = np.nanstd(available, axis=(0, 1), keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((np.isfinite(std)) & (std >= 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def _normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    err = prediction - truth
    return {
        "count": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(err ** 2))) if len(truth) else None,
        "mae": float(np.mean(np.abs(err))) if len(truth) else None,
        "pearson_r": _pearson(prediction, truth),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _within_subject_centered_r(prediction: np.ndarray, truth: np.ndarray, subjects: np.ndarray) -> float | None:
    pred_centered = prediction.astype(np.float32).copy()
    truth_centered = truth.astype(np.float32).copy()
    for subject in np.unique(subjects):
        mask = subjects == subject
        pred_centered[mask] -= float(pred_centered[mask].mean())
        truth_centered[mask] -= float(truth_centered[mask].mean())
    return _pearson(pred_centered, truth_centered)


def _per_subject_r(prediction: np.ndarray, truth: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    values = []
    rows = []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        r = _pearson(prediction[mask], truth[mask])
        rows.append({"subject_id": subject, "count": int(mask.sum()), "pearson_r": r})
        if r is not None:
            values.append(float(r))
    return {
        "mean": None if not values else float(np.mean(values)),
        "std": None if not values else float(np.std(values)),
        "subject_count": int(len(rows)),
        "valid_subject_r_count": int(len(values)),
        "subjects": rows,
    }


def _mask_coverage_by_split(
    token_mask: np.ndarray,
    branch_names: tuple[str, ...],
    split: dict[str, np.ndarray],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        indices = split[split_name]
        rows = {}
        for col, branch_name in enumerate(branch_names):
            valid = int(token_mask[indices, col].sum())
            rows[branch_name] = {
                "valid": valid,
                "total": int(len(indices)),
                "coverage": 0.0 if len(indices) == 0 else float(valid / len(indices)),
            }
        report[split_name] = rows
    return report


def _write_prediction_csv(
    path: Path,
    *,
    indices: np.ndarray,
    sample_id: np.ndarray,
    subject_id: np.ndarray,
    event_id: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "subject_id", "event_id", "target", "prediction"])
        for local_idx, row_idx in enumerate(indices.tolist()):
            writer.writerow([sample_id[row_idx], subject_id[row_idx], event_id[row_idx], float(target[row_idx]), float(prediction[local_idx])])


def _summary_markdown(summary: dict[str, Any]) -> str:
    runtime = summary["runtime"]
    lines = [
        "# B0 Fusion Matrix Summary",
        "",
        f"protocol: `{runtime['protocol']}`",
        f"target_label: `{runtime['target_label']}`",
        f"supervised_train_rule: `{runtime['supervised_train_rule']}`",
        f"split_counts: `{runtime['split_counts']}`",
        "",
        "| experiment | modalities | rows | RMSE | MAE | pooled r | centered r | per-subject r mean/std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["experiments"]:
        ps = row["per_subject_r"]
        lines.append(
            "| {experiment} | {modalities} | {rows} | {rmse} | {mae} | {r} | {centered} | {ps_mean}/{ps_std} |".format(
                experiment=row["experiment"],
                modalities=",".join(row["enabled_modalities"]),
                rows=row["row_count"],
                rmse=_fmt_float(row["test"]["rmse"]),
                mae=_fmt_float(row["test"]["mae"]),
                r=_fmt_float(row["pooled_raw_pearson_r"]),
                centered=_fmt_float(row["within_subject_centered_r"]),
                ps_mean=_fmt_float(ps["mean"]),
                ps_std=_fmt_float(ps["std"]),
            )
        )
    lines.extend(["", "## Mask Coverage", ""])
    for row in summary["experiments"]:
        lines.append(f"### {row['experiment']}")
        lines.append("")
        lines.append("| split | branch | valid | total | coverage |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for split_name, coverage in row["mask_coverage_by_split"].items():
            for branch, values in coverage.items():
                lines.append(
                    f"| {split_name} | {branch} | {values['valid']} | {values['total']} | {_fmt_float(values['coverage'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def _fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    return f"sub-{int(float(text)):02d}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
