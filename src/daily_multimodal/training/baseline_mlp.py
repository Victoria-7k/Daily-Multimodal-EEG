from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODALITY_RUNS: dict[str, tuple[str, ...]] = {
    "eeg_only": ("eeg",),
    "wear_only": ("wear",),
    "audio_only": ("audio",),
    "face_only": ("face",),
    "eeg_wear": ("eeg", "wear"),
    "eeg_audio": ("eeg", "audio"),
    "eeg_face": ("eeg", "face"),
    "full": ("eeg", "wear", "audio", "face"),
}
MODALITY_TO_EMB = {
    "eeg": "eeg_emb",
    "wear": "wear_emb",
    "face": "face_emb",
    "audio": "audio_emb",
}
MODALITY_TO_MASK_INDEX = {"eeg": 0, "wear": 1, "face": 2, "audio": 3}


@dataclass
class TrainedMlp:
    weights1: np.ndarray
    bias1: np.ndarray
    weights2: np.ndarray
    bias2: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def run_baseline_experiment(
    *,
    embeddings_path: Path | str,
    model_out: Path | str,
    metrics_out: Path | str,
    table_out: Path | str,
    target_label: str | None = None,
    overfit_limit: int = 128,
    epochs: int = 200,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    seed: int = 13,
) -> dict[str, Any]:
    data = _load_embedding_dataset(embeddings_path, target_label=target_label)
    split = _subject_split(data["subject_id"])
    overfit = _run_overfit_check(
        _features_for_modalities(data, MODALITY_RUNS["full"]),
        data["target"],
        limit=overfit_limit,
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )

    runs: dict[str, Any] = {}
    saved_model: TrainedMlp | None = None
    for offset, (run_name, modalities) in enumerate(MODALITY_RUNS.items()):
        x = _features_for_modalities(data, modalities)
        valid_mask = _available_modality_mask(data["modality_mask"], modalities)
        run_split = {
            name: indices[valid_mask[indices]]
            for name, indices in split["indices"].items()
        }
        model = _fit_mlp(
            x[run_split["train"]],
            data["target"][run_split["train"]],
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed + offset,
        )
        runs[run_name] = {
            "modalities": list(modalities),
            "train": _evaluate_split(model, x, data["target"], run_split["train"]),
            "val": _evaluate_split(model, x, data["target"], run_split["val"]),
            "test": _evaluate_split(model, x, data["target"], run_split["test"]),
            "sample_counts": {name: int(len(indices)) for name, indices in run_split.items()},
        }
        if run_name == "full":
            saved_model = model

    result = {
        "stage": 9,
        "target_label": data["target_label"],
        "split": {
            "train_subjects": split["train_subjects"],
            "val_subjects": split["val_subjects"],
            "test_subjects": split["test_subjects"],
        },
        "overfit_check": overfit,
        "runs": runs,
    }
    _write_metrics(result, metrics_out)
    _write_table(result, table_out)
    if saved_model is None:
        raise RuntimeError("full run did not produce a model")
    _save_model(saved_model, model_out, metadata={"target_label": data["target_label"], "run": "full"})
    return result


def _load_embedding_dataset(path: Path | str, *, target_label: str | None) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as loaded:
        labels = [json.loads(value) for value in loaded["labels"].tolist()]
        chosen_label = target_label or _first_numeric_label(labels)
        target = np.array([float(row[chosen_label]) for row in labels], dtype=np.float32)
        return {
            "sample_id": loaded["sample_id"].astype(str),
            "subject_id": loaded["subject_id"].astype(str),
            "target": target,
            "target_label": chosen_label,
            "modality_mask": loaded["modality_mask"].astype(bool),
            "eeg_emb": loaded["eeg_emb"].astype(np.float32),
            "wear_emb": loaded["wear_emb"].astype(np.float32),
            "face_emb": loaded["face_emb"].astype(np.float32),
            "audio_emb": loaded["audio_emb"].astype(np.float32),
        }


def _first_numeric_label(labels: list[dict[str, Any]]) -> str:
    for row in labels:
        for key, value in row.items():
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            return key
    raise ValueError("No numeric label found in embeddings labels.")


def _subject_split(subjects: np.ndarray) -> dict[str, Any]:
    train_default = {f"sub-{idx:02d}" for idx in range(1, 11)}
    val_default = {"sub-11", "sub-12"}
    test_default = {"sub-13", "sub-14", "sub-15"}
    observed = list(dict.fromkeys(subjects.tolist()))
    train_subjects = [subject for subject in observed if subject in train_default]
    val_subjects = [subject for subject in observed if subject in val_default]
    test_subjects = [subject for subject in observed if subject in test_default]

    assigned = set(train_subjects + val_subjects + test_subjects)
    leftovers = [subject for subject in observed if subject not in assigned]
    if not train_subjects and leftovers:
        train_subjects.append(leftovers.pop(0))
    if not val_subjects and leftovers:
        val_subjects.append(leftovers.pop(0))
    if not test_subjects and leftovers:
        test_subjects.append(leftovers.pop(0))

    return {
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "indices": {
            "train": np.flatnonzero(np.isin(subjects, train_subjects)),
            "val": np.flatnonzero(np.isin(subjects, val_subjects)),
            "test": np.flatnonzero(np.isin(subjects, test_subjects)),
        },
    }


def _features_for_modalities(data: dict[str, Any], modalities: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([data[MODALITY_TO_EMB[modality]] for modality in modalities], axis=1)


def _available_modality_mask(mask: np.ndarray, modalities: tuple[str, ...]) -> np.ndarray:
    columns = [MODALITY_TO_MASK_INDEX[modality] for modality in modalities]
    return mask[:, columns].all(axis=1)


def _run_overfit_check(
    x: np.ndarray,
    y: np.ndarray,
    *,
    limit: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    n = min(limit, len(y))
    model = _fit_mlp(
        x[:n],
        y[:n],
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
        capture_initial=True,
    )
    initial_loss = float(getattr(model, "initial_loss"))
    final_loss = _mse(_predict(model, x[:n]), y[:n])
    return {
        "sample_count": int(n),
        "initial_loss": initial_loss,
        "final_loss": float(final_loss),
        "passed": bool(final_loss < initial_loss),
    }


def _fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    capture_initial: bool = False,
) -> TrainedMlp:
    if len(y) == 0:
        raise ValueError("Cannot train baseline MLP with an empty training split.")
    rng = np.random.default_rng(seed)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_norm = (x - x_mean) / x_std
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0
    y_norm = ((y - y_mean) / y_std).reshape(-1, 1)

    input_dim = x_norm.shape[1]
    weights1 = rng.normal(0.0, 0.05, size=(input_dim, hidden_dim)).astype(np.float32)
    bias1 = np.zeros((1, hidden_dim), dtype=np.float32)
    weights2 = rng.normal(0.0, 0.05, size=(hidden_dim, 1)).astype(np.float32)
    bias2 = np.zeros((1, 1), dtype=np.float32)

    initial_loss = _forward_loss(x_norm, y_norm, weights1, bias1, weights2, bias2)
    for _ in range(max(1, epochs)):
        hidden = np.tanh(x_norm @ weights1 + bias1)
        pred = hidden @ weights2 + bias2
        grad_pred = (2.0 / len(y_norm)) * (pred - y_norm)
        grad_w2 = hidden.T @ grad_pred
        grad_b2 = grad_pred.sum(axis=0, keepdims=True)
        grad_hidden = grad_pred @ weights2.T
        grad_z1 = grad_hidden * (1.0 - hidden**2)
        grad_w1 = x_norm.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0, keepdims=True)
        weights1 -= learning_rate * grad_w1
        bias1 -= learning_rate * grad_b1
        weights2 -= learning_rate * grad_w2
        bias2 -= learning_rate * grad_b2

    model = TrainedMlp(
        weights1=weights1,
        bias1=bias1,
        weights2=weights2,
        bias2=bias2,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean,
        y_std=y_std,
    )
    if capture_initial:
        setattr(model, "initial_loss", initial_loss * (y_std**2))
    return model


def _forward_loss(
    x: np.ndarray,
    y: np.ndarray,
    weights1: np.ndarray,
    bias1: np.ndarray,
    weights2: np.ndarray,
    bias2: np.ndarray,
) -> float:
    pred = np.tanh(x @ weights1 + bias1) @ weights2 + bias2
    return float(np.mean((pred - y) ** 2))


def _predict(model: TrainedMlp, x: np.ndarray) -> np.ndarray:
    x_norm = (x - model.x_mean) / model.x_std
    pred = np.tanh(x_norm @ model.weights1 + model.bias1) @ model.weights2 + model.bias2
    return (pred.reshape(-1) * model.y_std + model.y_mean).astype(np.float32)


def _evaluate_split(model: TrainedMlp, x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> dict[str, float | int | None]:
    if len(indices) == 0:
        return {"count": 0, "mae": None, "rmse": None, "pearson": None}
    pred = _predict(model, x[indices])
    truth = y[indices]
    return {
        "count": int(len(indices)),
        "mae": float(np.mean(np.abs(pred - truth))),
        "rmse": float(math.sqrt(np.mean((pred - truth) ** 2))),
        "pearson": _pearson(pred, truth),
    }


def _mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((pred - truth) ** 2))


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _write_metrics(result: dict[str, Any], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _write_table(result: dict[str, Any], output: Path | str) -> Path:
    rows = [
        "| run | modalities | train_rmse | val_rmse | test_rmse | test_mae |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run_name, run in result["runs"].items():
        rows.append(
            "| {run} | {mods} | {train} | {val} | {test} | {mae} |".format(
                run=run_name,
                mods="+".join(run["modalities"]),
                train=_format_metric(run["train"]["rmse"]),
                val=_format_metric(run["val"]["rmse"]),
                test=_format_metric(run["test"]["rmse"]),
                mae=_format_metric(run["test"]["mae"]),
            )
        )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _save_model(model: TrainedMlp, output: Path | str, *, metadata: dict[str, Any]) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        np.savez_compressed(
            handle,
            weights1=model.weights1,
            bias1=model.bias1,
            weights2=model.weights2,
            bias2=model.bias2,
            x_mean=model.x_mean,
            x_std=model.x_std,
            y_mean=np.array([model.y_mean], dtype=np.float32),
            y_std=np.array([model.y_std], dtype=np.float32),
            metadata=np.array([json.dumps(metadata, ensure_ascii=False)], dtype=object),
        )
    return out
