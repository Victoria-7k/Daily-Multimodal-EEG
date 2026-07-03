from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.training.baseline_mlp import TrainedMlp, _fit_mlp, _predict
from daily_multimodal.training.subject_cv import build_subject_folds


QUALITY_FLAG_FIELDS = (
    "motion_intensity",
    "stationary_ratio",
    "heart_rate_plausible",
    "ppg_hr_plausible",
    "ppg_peak_insufficient",
    "gsr_slope_abnormal",
    "gsr_scr_abnormal",
    "acc_motion_high",
    "wear_quality_risk_count",
)


def run_wear_quality_ablation(
    *,
    window_index: Path | str,
    physio_embeddings: Path | str,
    deep_embeddings: Path | str,
    target_label: str,
    out_json: Path | str,
    out_table: Path | str,
    epochs: int = 200,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    seed: int = 31,
) -> dict[str, Any]:
    labels = _load_window_labels(window_index, target_label=target_label)
    physio = _load_wear_dataset(physio_embeddings, labels=labels)
    deep = _load_wear_dataset(deep_embeddings, labels=labels)
    specs = [
        ("W1_physio_full", physio, "wear_physio_features_v2", "all", False, False, 0),
        ("W2_deep_full", deep, "wear_deep_sequence_v1", "all", False, False, 1),
        ("W3_physio_high_quality", physio, "wear_physio_features_v2", "A", False, False, 2),
        ("W4_deep_high_quality", deep, "wear_deep_sequence_v1", "A", False, False, 3),
        ("W5a_deep_full", deep, "wear_deep_sequence_v1", "all", False, False, 50),
        ("W5b_deep_quality_flags_full", deep, "wear_deep_sequence_v1", "all", True, False, 50),
        ("W5c_deep_sample_weights_full", deep, "wear_deep_sequence_v1", "all", False, True, 50),
        ("W5d_deep_quality_flags_sample_weights_full", deep, "wear_deep_sequence_v1", "all", True, True, 50),
        ("W6_physio_ab_quality", physio, "wear_physio_features_v2", "A+B", False, False, 6),
        ("W7_deep_ab_quality", deep, "wear_deep_sequence_v1", "A+B", False, False, 7),
    ]
    experiments: dict[str, Any] = {}
    for name, data, feature_set, quality_subset, include_quality_flags, use_sample_weight, seed_offset in specs:
        experiments[name] = _run_experiment(
            name,
            data,
            feature_set=feature_set,
            quality_subset=quality_subset,
            include_quality_flags=include_quality_flags,
            use_sample_weight=use_sample_weight,
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed + seed_offset,
        )
    result = {
        "stage": 22,
        "target_label": target_label,
        "window_index": str(window_index),
        "physio_embeddings": str(physio_embeddings),
        "deep_embeddings": str(deep_embeddings),
        "experiments": experiments,
    }
    _write_outputs(result, out_json=out_json, out_table=out_table)
    return result


def _load_window_labels(path: Path | str, *, target_label: str) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            label_values = row.get("label_columns") or {}
            if sample_id and target_label in label_values:
                labels[sample_id] = {
                    "target": float(label_values[target_label]),
                    "subject_id": str(row.get("subject_id", "")),
                }
    return labels


def _load_wear_dataset(path: Path | str, *, labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as loaded:
        sample_ids = loaded["sample_id"].astype(str)
        subject_ids = loaded["subject_id"].astype(str)
        wear_emb = loaded["wear_emb"].astype(np.float32)
        masks = loaded["modality_mask"].astype(np.int8)
        quality_values = loaded["quality_flags"].tolist() if "quality_flags" in loaded.files else ["{}"] * len(sample_ids)
    rows = []
    for index, sample_id in enumerate(sample_ids):
        if sample_id not in labels or int(masks[index, 1]) != 1:
            continue
        quality = _parse_json_object(quality_values[index])
        rows.append(
            {
                "sample_id": sample_id,
                "subject_id": labels[sample_id].get("subject_id") or subject_ids[index],
                "target": float(labels[sample_id]["target"]),
                "wear_emb": wear_emb[index],
                "quality_flags": quality,
            }
        )
    if not rows:
        return {
            "sample_id": np.array([], dtype=str),
            "subject_id": np.array([], dtype=str),
            "target": np.zeros((0,), dtype=np.float32),
            "wear_emb": np.zeros((0, 256), dtype=np.float32),
            "quality_flags": [],
        }
    return {
        "sample_id": np.array([row["sample_id"] for row in rows], dtype=str),
        "subject_id": np.array([row["subject_id"] for row in rows], dtype=str),
        "target": np.array([row["target"] for row in rows], dtype=np.float32),
        "wear_emb": np.stack([row["wear_emb"] for row in rows]).astype(np.float32),
        "quality_flags": [row["quality_flags"] for row in rows],
    }


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _run_experiment(
    name: str,
    data: dict[str, Any],
    *,
    feature_set: str,
    quality_subset: str,
    include_quality_flags: bool,
    use_sample_weight: bool,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    selected = _selection_mask(data["quality_flags"], quality_subset=quality_subset)
    selected_qualities = [q for q, keep in zip(data["quality_flags"], selected) if keep]
    x = data["wear_emb"][selected]
    if include_quality_flags:
        x = np.concatenate([x, _quality_feature_matrix(selected_qualities)], axis=1)
    y = data["target"][selected]
    subjects = data["subject_id"][selected]
    sample_weight = _sample_weight_vector(selected_qualities) if use_sample_weight else np.ones((len(y),), dtype=np.float32)
    if len(y) == 0:
        return _failed_experiment(name, feature_set, quality_subset, include_quality_flags, use_sample_weight, "no selected rows")
    try:
        folds = build_subject_folds(subjects, strategy="leave_one_subject_out")
    except ValueError as exc:
        return _failed_experiment(name, feature_set, quality_subset, include_quality_flags, use_sample_weight, str(exc))

    fold_results = []
    for offset, fold in enumerate(folds):
        if len(fold.train) == 0 or len(fold.val) == 0 or len(fold.test) == 0:
            return _failed_experiment(name, feature_set, quality_subset, include_quality_flags, use_sample_weight, f"{fold.name} has empty split")
        if use_sample_weight:
            model = _fit_weighted_mlp(
                x[fold.train],
                y[fold.train],
                sample_weight[fold.train],
                epochs=epochs,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                seed=seed + offset,
            )
        else:
            model = _fit_mlp(
                x[fold.train],
                y[fold.train],
                epochs=epochs,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                seed=seed + offset,
            )
        fold_results.append(
            {
                "fold": fold.name,
                "train": _evaluate(model, x, y, fold.train),
                "val": _evaluate(model, x, y, fold.val),
                "test": _evaluate(model, x, y, fold.test),
            }
        )
    summary = _summarize_folds(fold_results)
    return {
        "experiment": name,
        "feature_set": feature_set,
        "quality_subset": quality_subset,
        "high_quality_only": quality_subset == "A",
        "include_quality_flags": bool(include_quality_flags),
        "use_sample_weight": bool(use_sample_weight),
        "sample_weight_mean": float(np.mean(sample_weight)) if sample_weight.size else None,
        "sample_weight_std": float(np.std(sample_weight)) if sample_weight.size else None,
        "row_count": int(len(y)),
        "fold_count": int(len(fold_results)),
        "folds": fold_results,
        **summary,
    }


def _selection_mask(qualities: list[dict[str, Any]], *, quality_subset: str) -> np.ndarray:
    if quality_subset == "all":
        return np.ones((len(qualities),), dtype=bool)
    if quality_subset == "A":
        return np.asarray([str(item.get("wear_quality_grade", "")) == "A" for item in qualities], dtype=bool)
    if quality_subset == "A+B":
        return np.asarray([str(item.get("wear_quality_grade", "")) in {"A", "B"} for item in qualities], dtype=bool)
    raise ValueError(f"unsupported quality subset: {quality_subset}")


def _quality_feature_matrix(qualities: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for item in qualities:
        row: list[float] = []
        for field in QUALITY_FLAG_FIELDS:
            value = item.get(field)
            if isinstance(value, bool):
                row.append(1.0 if value else 0.0)
            elif value is None:
                row.append(0.0)
            else:
                row.append(float(value))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(QUALITY_FLAG_FIELDS)), dtype=np.float32)


def _sample_weight_vector(qualities: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for item in qualities:
        grade = str(item.get("wear_quality_grade", "C"))
        if grade == "A":
            values.append(1.0)
        elif grade == "B":
            values.append(0.7)
        else:
            values.append(0.35)
    return np.asarray(values, dtype=np.float32)


def _fit_weighted_mlp(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> TrainedMlp:
    if len(y) == 0:
        raise ValueError("Cannot train weighted MLP with an empty training split.")
    rng = np.random.default_rng(seed)
    sample_weight = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    if sample_weight.shape[0] != len(y):
        raise ValueError("sample weights must match y")
    weight_sum = float(np.sum(sample_weight))
    if weight_sum <= 0.0:
        sample_weight = np.ones((len(y), 1), dtype=np.float32)
        weight_sum = float(len(y))
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
    for _ in range(max(1, epochs)):
        hidden = np.tanh(x_norm @ weights1 + bias1)
        pred = hidden @ weights2 + bias2
        grad_pred = (2.0 / weight_sum) * sample_weight * (pred - y_norm)
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
    return TrainedMlp(
        weights1=weights1,
        bias1=bias1,
        weights2=weights2,
        bias2=bias2,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean,
        y_std=y_std,
    )


def _evaluate(model: Any, x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> dict[str, float | int | None]:
    if len(indices) == 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "pred_std": None,
            "truth_std": None,
            "error_std": None,
        }
    pred = _predict(model, x[indices])
    truth = y[indices]
    error = pred - truth
    return {
        "count": int(len(indices)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(np.square(error)))),
        "pearson": _pearson(pred, truth),
        "pred_std": float(np.std(pred)),
        "truth_std": float(np.std(truth)),
        "error_std": float(np.std(error)),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metric_map = {
        "rmse": "rmse",
        "pearson_r": "pearson",
        "pred_std": "pred_std",
        "truth_std": "truth_std",
        "error_std": "error_std",
    }
    out: dict[str, Any] = {}
    for output_name, fold_key in metric_map.items():
        values = np.asarray(
            [fold["test"][fold_key] for fold in folds if fold["test"].get(fold_key) is not None],
            dtype=np.float32,
        )
        out[f"{output_name}_mean"] = None if values.size == 0 else float(np.mean(values))
        out[f"{output_name}_std"] = None if values.size == 0 else float(np.std(values))
    return out


def _failed_experiment(
    name: str,
    feature_set: str,
    quality_subset: str,
    include_quality_flags: bool,
    use_sample_weight: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "experiment": name,
        "feature_set": feature_set,
        "quality_subset": quality_subset,
        "high_quality_only": quality_subset == "A",
        "include_quality_flags": bool(include_quality_flags),
        "use_sample_weight": bool(use_sample_weight),
        "sample_weight_mean": None,
        "sample_weight_std": None,
        "row_count": 0,
        "fold_count": 0,
        "failure": reason,
        "folds": [],
        "rmse_mean": None,
        "rmse_std": None,
        "pearson_r_mean": None,
        "pearson_r_std": None,
        "pred_std_mean": None,
        "pred_std_std": None,
        "truth_std_mean": None,
        "truth_std_std": None,
        "error_std_mean": None,
        "error_std_std": None,
    }


def _write_outputs(result: dict[str, Any], *, out_json: Path | str, out_table: Path | str) -> None:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_table(result, out_table)


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| experiment | rows | RMSE mean ± std | Pearson r mean ± std | pred_std mean ± std | truth_std mean ± std | error_std mean ± std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, experiment in result["experiments"].items():
        rows.append(
            "| {name} | {rows} | {rmse} | {r} | {pred_std} | {truth_std} | {error_std} |".format(
                name=name,
                rows=experiment.get("row_count", 0),
                rmse=_format_pair(experiment.get("rmse_mean"), experiment.get("rmse_std")),
                r=_format_pair(experiment.get("pearson_r_mean"), experiment.get("pearson_r_std")),
                pred_std=_format_pair(experiment.get("pred_std_mean"), experiment.get("pred_std_std")),
                truth_std=_format_pair(experiment.get("truth_std_mean"), experiment.get("truth_std_std")),
                error_std=_format_pair(experiment.get("error_std_mean"), experiment.get("error_std_std")),
            )
        )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _format_pair(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "NA"
    return f"{float(mean):.4f} ± {float(std):.4f}"
