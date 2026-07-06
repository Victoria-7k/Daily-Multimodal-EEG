from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.embeddings.contracts import validate_embedding_shape
from daily_multimodal.training.video_variant_ablation import FACE_MASK_INDEX, _build_video_folds, _pearson


def run_video_embedding_probes(
    *,
    embeddings: Path | str,
    train_embeddings: Path | str | None = None,
    target_label: str,
    out_json: Path | str,
    out_table: Path | str,
    seed: int = 41,
    n_splits: int = 5,
    fold_strategy: str = "shuffled_k_fold",
    p3_fold_strategy: str | None = None,
) -> dict[str, Any]:
    p3_strategy = p3_fold_strategy or fold_strategy
    data = _load_probe_dataset(embeddings, target_label=target_label)
    if train_embeddings is not None:
        train_data = _load_probe_dataset(train_embeddings, target_label=target_label)
        data = _attach_train_probe_embeddings(data, train_data)
    result = {
        "embeddings": str(embeddings),
        "train_embeddings": None if train_embeddings is None else str(train_embeddings),
        "target_label": target_label,
        "row_count": int(len(data["target"])),
        "probes": {
            "P1_subject_logreg": _classification_probe(
                data["embedding"],
                data["subject_id"],
                train_x=data.get("train_embedding"),
                seed=seed,
                n_splits=n_splits,
            ),
            "P2_within_subject_session_logreg": _within_subject_session_probe(
                data,
                seed=seed,
                n_splits=n_splits,
            ),
            "P3_fatigue_ridge": _ridge_probe(
                data,
                seed=seed,
                n_splits=n_splits,
                fold_strategy=p3_strategy,
            ),
        },
    }
    _write_json(result, out_json)
    _write_table(result, out_table)
    return _json_ready(result)


def _load_probe_dataset(path: Path | str, *, target_label: str) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as loaded:
        sample_id = loaded["sample_id"].astype(str)
        subject_id = loaded["subject_id"].astype(str)
        event_id = loaded["event_id"].astype(str) if "event_id" in loaded.files else sample_id.copy()
        labels = [_parse_json_object(value) for value in loaded["labels"].tolist()]
        embedding = validate_embedding_shape("face_emb", loaded["face_emb"]).astype(np.float32)
        mask = loaded["modality_mask"].astype(np.int8)[:, FACE_MASK_INDEX].astype(bool)
    target = np.asarray([float(row[target_label]) for row in labels], dtype=np.float32)
    session_id = np.asarray(
        [_session_id(subject, event, sample) for subject, event, sample in zip(subject_id, event_id, sample_id)],
        dtype=str,
    )
    return {
        "sample_id": sample_id[mask],
        "subject_id": subject_id[mask],
        "event_id": event_id[mask],
        "session_id": session_id[mask],
        "target": target[mask],
        "embedding": embedding[mask],
    }


def _attach_train_probe_embeddings(data: dict[str, Any], train_data: dict[str, Any]) -> dict[str, Any]:
    aligned = _subset_probe_by_sample_ids(train_data, data["sample_id"].tolist())
    for key in ("sample_id", "subject_id", "event_id", "session_id"):
        if aligned[key].astype(str).tolist() != data[key].astype(str).tolist():
            raise ValueError(f"train embeddings metadata mismatch for {key}")
    if not np.allclose(aligned["target"], data["target"]):
        raise ValueError("train embeddings target values do not match eval embeddings")
    return {**data, "train_embedding": aligned["embedding"]}


def _subset_probe_by_sample_ids(data: dict[str, Any], sample_ids: list[str]) -> dict[str, Any]:
    index = {sample_id: idx for idx, sample_id in enumerate(data["sample_id"].astype(str).tolist())}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        raise ValueError(f"train embeddings missing sample_id values: {missing[:5]}")
    indices = np.asarray([index[sample_id] for sample_id in sample_ids], dtype=np.int64)
    return {
        "sample_id": data["sample_id"][indices],
        "subject_id": data["subject_id"][indices],
        "event_id": data["event_id"][indices],
        "session_id": data["session_id"][indices],
        "target": data["target"][indices],
        "embedding": data["embedding"][indices],
    }


def _classification_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_x: np.ndarray | None = None,
    seed: int,
    n_splits: int,
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return _classification_probe_numpy(x, y, train_x=train_x, seed=seed, n_splits=n_splits, backend=f"numpy_fallback:{exc}")
    y = np.asarray(y).astype(str)
    split_count = _stratified_split_count(y, n_splits)
    if split_count < 2:
        return {"failure": "not enough samples per class", "class_count": int(len(set(y.tolist())))}
    accs = []
    f1s = []
    folds = []
    cv = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=seed)
    for index, (train, test) in enumerate(cv.split(x, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
        fit_x = x if train_x is None else train_x
        model.fit(fit_x[train], y[train])
        pred = model.predict(x[test])
        acc = float(accuracy_score(y[test], pred))
        f1 = float(f1_score(y[test], pred, average="macro"))
        accs.append(acc)
        f1s.append(f1)
        folds.append({"fold": index, "test_count": int(len(test)), "accuracy": acc, "macro_f1": f1})
    return {
        "class_count": int(len(set(y.tolist()))),
        "fold_count": int(split_count),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "folds": folds,
    }


def _within_subject_session_probe(data: dict[str, Any], *, seed: int, n_splits: int) -> dict[str, Any]:
    subject_results = []
    for subject in sorted(set(data["subject_id"].tolist())):
        indices = np.flatnonzero(data["subject_id"] == subject)
        sessions = data["session_id"][indices]
        if len(set(sessions.tolist())) < 2:
            continue
        result = _classification_probe(
            data["embedding"][indices],
            sessions,
            train_x=None if "train_embedding" not in data else data["train_embedding"][indices],
            seed=seed,
            n_splits=n_splits,
        )
        if "failure" not in result:
            result = {**result, "subject_id": subject, "row_count": int(len(indices))}
            subject_results.append(result)
    accuracies = np.asarray([row["accuracy_mean"] for row in subject_results], dtype=np.float32)
    f1s = np.asarray([row["macro_f1_mean"] for row in subject_results], dtype=np.float32)
    return {
        "subject_count": int(len(subject_results)),
        "accuracy_mean": None if accuracies.size == 0 else float(np.mean(accuracies)),
        "accuracy_std": None if accuracies.size == 0 else float(np.std(accuracies)),
        "macro_f1_mean": None if f1s.size == 0 else float(np.mean(f1s)),
        "macro_f1_std": None if f1s.size == 0 else float(np.std(f1s)),
        "subjects": subject_results,
    }


def _ridge_probe(
    data: dict[str, Any],
    *,
    seed: int,
    n_splits: int,
    fold_strategy: str,
) -> dict[str, Any]:
    x = data["embedding"]
    train_x = data.get("train_embedding", x)
    y = data["target"]
    if fold_strategy != "shuffled_k_fold":
        try:
            video_folds = _build_video_folds(
                {
                    "sample_id": data["sample_id"],
                    "subject_id": data["subject_id"],
                    "event_id": data["event_id"],
                    "target": data["target"],
                },
                strategy=fold_strategy,
                n_splits=n_splits,
                seed=seed,
            )
        except ValueError as exc:
            return {"failure": str(exc), "fold_strategy": fold_strategy}
        splits = [(fold.name, fold.train, fold.test) for fold in video_folds]
        return _ridge_probe_from_splits(x, y, train_x=train_x, splits=splits, fold_strategy=fold_strategy)

    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return _ridge_probe_numpy(x, y, train_x=train_x, seed=seed, n_splits=n_splits, backend=f"numpy_fallback:{exc}")
    split_count = max(2, min(int(n_splits), len(y)))
    cv = KFold(n_splits=split_count, shuffle=True, random_state=seed)
    rmses = []
    rs = []
    folds = []
    for index, (train, test) in enumerate(cv.split(x)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_x[train], y[train])
        pred = model.predict(x[test])
        rmse = float(math.sqrt(np.mean(np.square(pred - y[test]))))
        r = _pearson(np.asarray(pred, dtype=np.float32), y[test].astype(np.float32))
        rmses.append(rmse)
        if r is not None:
            rs.append(r)
        folds.append({"fold": f"shuffled_{index:02d}", "test_count": int(len(test)), "rmse": rmse, "pearson": r})
    return {
        "fold_strategy": fold_strategy,
        "fold_count": int(split_count),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "pearson_r_mean": None if not rs else float(np.mean(rs)),
        "pearson_r_std": None if not rs else float(np.std(rs)),
        "folds": folds,
    }


def _ridge_probe_from_splits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_x: np.ndarray,
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    fold_strategy: str,
) -> dict[str, Any]:
    rmses = []
    rs = []
    folds = []
    for fold_name, train, test in splits:
        if len(train) == 0 or len(test) == 0:
            continue
        pred = _ridge_predict(train_x[train], y[train], x[test], alpha=1.0)
        rmse = float(math.sqrt(np.mean(np.square(pred - y[test]))))
        r = _pearson(np.asarray(pred, dtype=np.float32), y[test].astype(np.float32))
        rmses.append(rmse)
        if r is not None:
            rs.append(r)
        folds.append({"fold": fold_name, "test_count": int(len(test)), "rmse": rmse, "pearson": r})
    if not folds:
        return {"failure": "no non-empty P3 Ridge folds", "fold_strategy": fold_strategy}
    return {
        "fold_strategy": fold_strategy,
        "fold_count": int(len(folds)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "pearson_r_mean": None if not rs else float(np.mean(rs)),
        "pearson_r_std": None if not rs else float(np.std(rs)),
        "folds": folds,
    }


def _stratified_split_count(y: np.ndarray, n_splits: int) -> int:
    counts = [int(np.sum(y == value)) for value in set(y.tolist())]
    if not counts:
        return 0
    return min(int(n_splits), min(counts))


def _classification_probe_numpy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_x: np.ndarray | None = None,
    seed: int,
    n_splits: int,
    backend: str,
) -> dict[str, Any]:
    y = np.asarray(y).astype(str)
    split_count = _stratified_split_count(y, n_splits)
    if split_count < 2:
        return {"failure": "not enough samples per class", "class_count": int(len(set(y.tolist()))), "backend": backend}
    accs = []
    f1s = []
    folds = []
    fit_x = x if train_x is None else train_x
    for index, (train, test) in enumerate(_stratified_folds(y, split_count, seed=seed)):
        pred = _softmax_logreg_predict(fit_x[train], y[train], x[test], seed=seed + index)
        acc = float(np.mean(pred == y[test]))
        f1 = _macro_f1(y[test], pred)
        accs.append(acc)
        f1s.append(f1)
        folds.append({"fold": index, "test_count": int(len(test)), "accuracy": acc, "macro_f1": f1})
    return {
        "backend": backend,
        "class_count": int(len(set(y.tolist()))),
        "fold_count": int(split_count),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "folds": folds,
    }


def _softmax_logreg_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    epochs: int = 300,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
) -> np.ndarray:
    classes = np.asarray(sorted(set(train_y.tolist())), dtype=str)
    y_index = np.asarray([int(np.flatnonzero(classes == value)[0]) for value in train_y], dtype=np.int64)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (train_x - mean) / std
    x_test = (test_x - mean) / std
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(x_train.shape[1], len(classes))).astype(np.float32)
    bias = np.zeros(len(classes), dtype=np.float32)
    targets = np.eye(len(classes), dtype=np.float32)[y_index]
    for _ in range(epochs):
        logits = x_train @ weights + bias
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        error = (probs - targets) / max(1, len(x_train))
        weights -= learning_rate * (x_train.T @ error + l2 * weights)
        bias -= learning_rate * error.sum(axis=0)
    pred_index = np.argmax(x_test @ weights + bias, axis=1)
    return classes[pred_index]


def _stratified_folds(y: np.ndarray, n_splits: int, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    by_class = []
    for value in sorted(set(y.tolist())):
        indices = np.flatnonzero(y == value)
        rng.shuffle(indices)
        by_class.append(np.array_split(indices, n_splits))
    folds = []
    all_indices = np.arange(len(y), dtype=np.int64)
    for fold_index in range(n_splits):
        test = np.concatenate([parts[fold_index] for parts in by_class]).astype(np.int64)
        train = np.setdiff1d(all_indices, test, assume_unique=False).astype(np.int64)
        folds.append((train, test))
    return folds


def _macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for value in sorted(set(truth.tolist()) | set(pred.tolist())):
        tp = float(np.sum((truth == value) & (pred == value)))
        fp = float(np.sum((truth != value) & (pred == value)))
        fn = float(np.sum((truth == value) & (pred != value)))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _ridge_probe_numpy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_x: np.ndarray,
    seed: int,
    n_splits: int,
    backend: str,
) -> dict[str, Any]:
    split_count = max(2, min(int(n_splits), len(y)))
    rmses = []
    rs = []
    folds = []
    for index, (train, test) in enumerate(_kfold_indices(len(y), split_count, seed=seed)):
        pred = _ridge_predict(train_x[train], y[train], x[test], alpha=1.0)
        rmse = float(math.sqrt(np.mean(np.square(pred - y[test]))))
        r = _pearson(np.asarray(pred, dtype=np.float32), y[test].astype(np.float32))
        rmses.append(rmse)
        if r is not None:
            rs.append(r)
        folds.append({"fold": index, "test_count": int(len(test)), "rmse": rmse, "pearson": r})
    return {
        "backend": backend,
        "fold_strategy": "shuffled_k_fold",
        "fold_count": int(split_count),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "pearson_r_mean": None if not rs else float(np.mean(rs)),
        "pearson_r_std": None if not rs else float(np.std(rs)),
        "folds": folds,
    }


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, alpha: float) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (train_x - mean) / std
    x_test = (test_x - mean) / std
    x_aug = np.concatenate([x_train, np.ones((len(x_train), 1), dtype=np.float32)], axis=1)
    test_aug = np.concatenate([x_test, np.ones((len(x_test), 1), dtype=np.float32)], axis=1)
    regularizer = np.eye(x_aug.shape[1], dtype=np.float32) * float(alpha)
    regularizer[-1, -1] = 0.0
    weights = np.linalg.pinv(x_aug.T @ x_aug + regularizer) @ x_aug.T @ train_y
    return test_aug @ weights


def _kfold_indices(row_count: int, n_splits: int, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(row_count, dtype=np.int64)
    rng.shuffle(indices)
    parts = [part.astype(np.int64) for part in np.array_split(indices, n_splits)]
    folds = []
    for index, test in enumerate(parts):
        train = np.concatenate([part for part_index, part in enumerate(parts) if part_index != index]).astype(np.int64)
        folds.append((train, test))
    return folds


def _session_id(subject_id: str, event_id: str, sample_id: str) -> str:
    for value in (event_id, sample_id):
        match = re.search(r"(sub-[^_]+)_+(ses-[^_]+)", value)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
    return f"{subject_id}_unknown-session"


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _write_json(result: dict[str, Any], output: Path | str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| probe | primary metric | secondary metric | details |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, probe in result["probes"].items():
        if name.endswith("logreg"):
            primary = _format_metric(probe.get("accuracy_mean"))
            secondary = _format_metric(probe.get("macro_f1_mean"))
            details = f"classes={probe.get('class_count', 'NA')} subjects={probe.get('subject_count', 'NA')}"
        else:
            primary = _format_metric(probe.get("rmse_mean"))
            secondary = _format_metric(probe.get("pearson_r_mean"))
            details = f"folds={probe.get('fold_count', 'NA')}"
        rows.append(f"| {name} | {primary} | {secondary} | {details} |")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
