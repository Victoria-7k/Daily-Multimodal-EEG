from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionRecords:
    sample_id: np.ndarray
    event_id: np.ndarray
    subject_id: np.ndarray
    session_id: np.ndarray
    fold_id: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    attention: np.ndarray | None
    model_name: str
    experiment: str
    protocol: str


def regression_metrics(prediction: Sequence[float], target: Sequence[float]) -> dict:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    err = pred - truth
    return {
        "count": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": _pearson(pred, truth),
    }


def aggregate_event_predictions(records: PredictionRecords, *, tolerance: float = 1e-8) -> PredictionRecords:
    groups: dict[tuple[str, str, str], list[int]] = {}
    order: list[tuple[str, str, str]] = []
    for index, key in enumerate(_event_keys(records)):
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)

    sample_id = []
    event_id = []
    subject_id = []
    session_id = []
    fold_id = []
    target = []
    prediction = []
    attention_rows = [] if records.attention is not None else None
    for key in order:
        indices = np.asarray(groups[key], dtype=np.int64)
        values = records.target[indices].astype(np.float32)
        if float(np.max(values) - np.min(values)) > float(tolerance):
            raise ValueError(f"inconsistent target for event key {key}")
        subject, session, event = key
        sample_id.append("|".join(key))
        subject_id.append(subject)
        session_id.append(session)
        event_id.append(event)
        fold_id.append(records.fold_id[indices[0]])
        target.append(float(values[0]))
        prediction.append(float(np.mean(records.prediction[indices].astype(np.float32))))
        if attention_rows is not None:
            attention_rows.append(np.mean(records.attention[indices], axis=0))

    return PredictionRecords(
        sample_id=np.asarray(sample_id, dtype=str),
        event_id=np.asarray(event_id, dtype=str),
        subject_id=np.asarray(subject_id, dtype=str),
        session_id=np.asarray(session_id, dtype=str),
        fold_id=np.asarray(fold_id),
        target=np.asarray(target, dtype=np.float32),
        prediction=np.asarray(prediction, dtype=np.float32),
        attention=None if attention_rows is None else np.asarray(attention_rows, dtype=np.float32),
        model_name=records.model_name,
        experiment=records.experiment,
        protocol=records.protocol,
    )


def summarize_subject_oof(
    records: PredictionRecords,
    expected_sample_ids: Sequence[str] | np.ndarray,
) -> dict:
    actual = records.sample_id.astype(str)
    expected = np.asarray(expected_sample_ids).astype(str)
    actual_list = actual.tolist()
    expected_list = expected.tolist()
    if len(actual_list) != len(set(actual_list)):
        raise ValueError("duplicate OOF sample")
    if len(actual_list) != len(expected_list):
        raise ValueError("expected OOF sample coverage mismatch")
    actual_set = set(actual_list)
    expected_set = set(expected_list)
    if actual_set != expected_set:
        unexpected = sorted(actual_set - expected_set)
        if unexpected:
            raise ValueError(f"unexpected OOF sample: {unexpected[:5]}")
        raise ValueError("expected OOF sample coverage mismatch")
    event_records = aggregate_event_predictions(records)
    return {
        "expected_sample_count": int(len(expected_list)),
        "actual_sample_count": int(len(actual_list)),
        "expected_sample_id_sha256": _sha256_lines(expected_list),
        "actual_sample_id_sha256": _sha256_lines(actual_list),
        "oof_complete": True,
        "window": regression_metrics(records.prediction, records.target),
        "event": regression_metrics(event_records.prediction, event_records.target),
    }


def summarize_pooled_oof(records: PredictionRecords) -> dict:
    raw = regression_metrics(records.prediction, records.target)
    centered_prediction = records.prediction.astype(np.float32).copy()
    centered_target = records.target.astype(np.float32).copy()
    for subject in _ordered_unique(records.subject_id.astype(str).tolist()):
        mask = records.subject_id.astype(str) == subject
        centered_prediction[mask] -= float(np.mean(centered_prediction[mask]))
        centered_target[mask] -= float(np.mean(centered_target[mask]))
    return {
        "rmse": raw["rmse"],
        "mae": raw["mae"],
        "raw_pearson": raw["pearson"],
        "within_subject_centered_pearson": _pearson(centered_prediction, centered_target),
        "pearson_definition": "center prediction and target by subject OOF mean",
    }


def audit_fold_target_variance(
    target: Sequence[float] | np.ndarray,
    *,
    train: Sequence[int] | np.ndarray,
    val: Sequence[int] | np.ndarray,
    test: Sequence[int] | np.ndarray,
    tolerance: float = 1e-8,
) -> dict:
    values = np.asarray(target, dtype=np.float32)
    result = {}
    for name, indices in {
        "train": np.asarray(train, dtype=np.int64),
        "val": np.asarray(val, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
    }.items():
        subset = values[indices]
        result[f"{name}_unique_target_count"] = int(len(np.unique(subset)))
        result[f"{name}_target_std"] = float(np.std(subset))
    if result["train_target_std"] <= float(tolerance):
        raise ValueError("degenerate_train_target")
    return result


def fit_predict_train_mean(
    target: Sequence[float] | np.ndarray,
    train: Sequence[int] | np.ndarray,
    test: Sequence[int] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(target, dtype=np.float32)
    train_idx = np.asarray(train, dtype=np.int64)
    test_idx = np.asarray(test, dtype=np.int64)
    return np.full(len(test_idx), float(np.mean(values[train_idx])), dtype=np.float32)


def fit_predict_concat_ridge(
    tokens: np.ndarray,
    mask: np.ndarray,
    target: Sequence[float] | np.ndarray,
    train: Sequence[int] | np.ndarray,
    test: Sequence[int] | np.ndarray,
    *,
    alpha: float = 10.0,
) -> tuple[np.ndarray, dict]:
    features = _concat_features(tokens, mask)
    train_idx = np.asarray(train, dtype=np.int64)
    test_idx = np.asarray(test, dtype=np.int64)
    train_x = features[train_idx]
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std >= 1e-6, std, 1.0)
    x_train = (train_x - mean) / std
    x_test = (features[test_idx] - mean) / std
    y_train = np.asarray(target, dtype=np.float32)[train_idx]
    y_mean = float(y_train.mean())
    y_centered = y_train - y_mean
    gram = x_train.T @ x_train
    coef = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float32),
        x_train.T @ y_centered,
    )
    prediction = (x_test @ coef + y_mean).astype(np.float32)
    return prediction, {
        "normalization_fit_scope": "train_only",
        "fit_sample_count": int(len(train_idx)),
        "alpha": float(alpha),
        "x_mean_sha256": _sha256_arrays(mean.astype(np.float32)),
        "x_std_sha256": _sha256_arrays(std.astype(np.float32)),
    }


def save_prediction_shard(path: Path | str, records: PredictionRecords, metadata: dict) -> dict:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": records.sample_id.astype(str),
        "event_id": records.event_id.astype(str),
        "subject_id": records.subject_id.astype(str),
        "session_id": records.session_id.astype(str),
        "fold_id": records.fold_id,
        "target": records.target.astype(np.float32),
        "prediction": records.prediction.astype(np.float32),
    }
    if records.attention is not None:
        payload["attention"] = records.attention.astype(np.float32)
    np.savez_compressed(out, **payload)
    sidecar = {
        **metadata,
        "schema_version": 1,
        "npz_sha256": _sha256_file(out),
    }
    out.with_suffix(out.suffix + ".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar


def _concat_features(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    token_values = np.asarray(tokens, dtype=np.float32)
    token_mask = np.asarray(mask, dtype=bool)
    zeroed = np.where(token_mask[:, :, None], token_values, 0.0)
    return np.concatenate(
        [zeroed.reshape(len(zeroed), -1), token_mask.astype(np.float32)],
        axis=1,
    ).astype(np.float32)


def _event_keys(records: PredictionRecords) -> list[tuple[str, str, str]]:
    return [
        (str(subject), str(session), str(event))
        for subject, session, event in zip(
            records.subject_id.tolist(),
            records.session_id.tolist(),
            records.event_id.tolist(),
        )
    ]


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _ordered_unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _sha256_lines(values: Sequence[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
