"""Regression metrics that separate subject-level and within-subject signal."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def safe_pearsonr(y_true: Any, y_pred: Any) -> float | None:
    """Return Pearson correlation, or ``None`` for an undefined correlation."""

    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size < 2 or pred.size != true.size:
        return None
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        return None
    true = true - true.mean()
    pred = pred - pred.mean()
    denominator = float(np.sqrt(np.sum(true * true) * np.sum(pred * pred)))
    if denominator <= 0.0:
        return None
    return float(np.sum(true * pred) / denominator)


def within_subject_centered_arrays(
    y_true: Any,
    y_pred: Any,
    subject_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Center truth and prediction by the mean of each subject.

    The returned tuple is ``(y_true_centered, y_pred_centered)`` and preserves
    the input order.  Subject means are computed only from the supplied rows;
    callers should therefore pass the split being evaluated (for example, the
    test split), matching the EEGPT report definition.
    """

    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    subjects = np.asarray(subject_ids).reshape(-1)
    if true.size != pred.size or true.size != subjects.size:
        raise ValueError("y_true, y_pred, and subject_ids must have equal length")
    true_centered = true.copy()
    pred_centered = pred.copy()
    for subject in np.unique(subjects):
        mask = subjects == subject
        true_centered[mask] -= true[mask].mean()
        pred_centered[mask] -= pred[mask].mean()
    return true_centered.astype(np.float32), pred_centered.astype(np.float32)


def _per_subject_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, Any]:
    values: list[float] = []
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subject_ids).tolist(), key=str):
        mask = subject_ids == subject
        r = safe_pearsonr(y_true[mask], y_pred[mask])
        rows.append({"subject_id": str(subject), "count": int(mask.sum()), "pearson_r": r})
        if r is not None:
            values.append(float(r))
    return {
        "mean": None if not values else float(np.mean(values)),
        "std": None if not values else float(np.std(values)),
        "subject_count": int(len(rows)),
        "valid_subject_r_count": int(len(values)),
        "subjects": rows,
    }


def evaluate_regression_with_centered(
    y_true: Any,
    y_pred: Any,
    subject_ids: Any,
) -> dict[str, Any]:
    """Compute RMSE/MAE, raw r, centered r, and per-subject r diagnostics."""

    true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    subjects = np.asarray(subject_ids).reshape(-1)
    if true.size != pred.size or true.size != subjects.size:
        raise ValueError("y_true, y_pred, and subject_ids must have equal length")
    error = pred - true
    centered_true, centered_pred = within_subject_centered_arrays(true, pred, subjects)
    return {
        "count": int(true.size),
        "rmse": float(np.sqrt(np.mean(error * error))) if true.size else None,
        "mae": float(np.mean(np.abs(error))) if true.size else None,
        "raw_r": safe_pearsonr(true, pred),
        "within_subject_centered_r": safe_pearsonr(centered_true, centered_pred),
        "per_subject_r": _per_subject_pearson(true, pred, subjects),
    }


def predict_subject_train_mean(
    train_y: Any,
    train_subjects: Any,
    test_subjects: Any,
) -> np.ndarray:
    """Predict each test row with its train-only subject mean.

    Unseen test subjects use the global train mean.  The function deliberately
    does not inspect test labels, so it is safe for split-level baselines.
    """

    values = np.asarray(train_y, dtype=np.float32).reshape(-1)
    subjects = np.asarray(train_subjects).reshape(-1)
    test = np.asarray(test_subjects).reshape(-1)
    if values.size != subjects.size:
        raise ValueError("train_y and train_subjects must have equal length")
    if values.size == 0:
        raise ValueError("train_y must contain at least one value")
    global_mean = float(values.mean())
    grouped: dict[Any, list[float]] = defaultdict(list)
    for value, subject in zip(values.tolist(), subjects.tolist()):
        grouped[subject].append(float(value))
    means = {subject: float(np.mean(rows)) for subject, rows in grouped.items()}
    return np.asarray([means.get(subject, global_mean) for subject in test.tolist()], dtype=np.float32)
