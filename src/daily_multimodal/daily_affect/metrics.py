from __future__ import annotations

from typing import Any

import numpy as np

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered, safe_pearsonr


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def confusion_matrix(y_true: Any, y_pred: Any, *, num_classes: int = 5) -> np.ndarray:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for expected, observed in zip(true.tolist(), pred.tolist()):
        if 0 <= expected < num_classes and 0 <= observed < num_classes:
            matrix[expected, observed] += 1
    return matrix


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    values: list[float] = []
    for label in range(matrix.shape[0]):
        tp = float(matrix[label, label])
        fp = float(matrix[:, label].sum() - tp)
        fn = float(matrix[label, :].sum() - tp)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        values.append(safe_divide(2.0 * precision * recall, precision + recall))
    return float(np.mean(values)) if values else 0.0


def per_class_metrics_from_confusion(matrix: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for label in range(matrix.shape[0]):
        tp = float(matrix[label, label])
        fp = float(matrix[:, label].sum() - tp)
        fn = float(matrix[label, :].sum() - tp)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        rows.append(
            {
                "label": int(label + 1),
                "support": int(matrix[label, :].sum()),
                "precision": precision,
                "recall": recall,
                "f1": safe_divide(2.0 * precision * recall, precision + recall),
            }
        )
    return rows


def quadratic_weighted_kappa(y_true: Any, y_pred: Any, *, num_classes: int = 5) -> float | None:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.size == 0 or true.size != pred.size:
        return None
    observed = confusion_matrix(true, pred, num_classes=num_classes).astype(np.float64)
    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(1.0, float(true.size))
    labels = np.arange(num_classes, dtype=np.float64)
    weights = ((labels[:, None] - labels[None, :]) ** 2) / float((num_classes - 1) ** 2)
    observed_score = float(np.sum(weights * observed))
    expected_score = float(np.sum(weights * expected))
    if expected_score <= 0:
        return None
    return float(1.0 - observed_score / expected_score)


def classification_metrics(
    y_true_zero_based: Any,
    y_pred_zero_based: Any,
    *,
    expected_score: Any | None = None,
    subject_ids: Any | None = None,
    probabilities: Any | None = None,
    num_classes: int = 5,
) -> dict[str, Any]:
    true = np.asarray(y_true_zero_based, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred_zero_based, dtype=np.int64).reshape(-1)
    if true.size != pred.size:
        raise ValueError("truth and prediction must have equal length")
    matrix = confusion_matrix(true, pred, num_classes=num_classes)
    error = pred.astype(np.float64) - true.astype(np.float64)
    per_class = per_class_metrics_from_confusion(matrix)
    recalls = [float(row["recall"]) for row in per_class]
    metrics: dict[str, Any] = {
        "count": int(true.size),
        "accuracy": float(np.mean(pred == true)) if true.size else None,
        "macro_f1": macro_f1_from_confusion(matrix) if true.size else None,
        "qwk": quadratic_weighted_kappa(true, pred, num_classes=num_classes),
        "ordinal_mae": float(np.mean(np.abs(error))) if true.size else None,
        "ordinal_rmse": float(np.sqrt(np.mean(error * error))) if true.size else None,
        "balanced_accuracy": float(np.mean(recalls)) if true.size else None,
        "adjacent_error_rate": float(np.mean(np.abs(error) == 1.0)) if true.size else None,
        "severe_error_rate": float(np.mean(np.abs(error) >= 2.0)) if true.size else None,
        "error_distance_counts": {
            str(distance): int(np.sum(np.abs(error) == distance))
            for distance in range(num_classes)
        },
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    if expected_score is not None:
        score = np.asarray(expected_score, dtype=np.float64).reshape(-1)
        if score.size != true.size:
            raise ValueError("expected_score and truth must have equal length")
        raw_true = true.astype(np.float64) + 1.0
        metrics.update(regression_bridge_metrics(raw_true, score, subject_ids=subject_ids))
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (true.size, num_classes):
            raise ValueError(f"probabilities must have shape {(true.size, num_classes)}, got {probs.shape}")
        clipped = np.clip(probs[np.arange(true.size), true], 1e-12, 1.0) if true.size else np.asarray([], dtype=np.float64)
        metrics["nll"] = float(-np.mean(np.log(clipped))) if clipped.size else None
        one_hot = np.eye(num_classes, dtype=np.float64)[true] if true.size else np.zeros_like(probs)
        metrics["brier"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))) if true.size else None
        cumulative_probs = np.cumsum(probs, axis=1)[:, :-1]
        cumulative_target = (true[:, None] <= np.arange(num_classes - 1)[None, :]).astype(np.float64)
        metrics["rps"] = float(np.mean(np.sum((cumulative_probs - cumulative_target) ** 2, axis=1) / (num_classes - 1))) if true.size else None
        metrics["ece"] = expected_calibration_error(probs, true)
    if subject_ids is not None:
        metrics.update(within_subject_classification_metrics(true, pred, subject_ids, num_classes=num_classes))
    return metrics


def expected_calibration_error(probabilities: np.ndarray, labels_zero_based: np.ndarray, *, bins: int = 10) -> float | None:
    if probabilities.size == 0:
        return None
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == labels_zero_based).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(len(labels_zero_based))
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        select = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if not bool(select.any()):
            continue
        value += float(select.sum()) / total * abs(float(correct[select].mean()) - float(confidence[select].mean()))
    return float(value)


def within_subject_classification_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    subject_ids: Any,
    *,
    num_classes: int,
) -> dict[str, float | int | None]:
    subjects = np.asarray(subject_ids).astype(str).reshape(-1)
    if subjects.size != true.size:
        raise ValueError("subject_ids and labels must have equal length")
    macro_values: list[float] = []
    qwk_values: list[float] = []
    for subject in np.unique(subjects):
        select = subjects == subject
        matrix = confusion_matrix(true[select], pred[select], num_classes=num_classes)
        macro_values.append(macro_f1_from_confusion(matrix))
        qwk = quadratic_weighted_kappa(true[select], pred[select], num_classes=num_classes)
        if qwk is not None:
            qwk_values.append(float(qwk))
    return {
        "within_subject_macro_f1_mean": float(np.mean(macro_values)) if macro_values else None,
        "within_subject_macro_f1_std": float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0 if macro_values else None,
        "within_subject_qwk_mean": float(np.mean(qwk_values)) if qwk_values else None,
        "within_subject_qwk_std": float(np.std(qwk_values, ddof=1)) if len(qwk_values) > 1 else 0.0 if qwk_values else None,
        "within_subject_metric_subject_count": int(len(np.unique(subjects))),
    }


def regression_bridge_metrics(
    y_true_raw: Any,
    expected_score: Any,
    *,
    subject_ids: Any | None = None,
) -> dict[str, Any]:
    """Score ordinal expected values on the shared 1--5 regression scale.

    Daily-affect remains an EMA-bag ordinal task.  These metrics only expose
    its expected ordinal score through the same RMSE/raw-r/centered-r lens used
    by the window-level fatigue route; they do not change its optimization or
    promotion gate.
    """

    true = np.asarray(y_true_raw, dtype=np.float64).reshape(-1)
    score = np.asarray(expected_score, dtype=np.float64).reshape(-1)
    if true.size != score.size:
        raise ValueError("y_true_raw and expected_score must have equal length")
    if subject_ids is None:
        error = score - true
        return {
            "expected_mae": float(np.mean(np.abs(error))) if true.size else None,
            "expected_rmse": float(np.sqrt(np.mean(error * error))) if true.size else None,
            "expected_raw_r": safe_pearsonr(true, score),
            "expected_within_subject_centered_r": None,
        }
    evaluated = evaluate_regression_with_centered(true, score, subject_ids)
    return {
        "expected_mae": evaluated["mae"],
        "expected_rmse": evaluated["rmse"],
        "expected_raw_r": evaluated["raw_r"],
        "expected_within_subject_centered_r": evaluated["within_subject_centered_r"],
    }


def metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result["selection_qwk"] = result.get("qwk")
    result["selection_macro_f1"] = result.get("macro_f1")
    return result
