from __future__ import annotations

from typing import Any

import numpy as np


def fit_temperature_grid(
    logits: Any,
    labels_zero_based: Any,
    *,
    candidates: tuple[float, ...] | None = None,
) -> dict[str, float]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels_zero_based, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != labels.size:
        raise ValueError("logits must be (N,C) and labels must be (N,)")
    if candidates is None:
        candidates = tuple(float(v) for v in np.geomspace(0.5, 5.0, 31))
    best_temp = 1.0
    best_nll = float("inf")
    for temperature in candidates:
        probs = softmax(values / max(float(temperature), 1e-6))
        nll = _nll(probs, labels)
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temperature)
    return {"temperature": best_temp, "val_nll": float(best_nll)}


def apply_temperature(logits: Any, temperature: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)).astype(np.float32)


def fit_probe_temperatures_grid(
    logits: Any,
    labels_zero_based: Any,
    valid_modality_mask: Any,
    *,
    candidates: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Fit one positive temperature per modality for cumulative Probe logits."""

    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels_zero_based, dtype=np.int64).reshape(-1)
    valid = np.asarray(valid_modality_mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("probe logits must have shape (N, M, 4)")
    if valid.shape != values.shape[:2] or values.shape[0] != labels.size:
        raise ValueError("probe logits, labels, and valid modality mask must agree")
    if candidates is None:
        candidates = tuple(float(value) for value in np.geomspace(0.5, 5.0, 31))
    temperatures: list[float] = []
    modality_nll: list[float | None] = []
    modality_counts: list[int] = []
    for modality in range(values.shape[1]):
        rows = valid[:, modality]
        modality_counts.append(int(rows.sum()))
        if not bool(rows.any()):
            temperatures.append(1.0)
            modality_nll.append(None)
            continue
        best_temperature = 1.0
        best_nll = float("inf")
        for temperature in candidates:
            probabilities = cumulative_logits_to_probs(values[rows, modality] / max(float(temperature), 1e-6))
            nll = _nll(probabilities, labels[rows])
            if nll < best_nll:
                best_nll = nll
                best_temperature = float(temperature)
        temperatures.append(best_temperature)
        modality_nll.append(float(best_nll))
    return {
        "temperatures": temperatures,
        "val_nll_by_modality": modality_nll,
        "val_count_by_modality": modality_counts,
        "candidate_count": len(candidates),
    }


def softmax(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=1, keepdims=True).clip(min=1e-12)


def cumulative_logits_to_probs(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    cumulative = 1.0 / (1.0 + np.exp(-values))
    cumulative = np.minimum.accumulate(cumulative, axis=-1)
    probabilities = np.concatenate(
        [
            1.0 - cumulative[..., :1],
            cumulative[..., :1] - cumulative[..., 1:2],
            cumulative[..., 1:2] - cumulative[..., 2:3],
            cumulative[..., 2:3] - cumulative[..., 3:4],
            cumulative[..., 3:4],
        ],
        axis=-1,
    )
    probabilities = np.clip(probabilities, 0.0, None)
    return probabilities / np.sum(probabilities, axis=-1, keepdims=True).clip(min=1e-12)


def _nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("inf")
    picked = probabilities[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, 1.0))))
