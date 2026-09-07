from __future__ import annotations

from typing import Any

import numpy as np


DIAGNOSTIC_KEYS = (
    "modality_weights",
    "probe_ordinal_logits",
    "probe_class_logits",
    "probe_probs",
    "probe_entropy",
    "probe_ordinal_var",
    "modality_difficulty",
    "valid_modality_mask",
    "temporal_weights",
    "kernel_mixture",
    "states",
)


def stack_diagnostics(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        return {}
    result: dict[str, np.ndarray] = {}
    for key in DIAGNOSTIC_KEYS:
        values = [chunk[key] for chunk in chunks if key in chunk]
        if values:
            result[key] = np.concatenate(values, axis=0)
    return result


def summarize_mask(mask: Any) -> dict[str, Any]:
    values = np.asarray(mask).astype(bool)
    return {
        "shape": list(values.shape),
        "valid_count": int(values.sum()),
        "coverage": float(values.mean()) if values.size else 0.0,
        "modality_valid": values.sum(axis=(0, 1)).astype(int).tolist() if values.ndim == 3 else [],
    }
