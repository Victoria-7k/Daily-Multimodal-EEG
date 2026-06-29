from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EMBEDDING_DIM = 256
REAL_MODALITY_ORDER = ("eeg", "wear", "face", "audio")


@dataclass
class RealEmbeddingResult:
    sample_id: str
    event_id: str
    subject_id: str
    modality: str
    embedding: np.ndarray
    mask_value: int | bool
    quality_flags: dict[str, Any]
    encoder_version: str
    source_paths: dict[str, Any]

    def __post_init__(self) -> None:
        if self.modality not in REAL_MODALITY_ORDER:
            raise ValueError(f"modality must be one of {REAL_MODALITY_ORDER}: {self.modality!r}")
        self.embedding = validate_embedding_shape(f"{self.modality}_emb", self.embedding)
        if isinstance(self.mask_value, bool):
            self.mask_value = int(self.mask_value)
        if self.mask_value not in (0, 1):
            raise ValueError(f"mask_value must be 0 or 1 for {self.sample_id}: {self.mask_value!r}")
        if not self.encoder_version:
            raise ValueError(f"encoder_version is required for {self.sample_id}")


def validate_embedding_shape(
    name: str,
    array: np.ndarray,
    *,
    expected_dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """Validate a single `[256]` vector or `[N, 256]` batch and return float32."""
    value = np.asarray(array)
    valid_shape = value.shape == (expected_dim,) or (
        len(value.shape) == 2 and value.shape[1] == expected_dim
    )
    if not valid_shape:
        raise ValueError(
            f"{name} expected shape ({expected_dim},) or (N, {expected_dim}), got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype, got {value.dtype}")
    if np.isnan(value).any():
        raise ValueError(f"{name} contains NaN values")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains infinite values")
    return value.astype(np.float32, copy=False)
