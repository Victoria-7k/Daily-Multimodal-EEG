from __future__ import annotations

import numpy as np


def concatenate_modalities(
    eeg_emb: np.ndarray,
    wear_emb: np.ndarray,
    face_emb: np.ndarray,
    audio_emb: np.ndarray,
) -> np.ndarray:
    """Concatenate canonical 256-dim modality embeddings into a 1024-dim vector."""
    return np.concatenate([eeg_emb, wear_emb, face_emb, audio_emb], axis=-1)
