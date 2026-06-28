from __future__ import annotations

import numpy as np


MODALITY_ORDER = ("eeg", "wear", "face", "audio")


def modality_mask(*, eeg: bool, wear: bool, face: bool, audio: bool) -> np.ndarray:
    """Return mask in the canonical [eeg, wear, face, audio] order."""
    return np.array([eeg, wear, face, audio], dtype=np.int8)
