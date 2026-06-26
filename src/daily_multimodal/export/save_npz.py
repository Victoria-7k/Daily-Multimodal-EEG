from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz(path: Path | str, **arrays: Any) -> Path:
    """Save compressed NPZ arrays and create parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out
