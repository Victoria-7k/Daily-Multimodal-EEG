from __future__ import annotations

import numpy as np


def shuffled_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(values)
    size = max(1, int(batch_size))
    return [values[start : start + size] for start in range(0, len(values), size)]
