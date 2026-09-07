"""Compatibility helpers for reading NPZ files across NumPy major versions."""

from __future__ import annotations

import sys

import numpy as np


def install_numpy_core_pickle_aliases() -> None:
    """Allow NumPy 1.x to unpickle object arrays written by NumPy 2.x.

    NumPy 2 writes object-array pickle references under ``numpy._core``.
    The project runtime used for some GPU jobs still ships NumPy 1.x, where
    the public import path is ``numpy.core``. Numeric arrays are unaffected;
    this only keeps metadata arrays such as ``sample_id`` readable.
    """

    if "numpy._core" in sys.modules:
        return
    try:
        import numpy.core as core
        import numpy.core.multiarray as multiarray
        import numpy.core.numeric as numeric
    except Exception:
        return
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core.numeric", numeric)

