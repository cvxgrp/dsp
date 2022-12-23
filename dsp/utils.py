from __future__ import annotations

import numpy as np


def np_vec(x: float | np.ndarray, order: str = "F") -> np.ndarray:
    """
    Convert a **scalar** or ndarray to a 1D array.
    """
    return np.atleast_1d(x).flatten(order=order)
