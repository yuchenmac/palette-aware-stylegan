"""Explicit RGB array ranges used by patch construction."""

from __future__ import annotations

import numpy as np


def normalise_rgb_array(image: np.ndarray) -> np.ndarray:
    """Return float32 RGB in [0, 1] from uint8, uint16, or normalised floats.

    Integer arrays use their full dtype range. Floating-point arrays must already
    contain finite values in [0, 1].
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
        raise ValueError("Expected a nonempty H x W x 3 RGB array.")
    if array.dtype.kind == "u" and array.dtype.itemsize in (1, 2):
        maximum = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32) / maximum
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Floating-point RGB values must be finite.")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("Floating-point RGB values must already be in [0, 1].")
        return array.astype(np.float32)
    raise TypeError("RGB arrays must use uint8, uint16, or floating-point values in [0, 1].")
