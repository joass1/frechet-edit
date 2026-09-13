"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import numpy as np
import pytest


def curve(values) -> np.ndarray:
    """Build a float64 ``(k, d)`` curve from a nested or flat sequence."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr)


@pytest.fixture
def c():
    return curve


ALL_MODES = ("delete", "insert", "both")
ALL_BACKENDS = ("python", "reference")
