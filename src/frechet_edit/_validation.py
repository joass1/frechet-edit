"""Boundary validation for the public API.

Rules are normative in ``docs/definition.md`` section 6. In particular the
validator never sorts, deduplicates, reverses or otherwise mutates a curve, and
it never rewrites ``delta``.
"""

from __future__ import annotations

import math

import numpy as np

from ._numerics import NumericPolicy
from ._types import BACKENDS, OPERATIONS, Operations, UnsupportedDimensionError

#: Dimensions for which each mode has a certified backend.
#: Deletion needs point distances only; insertion needs enclosing balls.
#: Insertion-capable modes work in any dimension up to this bound. The limit is
#: the cost of the exact enclosing-ball kernel, whose per-round work grows like
#: 2**(d+2) candidate subsets each solving a d x d rational system, so it stops
#: being practical well before it stops being correct. Raised from the former
#: plane-only restriction once `_meb` supplied a general-dimension exact ball.
MAX_INSERTION_DIMENSION: int = 8

NUMERIC_POLICIES: tuple[str, ...] = ("certified", "fast")


def as_curve(array: object, name: str) -> np.ndarray:
    """Coerce input to a ``(k, d)`` float64 array, rejecting anything doubtful.

    A one-dimensional input of length ``k`` is read as ``k`` points in 1D, which
    is the only implicit reshape allowed and is documented in the API.
    """
    try:
        arr = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array of points: {exc}") from exc

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a (k, d) array of points or a 1-D array of scalars, "
            f"got shape {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError(
            f"{name} must be non-empty; the empty-prefix state exists only inside "
            "the dynamic program, never as public input"
        )
    if arr.shape[1] == 0:
        raise ValueError(f"{name} has zero-width points (shape {arr.shape})")
    if not np.all(np.isfinite(arr)):
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(f"{name} contains {bad} non-finite coordinate(s) (NaN or inf)")
    # A copy, so that no caller array is ever mutated and no view aliases theirs.
    return np.ascontiguousarray(arr, dtype=np.float64)


def check_delta(delta: object) -> float:
    """``delta`` must be a finite, strictly positive float."""
    try:
        value = float(delta)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"delta must be a real number, got {delta!r}") from exc
    if math.isnan(value):
        raise ValueError("delta must not be NaN")
    if math.isinf(value):
        raise ValueError("delta must be finite")
    if value <= 0.0:
        raise ValueError(
            f"delta must be strictly positive, got {value!r}; zero-threshold support "
            "is not part of this release"
        )
    return value


def check_operations(operations: object) -> Operations:
    if operations not in OPERATIONS:
        raise ValueError(f"operations must be one of {OPERATIONS}, got {operations!r}")
    return operations


def check_backend(backend: object) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    return backend


def check_numeric_policy(policy: object) -> NumericPolicy:
    if policy not in NUMERIC_POLICIES:
        raise ValueError(
            f"numeric_policy must be one of {NUMERIC_POLICIES}, got {policy!r}"
        )
    return policy  # type: ignore[return-value]


def check_pair(reference: np.ndarray, observation: np.ndarray) -> int:
    """Both curves must live in the same dimension. Returns that dimension."""
    if reference.shape[1] != observation.shape[1]:
        raise ValueError(
            f"reference and observation dimensions differ: {reference.shape[1]} "
            f"vs {observation.shape[1]}"
        )
    return int(reference.shape[1])


def check_mode_dimension(mode: Operations, dimension: int) -> None:
    """Insertion-capable modes are certified up to MAX_INSERTION_DIMENSION.

    This is a scope restriction of the enclosing-ball backend shipped here, not
    a limitation of the underlying theorem, and it is reported as its own
    exception type rather than silently degraded or reported as infeasible.
    """
    if mode == "delete":
        return
    if dimension > MAX_INSERTION_DIMENSION:
        raise UnsupportedDimensionError(
            f"operations={mode!r} is certified for dimensions 1 to "
            f"{MAX_INSERTION_DIMENSION}, got {dimension}. The restriction is the "
            "cost of this package's exact minimum-enclosing-ball backend, not a "
            "limitation of the underlying result."
        )
