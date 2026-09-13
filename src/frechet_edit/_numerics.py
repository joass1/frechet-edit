"""Certified numerical predicates.

Implements the three-tier decision procedure of ``docs/numerics.md``:

1. float64 with a rigorous forward error bound,
2. exact ``fractions.Fraction`` arithmetic for boundary cases,
3. explicit abstention (``NumericallyAmbiguous``) when an exact decision would
   exceed the configured budget.

``delta`` is never modified to make a predicate return a particular answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Final, Literal, Union

import numpy as np

NumericPolicy = Literal["certified", "fast"]

#: Anything indexable that yields float-convertible coordinates.
PointLike = Union[Sequence[float], "np.ndarray"]

#: Unit roundoff for IEEE-754 binary64.
U: Final[float] = 2.0**-53

#: Multiplier applied to the derived error bound. Larger means the exact tier is
#: entered more often, never that a wrong answer is accepted.
SAFETY: Final[float] = 4.0

#: Maximum block size for which an exact minimum-enclosing-ball decision is
#: attempted. Above this the call abstains rather than guesses.
EXACT_BALL_MAX_POINTS: Final[int] = 48


class NumericallyAmbiguous(Exception):
    """Raised when the numerical policy is exhausted for a single predicate.

    The API converts this into ``status="numerically_ambiguous"``. It is never
    converted into infinity or into a feasible answer.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class PredicateStats:
    """Mutable counters describing how a computation was decided."""

    __slots__ = ("ambiguous", "ball_calls", "exact_fallbacks", "float_decisions")

    def __init__(self) -> None:
        self.float_decisions = 0
        self.exact_fallbacks = 0
        self.ball_calls = 0
        self.ambiguous = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "float_decisions": self.float_decisions,
            "exact_fallbacks": self.exact_fallbacks,
            "enclosing_ball_calls": self.ball_calls,
            "ambiguous_predicates": self.ambiguous,
        }


def _gamma(d: int) -> float:
    """Relative forward error bound for the computed squared distance in R^d."""
    k = (d + 3) * U
    return k / (1.0 - k)


def frac_sq_dist(a: PointLike, b: PointLike) -> Fraction:
    """Exact squared Euclidean distance between two float64 points."""
    total = Fraction(0)
    for x, y in zip(a, b, strict=True):
        diff = Fraction(float(x)) - Fraction(float(y))
        total += diff * diff
    return total


def exact_within(a: PointLike, b: PointLike, delta: float) -> bool:
    """Exact ``||a - b|| <= delta``. float64 values are exact binary rationals."""
    d2 = Fraction(float(delta))
    return frac_sq_dist(a, b) <= d2 * d2


def within(
    a: np.ndarray,
    b: np.ndarray,
    delta: float,
    *,
    policy: NumericPolicy = "certified",
    stats: PredicateStats | None = None,
) -> bool:
    """Decide ``||a - b|| <= delta`` with the tiered policy.

    The comparison is closed: equality counts as within.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d2 = float(np.dot(diff, diff))
    dim = diff.shape[0]
    delta2 = float(delta) * float(delta)

    g = SAFETY * _gamma(dim)
    tol = SAFETY * U
    if d2 * (1.0 + g) < delta2 * (1.0 - tol):
        if stats is not None:
            stats.float_decisions += 1
        return True
    if d2 * (1.0 - g) > delta2 * (1.0 + tol):
        if stats is not None:
            stats.float_decisions += 1
        return False

    if policy == "fast":
        if stats is not None:
            stats.float_decisions += 1
        return d2 <= delta2

    if stats is not None:
        stats.exact_fallbacks += 1
    return exact_within(a, b, delta)


def within_row(
    point: np.ndarray,
    curve: np.ndarray,
    delta: float,
    *,
    policy: NumericPolicy = "certified",
    stats: PredicateStats | None = None,
) -> np.ndarray:
    """Vectorised ``within`` of one point against every vertex of ``curve``.

    Returns a boolean array of length ``len(curve)``. Entries whose float
    decision is not certain are re-decided one at a time by the tiered policy,
    so the result is exactly what element-wise :func:`within` would produce.
    """
    diff = curve - point
    d2 = np.einsum("ij,ij->i", diff, diff)
    dim = curve.shape[1]
    delta2 = float(delta) * float(delta)

    g = SAFETY * _gamma(dim)
    tol = SAFETY * U
    certain_true = d2 * (1.0 + g) < delta2 * (1.0 - tol)
    certain_false = d2 * (1.0 - g) > delta2 * (1.0 + tol)
    out = certain_true.copy()

    boundary = ~(certain_true | certain_false)
    n_certain = int(certain_true.sum() + certain_false.sum())
    if stats is not None:
        stats.float_decisions += n_certain

    if boundary.any():
        for idx in np.flatnonzero(boundary):
            out[idx] = within(
                point, curve[idx], delta, policy=policy, stats=stats
            )
    return np.asarray(out, dtype=bool)
