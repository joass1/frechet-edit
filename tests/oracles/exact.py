"""Exact rational primitives for the independent oracles.

Nothing in this package consults the production package.  Every value here is
derived from ``docs/definition.md`` alone.

All arithmetic is performed on :class:`fractions.Fraction`, so every comparison
is exact.  A finite float64 is an exact binary rational, so ``Fraction(x)``
converts it without rounding and no hidden epsilon can ever enlarge or shrink
``delta`` (``docs/definition.md`` section 6: comparisons are CLOSED, a pair is
acceptable when ``distance <= delta``).
"""

from __future__ import annotations

import math
import numbers
from fractions import Fraction
from typing import Any

__all__ = [
    "delta_sq",
    "sq_dist",
    "to_frac",
    "to_frac_point",
    "within",
]


def to_frac(x: Any) -> Fraction:
    """Convert a single scalar to an exact :class:`Fraction`.

    Raises ``ValueError`` for NaN/infinite values, which are not legal
    coordinates (``docs/definition.md`` section 6).
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, numbers.Integral):
        return Fraction(int(x))
    if isinstance(x, (str, bytes)):
        raise ValueError(f"not a numeric coordinate: {x!r}")
    try:
        value = float(x)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"not a numeric coordinate: {x!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite coordinate: {x!r}")
    # float -> Fraction is exact: a finite binary float IS a rational.
    return Fraction(value)


def to_frac_point(p: Any) -> tuple[Fraction, ...]:
    """Convert a point to a tuple of exact :class:`Fraction` coordinates.

    Accepts sequences, numpy arrays and bare scalars (a scalar is read as a
    one-dimensional point).  Raises ``ValueError`` on an empty point or a
    non-finite coordinate.
    """
    if isinstance(p, tuple) and p and all(type(c) is Fraction for c in p):
        return p  # already canonical
    if isinstance(p, (str, bytes)):
        raise ValueError(f"not a point: {p!r}")
    if isinstance(p, (numbers.Number, Fraction)):
        return (to_frac(p),)
    try:
        coords = list(p)
    except TypeError:
        return (to_frac(p),)
    if not coords:
        raise ValueError("a point must have at least one coordinate")
    return tuple(to_frac(c) for c in coords)


def sq_dist(a: Any, b: Any) -> Fraction:
    """Exact squared Euclidean distance between two points."""
    pa = to_frac_point(a)
    pb = to_frac_point(b)
    if len(pa) != len(pb):
        raise ValueError(f"dimension mismatch: {len(pa)} vs {len(pb)}")
    total = Fraction(0)
    for x, y in zip(pa, pb, strict=True):
        diff = x - y
        total += diff * diff
    return total


def delta_sq(delta: Any) -> Fraction:
    """Exact ``delta ** 2``, validating ``delta`` per the public contract."""
    if isinstance(delta, Fraction):
        value = delta
        finite = True
    else:
        try:
            as_float = float(delta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"delta must be a finite positive number: {delta!r}") from exc
        finite = math.isfinite(as_float)
        value = Fraction(as_float) if finite else Fraction(0)
    if not finite:
        raise ValueError(f"delta must be finite: {delta!r}")
    if value <= 0:
        raise ValueError(f"delta must be strictly positive: {delta!r}")
    return value * value


def within(a: Any, b: Any, delta: Any) -> bool:
    """Exact CLOSED predicate ``dist(a, b) <= delta``.

    Implemented as ``sq_dist(a, b) <= Fraction(delta) ** 2``; squaring is
    monotone on non-negative reals, so this is equivalent and needs no square
    root.
    """
    return sq_dist(a, b) <= delta_sq(delta)
