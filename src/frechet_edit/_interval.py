"""Vectorised interval arithmetic with outward rounding.

Each value is a pair ``(lo, hi)`` of float64 arrays with ``lo <= true <= hi``.
IEEE-754 round-to-nearest returns the correctly rounded result of every
``+ - * / sqrt``, which lies within half an ulp of the exact value, subnormals
included. Stepping one ulp outward with ``nextafter`` after every operation
therefore gives a rigorous enclosure without switching the rounding mode.

Overflow is harmless: an enclosure that reaches ``inf`` is still an enclosure.
The one thing that is not is ``nan`` (from ``inf - inf`` or ``0 * inf``), so
:func:`sanitize` widens any such entry to the whole real line. A whole-line
enclosure is never wrong; it only means the exact tier decides that entry.
"""

from __future__ import annotations

import numpy as np

Iv = tuple[np.ndarray, np.ndarray]

_NEG_INF = -np.inf
_POS_INF = np.inf


def _down(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.nextafter(x, _NEG_INF))


def _up(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.nextafter(x, _POS_INF))


def exact(x: np.ndarray) -> Iv:
    """A float64 array is exact: its enclosure has zero width."""
    arr = np.asarray(x, dtype=np.float64)
    return arr, arr


def sub_exact(a: np.ndarray, b: np.ndarray) -> Iv:
    """Enclosure of ``a - b`` for exact float64 inputs."""
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return _down(r), _up(r)


def add(x: Iv, y: Iv) -> Iv:
    with np.errstate(over="ignore", invalid="ignore"):
        return _down(x[0] + y[0]), _up(x[1] + y[1])


def sub(x: Iv, y: Iv) -> Iv:
    with np.errstate(over="ignore", invalid="ignore"):
        return _down(x[0] - y[1]), _up(x[1] - y[0])


def neg(x: Iv) -> Iv:
    return -x[1], -x[0]


def mul(x: Iv, y: Iv) -> Iv:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        p1 = x[0] * y[0]
        p2 = x[0] * y[1]
        p3 = x[1] * y[0]
        p4 = x[1] * y[1]
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return _down(lo), _up(hi)


def sqr(x: Iv) -> Iv:
    """Enclosure of ``x * x``, which is tighter than ``mul(x, x)`` across zero."""
    lo, hi = x
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        lo2 = lo * lo
        hi2 = hi * hi
    low = np.where(lo >= 0.0, lo2, np.where(hi <= 0.0, hi2, 0.0))
    high = np.maximum(lo2, hi2)
    return np.where(low > 0.0, _down(low), 0.0), _up(high)


def sqrt_nonneg(x: Iv) -> Iv:
    """Enclosure of ``sqrt(max(x, 0))``; callers decide sign separately."""
    with np.errstate(invalid="ignore"):
        lo = np.sqrt(np.maximum(x[0], 0.0))
        hi = np.sqrt(np.maximum(x[1], 0.0))
    return np.where(lo > 0.0, _down(lo), 0.0), _up(hi)


def div_pos(x: Iv, y: Iv) -> Iv:
    """Enclosure of ``x / y`` where the caller guarantees ``y > 0`` exactly.

    If the enclosure of ``y`` still touches zero (possible after underflow),
    the result is the whole real line rather than a false bound.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        q1 = x[0] / y[0]
        q2 = x[0] / y[1]
        q3 = x[1] / y[0]
        q4 = x[1] / y[1]
        lo = np.minimum(np.minimum(q1, q2), np.minimum(q3, q4))
        hi = np.maximum(np.maximum(q1, q2), np.maximum(q3, q4))
    unsafe = ~(y[0] > 0.0)
    lo = np.where(unsafe, _NEG_INF, _down(lo))
    hi = np.where(unsafe, _POS_INF, _up(hi))
    return lo, hi


def sanitize(x: Iv) -> Iv:
    """Replace any ``nan`` bound by the corresponding infinity."""
    lo, hi = x
    bad = np.isnan(lo) | np.isnan(hi)
    if not bad.any():
        return lo, hi
    return np.where(bad, _NEG_INF, lo), np.where(bad, _POS_INF, hi)


def dot(xs: list[Iv], ys: list[Iv]) -> Iv:
    """Enclosure of ``sum_k xs[k] * ys[k]``."""
    total = mul(xs[0], ys[0])
    for xk, yk in zip(xs[1:], ys[1:], strict=True):
        total = add(total, mul(xk, yk))
    return total


def sumsq(xs: list[Iv]) -> Iv:
    """Enclosure of ``sum_k xs[k]**2``."""
    total = sqr(xs[0])
    for xk in xs[1:]:
        total = add(total, sqr(xk))
    return total
