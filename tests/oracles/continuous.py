"""Independent continuous Frechet oracles, written from the definitions alone.

Source of truth: Alt and Godau's free-space characterisation, as restated in
Section 2 of Fox, Nayyeri, Perry and Raichel (SoCG 2024):

    d_F(P, Q) <= delta  iff  there is an (x, y)-monotone path from (0, 0) to
    (m-1, n-1) inside { (s, t) : ||P(s) - Q(t)|| <= delta },

and the definition of deletion-only continuous edit distance: the fewest
vertices of Q whose removal leaves a NON-EMPTY curve Q' with
d_F(P, Q') <= delta.

This file shares nothing with ``src/frechet_edit``. In particular it uses a
DIFFERENT exact technique from the package for comparing interval endpoints
of the form ``x + s*sqrt(y)``: equality by normalising rational squares (two
such numbers with irrational roots are equal only if their triples are
identical), and order by refining integer square roots until the rational
enclosures separate. The package instead squares with sign bookkeeping. A
shared blind spot would therefore need two different wrong proofs.

``deletion_oracle`` enumerates every retained subsequence, largest first. It
has no dynamic program, no DAG complex and no copies.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

__all__ = ["OracleUndecided", "deletion_oracle", "frechet_le", "free_interval"]

Point = tuple[Fraction, ...]
Alg = tuple[Fraction, int, Fraction]  # x + s * sqrt(y)

_MAX_BITS = 8192


class OracleUndecided(Exception):
    """Refinement failed to separate two unequal-looking values. Never a pass."""


def _point(p: Any) -> Point:
    return tuple(Fraction(float(x)) for x in (p if hasattr(p, "__len__") else [p]))


def _sq(u: Point, v: Point) -> Fraction:
    return sum(((a - b) * (a - b) for a, b in zip(u, v, strict=True)), Fraction(0))


def _rational_sqrt(y: Fraction) -> Fraction | None:
    """``sqrt(y)`` if it is rational, else ``None``."""
    num, den = y.numerator, y.denominator
    rn, rd = math.isqrt(num), math.isqrt(den)
    if rn * rn == num and rd * rd == den:
        return Fraction(rn, rd)
    return None


def _normal(r: Alg) -> Alg:
    x, s, y = r
    if s == 0 or y == 0:
        return (x, 0, Fraction(0))
    root = _rational_sqrt(y)
    if root is not None:
        return (x + s * root, 0, Fraction(0))
    return r


def _bounds(r: Alg, bits: int) -> tuple[Fraction, Fraction]:
    x, s, y = r
    if s == 0:
        return x, x
    num = y.numerator * y.denominator << (2 * bits)
    root = math.isqrt(num)
    scale = y.denominator << bits
    lo, hi = Fraction(root, scale), Fraction(root + 1, scale)
    return (x + lo, x + hi) if s > 0 else (x - hi, x - lo)


def _cmp(r1: Alg, r2: Alg) -> int:
    a, b = _normal(r1), _normal(r2)
    if a == b:
        return 0
    if a[1] == 0 and b[1] == 0:
        return (a[0] > b[0]) - (a[0] < b[0])
    # At least one is irrational and the triples differ, so the values differ
    # (see module docstring); refine until the enclosures separate.
    bits = 64
    while bits <= _MAX_BITS:
        alo, ahi = _bounds(a, bits)
        blo, bhi = _bounds(b, bits)
        if ahi < blo:
            return -1
        if bhi < alo:
            return 1
        bits *= 2
    raise OracleUndecided(f"could not separate {a} and {b}")


ZERO: Alg = (Fraction(0), 0, Fraction(0))
ONE: Alg = (Fraction(1), 0, Fraction(0))


def free_interval(c: Point, u: Point, v: Point, d2: Fraction) -> tuple[Alg, Alg] | None:
    """``{t in [0,1] : ||u + t (v - u) - c||^2 <= d2}`` as exact endpoints, or None."""
    d = [b - a for a, b in zip(u, v, strict=True)]
    w = [a - b for a, b in zip(u, c, strict=True)]
    qa = sum((x * x for x in d), Fraction(0))
    qb = sum((x * y for x, y in zip(d, w, strict=True)), Fraction(0))
    qc = sum((x * x for x in w), Fraction(0)) - d2
    if qa == 0:
        return (ZERO, ONE) if qc <= 0 else None
    start_free = qc <= 0
    end_free = qa + 2 * qb + qc <= 0
    disc = qb * qb - qa * qc
    if not start_free and not end_free and (disc < 0 or not (0 < -qb < qa)):
        return None
    x, y = -qb / qa, disc / (qa * qa)
    lo = ZERO if start_free else (x, -1, y)
    hi = ONE if end_free else (x, 1, y)
    return lo, hi


def frechet_le(P: Sequence[Any], Q: Sequence[Any], delta: float) -> bool:
    """Exact decision ``d_F(P, Q) <= delta`` for polygonal curves of any length >= 1."""
    ps = [_point(p) for p in P]
    qs = [_point(q) for q in Q]
    if not ps or not qs:
        raise ValueError("curves must be non-empty")
    d2 = Fraction(float(delta)) ** 2
    m, n = len(ps), len(qs)
    if m == 1:
        return all(_sq(ps[0], q) <= d2 for q in qs)
    if n == 1:
        return all(_sq(p, qs[0]) <= d2 for p in ps)
    if _sq(ps[0], qs[0]) > d2 or _sq(ps[-1], qs[-1]) > d2:
        return False

    # left[i][j]: reachable part of the vertical edge {i} x [j, j+1].
    # low[i][j]:  reachable part of the horizontal edge [i, i+1] x {j}.
    left: list[list[tuple[Alg, Alg] | None]] = [[None] * (n - 1) for _ in range(m)]
    low: list[list[tuple[Alg, Alg] | None]] = [[None] * n for _ in range(m - 1)]

    for j in range(n - 1):
        if all(_sq(ps[0], qs[t]) <= d2 for t in range(j + 1)):
            left[0][j] = free_interval(ps[0], qs[j], qs[j + 1], d2)
    for i in range(m - 1):
        if all(_sq(ps[t], qs[0]) <= d2 for t in range(i + 1)):
            low[i][0] = free_interval(qs[0], ps[i], ps[i + 1], d2)

    for i in range(m - 1):
        for j in range(n - 1):
            lft, bot = left[i][j], low[i][j]
            right_free = free_interval(ps[i + 1], qs[j], qs[j + 1], d2)
            top_free = free_interval(qs[j + 1], ps[i], ps[i + 1], d2)
            if right_free is not None:
                if bot is not None:
                    left[i + 1][j] = right_free
                elif lft is not None and _cmp(right_free[1], lft[0]) >= 0:
                    lo = right_free[0] if _cmp(right_free[0], lft[0]) >= 0 else lft[0]
                    left[i + 1][j] = (lo, right_free[1])
            if top_free is not None:
                if lft is not None:
                    low[i][j + 1] = top_free
                elif bot is not None and _cmp(top_free[1], bot[0]) >= 0:
                    lo = top_free[0] if _cmp(top_free[0], bot[0]) >= 0 else bot[0]
                    low[i][j + 1] = (lo, top_free[1])

    return left[m - 1][n - 2] is not None or low[m - 2][n - 1] is not None


def deletion_oracle(P: Sequence[Any], Q: Sequence[Any], delta: float) -> float:
    """Fewest deletions from ``Q`` leaving a non-empty ``Q'`` with ``d_F(P, Q') <= delta``.

    Exhaustive over retained subsequences, largest first; ``inf`` if none works.
    """
    n = len(Q)
    for deleted in range(n):
        for kept in itertools.combinations(range(n), n - deleted):
            if frechet_le(P, [Q[i] for i in kept], delta):
                return float(deleted)
    return math.inf
