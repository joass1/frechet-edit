"""Minimum-enclosing-ball geometry for the insertion recurrence.

Two things are needed by ``_reference_dp``:

``mu_indices``
    for every reference index ``i``, the smallest ``t`` such that the minimum
    enclosing ball of ``R[t..i]`` has radius ``<= delta``. A single inserted
    point can be coupled to exactly the blocks ``R[k..i]`` with ``k >= mu(i)``.

``certified_block_centre``
    a float64 point that is provably within ``delta`` of every vertex of a
    block, for witness emission - or ``None`` when no such float64 point could
    be certified (see ``docs/numerics.md`` section 3).

The decision procedure is the tiered policy of ``docs/numerics.md``: a float64
Welzl pass proposes a support set, exact rational arithmetic certifies YES (an
enclosing ball of exact radius <= delta) or NO (a subset of at most three
points whose own minimum enclosing ball already exceeds delta), and only
genuinely undecided cases reach the capped exact enumeration or abstain.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from ._meb import FracBall, exact_meb_nd
from ._meb import ball_from_one as _ball_from_one
from ._meb import ball_from_two as _ball_from_two
from ._meb import encloses as _encloses
from ._numerics import (
    EXACT_BALL_MAX_POINTS,
    SAFETY,
    NumericallyAmbiguous,
    NumericPolicy,
    PredicateStats,
    U,
    _gamma,
    frac_sq_dist,
)

#: Seed for the shuffle in the float fast path. Reported in result metadata so
#: that any run can be reproduced exactly.
DEFAULT_SEED = 20240612


def _frac_point(p: Sequence[float]) -> tuple[Fraction, ...]:
    return tuple(Fraction(float(x)) for x in p)


def _circumball_2d(
    a: Sequence[Fraction], b: Sequence[Fraction], c: Sequence[Fraction]
) -> FracBall | None:
    """Exact circumcircle of three planar points, or None when they are collinear."""
    ax, ay = a[0], a[1]
    bx, by = b[0] - ax, b[1] - ay
    cx, cy = c[0] - ax, c[1] - ay
    d = 2 * (bx * cy - by * cx)
    if d == 0:
        return None
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    return FracBall((ax + ux, ay + uy), ux * ux + uy * uy)


def _exact_meb_2d(pts: Sequence[tuple[Fraction, ...]]) -> FracBall:
    """Exact planar minimum enclosing ball by complete candidate enumeration."""
    n = len(pts)
    best: FracBall | None = None
    for i in range(n):
        cand = _ball_from_one(pts[i])
        if _encloses(cand, pts) and (best is None or cand.sq_radius < best.sq_radius):
            best = cand
    for i in range(n):
        for j in range(i + 1, n):
            cand = _ball_from_two(pts[i], pts[j])
            if _encloses(cand, pts) and (best is None or cand.sq_radius < best.sq_radius):
                best = cand
    if best is None:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    c3 = _circumball_2d(pts[i], pts[j], pts[k])
                    if c3 is None:
                        continue
                    if _encloses(c3, pts) and (best is None or c3.sq_radius < best.sq_radius):
                        best = c3
    if best is None:  # pragma: no cover - an enclosing candidate always exists
        raise AssertionError("no enclosing candidate found")
    return best


def exact_meb(points: np.ndarray) -> FracBall:
    """Exact minimum enclosing ball of a point set in any supported dimension."""
    if len(points) == 0:
        raise ValueError("exact_meb requires at least one point")
    dim = points.shape[1]
    pts = [_frac_point(p) for p in points]
    if dim == 1:
        lo = min(p[0] for p in pts)
        hi = max(p[0] for p in pts)
        half = (hi - lo) / 2
        return FracBall(((lo + hi) / 2,), half * half)
    if len(pts) > EXACT_BALL_MAX_POINTS:
        raise NumericallyAmbiguous(
            f"exact enclosing-ball enumeration is capped at {EXACT_BALL_MAX_POINTS} "
            f"points, got {len(pts)}"
        )
    if dim == 2:
        return _exact_meb_2d(pts)
    # Dimension 3 and above. The planar enumeration would need C(n, d+1)
    # candidate subsets, so it is replaced by support refinement; see _meb.
    try:
        return exact_meb_nd(pts, dim)
    except RuntimeError as exc:
        raise NumericallyAmbiguous(str(exc)) from exc


def exact_meb_small(pts: Sequence[tuple[Fraction, ...]]) -> FracBall:
    """Exact minimum enclosing ball of a small already-rational support set."""
    if len(pts) == 1:
        return _ball_from_one(pts[0])
    dim = len(pts[0])
    if dim == 1:
        lo = min(p[0] for p in pts)
        hi = max(p[0] for p in pts)
        half = (hi - lo) / 2
        return FracBall(((lo + hi) / 2,), half * half)
    if dim == 2:
        return _exact_meb_2d(pts)
    return exact_meb_nd(pts, dim)


# ---------------------------------------------------------------------------
# float64 fast path
# ---------------------------------------------------------------------------


def _welzl_support_2d(pts: np.ndarray, rng: random.Random) -> list[int]:
    """Indices of a support set of at most three points, proposed in float64.

    This is a candidate generator only. Every decision it feeds is afterwards
    certified (or refuted) in exact arithmetic, so a float slip here costs
    performance, never correctness.

    Deliberately free of numpy. Profiling a mixed-mode solve at m = n = 400
    showed this function dominating the whole package - 1.46 million containment
    checks, 40 percent of total runtime, more than the dynamic program it feeds.
    The obvious fix, vectorising each scan, was tried and measured and made
    things WORSE: roughly twice as slow for blocks of ten points or fewer, which
    is nearly all of them, because `mu_indices` calls this on short windows and
    numpy's per-call overhead swamps the work. It only won past about a hundred
    points, a size that barely occurs.

    So the arithmetic here is plain Python floats on unpacked coordinates. In
    two dimensions that is a handful of multiplications with no array allocation,
    no dtype dispatch and no ufunc call.
    """
    n = len(pts)
    order = list(range(n))
    rng.shuffle(order)
    # Unpack once into scalars; every operation below is plain float arithmetic.
    xs = [float(v) for v in pts[:, 0]]
    ys = [float(v) for v in pts[:, 1]]

    def ball2(i: int, j: int) -> tuple[float, float, float, list[int]]:
        cx = (xs[i] + xs[j]) / 2.0
        cy = (ys[i] + ys[j]) / 2.0
        dx, dy = xs[i] - cx, ys[i] - cy
        return cx, cy, dx * dx + dy * dy, [i, j]

    def ball3(i: int, j: int, k: int) -> tuple[float, float, float, list[int]]:
        ax, ay = xs[i], ys[i]
        bx, by = xs[j] - ax, ys[j] - ay
        cx_, cy_ = xs[k] - ax, ys[k] - ay
        det = 2.0 * (bx * cy_ - by * cx_)
        if det == 0.0:  # collinear: the widest defining pair wins
            return max((ball2(i, j), ball2(i, k), ball2(j, k)), key=lambda t: t[2])
        b2 = bx * bx + by * by
        c2 = cx_ * cx_ + cy_ * cy_
        ux = (cy_ * b2 - by * c2) / det
        uy = (bx * c2 - cx_ * b2) / det
        return ax + ux, ay + uy, ux * ux + uy * uy, [i, j, k]

    cx, cy, r2, sup = xs[order[0]], ys[order[0]], 0.0, [order[0]]
    for a in range(1, n):
        ia = order[a]
        dx, dy = xs[ia] - cx, ys[ia] - cy
        if dx * dx + dy * dy <= r2 + 1e-12 * (r2 if r2 > 1.0 else 1.0):
            continue
        cx, cy, r2, sup = xs[ia], ys[ia], 0.0, [ia]
        for b in range(a):
            ib = order[b]
            dx, dy = xs[ib] - cx, ys[ib] - cy
            if dx * dx + dy * dy <= r2 + 1e-12 * (r2 if r2 > 1.0 else 1.0):
                continue
            cx, cy, r2, sup = ball2(ia, ib)
            for c in range(b):
                ic = order[c]
                dx, dy = xs[ic] - cx, ys[ic] - cy
                if dx * dx + dy * dy <= r2 + 1e-12 * (r2 if r2 > 1.0 else 1.0):
                    continue
                cx, cy, r2, sup = ball3(ia, ib, ic)
    return sup


def _diameter_triple(pts: np.ndarray) -> list[int]:
    """A cheap candidate subset for dimension 3 and above.

    A far-apart pair plus the point farthest from their midpoint. This is only a
    candidate generator: whatever it returns is a SUBSET of the block, so the
    exact ball computed from it is a lower bound on the block's own minimum
    enclosing ball, and the NO certificate built from it stays sound. A weak
    guess costs a trip to the exact path, never an answer.

    A full Welzl pass in general dimension would give a tighter subset. It is
    not worth it here: dimension 3 and above is the uncommon case, and the exact
    path behind this is complete on its own.
    """
    centroid = pts.mean(axis=0)
    first = int(np.argmax(((pts - centroid) ** 2).sum(axis=1)))
    second = int(np.argmax(((pts - pts[first]) ** 2).sum(axis=1)))
    midpoint = (pts[first] + pts[second]) / 2.0
    third = int(np.argmax(((pts - midpoint) ** 2).sum(axis=1)))
    return [first, second, third]


def _support_indices(pts: np.ndarray, rng: random.Random) -> list[int]:
    dim = pts.shape[1]
    if dim == 1:
        return [int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0]))]
    if dim == 2:
        return _welzl_support_2d(pts, rng)
    return _diameter_triple(pts)


def meb_radius_le(
    block: np.ndarray,
    delta: float,
    *,
    policy: NumericPolicy = "certified",
    stats: PredicateStats | None = None,
    rng: random.Random | None = None,
) -> bool:
    """Decide whether the minimum enclosing ball of ``block`` has radius <= delta.

    ``block`` is a ``(k, d)`` float64 array with ``d`` in ``{1, 2}``. The
    comparison is closed.
    """
    if stats is not None:
        stats.ball_calls += 1
    if len(block) == 1:
        return True

    rng = rng or random.Random(DEFAULT_SEED)
    delta_f = float(delta)
    delta2 = Fraction(delta_f) ** 2

    # The support set has at most three points, so exact arithmetic on it is
    # cheap. Only the two paths below ever touch the whole block in rationals.
    sup = [_frac_point(block[i]) for i in dict.fromkeys(_support_indices(block, rng))]
    cand = exact_meb_small(sup)

    # NO certificate: MEB(support) <= MEB(block), so a support ball that already
    # exceeds delta settles it. Exact, and touches at most three points.
    if cand.sq_radius > delta2:
        return False

    # YES certificate in float64 with a rigorous forward error bound. The
    # rounded centre is an ACTUAL float64 point, so if every squared distance
    # from it is provably below delta^2 the point itself is the certificate -
    # whether or not it is the true minimum-enclosing centre.
    centre_f = np.array([float(c) for c in cand.centre], dtype=np.float64)
    diff = block - centre_f
    max_d2 = float(np.einsum("ij,ij->i", diff, diff).max())
    dim = block.shape[1]
    if max_d2 * (1.0 + SAFETY * _gamma(dim)) < delta_f * delta_f * (1.0 - SAFETY * U):
        return True

    # Boundary. Fall back to exact arithmetic over the whole block.
    pts = [_frac_point(p) for p in block]
    if _encloses(cand, pts):
        return True

    if policy == "fast":
        return bool(max_d2 <= delta_f * delta_f)

    if stats is not None:
        stats.exact_fallbacks += 1
    if len(block) > EXACT_BALL_MAX_POINTS:
        if stats is not None:
            stats.ambiguous += 1
        raise NumericallyAmbiguous(
            f"enclosing-ball decision for a block of {len(block)} points sits on the "
            f"delta boundary and exceeds the exact cap of {EXACT_BALL_MAX_POINTS}"
        )
    # Dimension-general: exact_meb dispatches to the planar enumeration for
    # d <= 2 and to support refinement above it.
    return exact_meb(block).sq_radius <= delta2


def mu_indices(
    reference: np.ndarray,
    delta: float,
    *,
    policy: NumericPolicy = "certified",
    stats: PredicateStats | None = None,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """``mu[i]`` is the smallest ``t`` with ``radius(MEB(reference[t..i])) <= delta``.

    Zero-based, inclusive at both ends. ``mu`` is non-decreasing, which is what
    lets the caller slide a monotone minimum over the window.

    The two-pointer loop performs ``O(m)`` enclosing-ball calls in total:
    every iteration either advances ``i`` or advances ``t``, and neither index
    ever moves backwards. ``mu[i] <= i`` always holds because a single point
    has radius 0 and ``delta > 0``.
    """
    m = len(reference)
    rng = random.Random(seed)
    mu = np.zeros(m, dtype=np.int64)
    t = 0
    for i in range(m):
        while t < i and not meb_radius_le(
            reference[t : i + 1], delta, policy=policy, stats=stats, rng=rng
        ):
            t += 1
        mu[i] = t
    return mu


def certified_block_centre(
    block: np.ndarray,
    delta: float,
) -> np.ndarray | None:
    """A float64 point provably within ``delta`` of every vertex of ``block``.

    Returns ``None`` when no float64 point near the exact centre can be
    certified. The caller then reports ``witness_status="unavailable"`` and
    leaves the cost untouched; ``delta`` is never adjusted to rescue a witness.
    """
    dim = block.shape[1]
    delta2 = Fraction(float(delta)) ** 2
    pts = [_frac_point(p) for p in block]

    rng = random.Random(DEFAULT_SEED)
    sup = [_frac_point(block[i]) for i in dict.fromkeys(_support_indices(block, rng))]
    ball = exact_meb_small(sup)
    if not (ball.sq_radius <= delta2 and _encloses(ball, pts)):
        try:
            ball = exact_meb(block)
        except NumericallyAmbiguous:
            return None
        if ball.sq_radius > delta2:
            return None

    base = np.array([float(c) for c in ball.centre], dtype=np.float64)
    if all(frac_sq_dist(base, p) <= delta2 for p in block):
        return base

    # Deterministic neighbour search: the rounded centre plus one ulp in each
    # direction on each axis. At most 3**d candidates, tried in a fixed order.
    centroid = block.mean(axis=0)
    combos: list[list[float]] = [[]]
    for axis in range(dim):
        towards = float(centroid[axis])
        away = -np.inf if towards > base[axis] else np.inf
        choices = [
            float(base[axis]),
            float(np.nextafter(base[axis], towards)),
            float(np.nextafter(base[axis], away)),
        ]
        combos = [[*prev, c] for prev in combos for c in choices]
    for coords in combos:
        cand = np.array(coords, dtype=np.float64)
        if all(frac_sq_dist(cand, p) <= delta2 for p in block):
            return cand
    return None
