"""Certified free-space intervals for the continuous Frechet machinery.

For a point ``c`` and a segment ``p -> q`` the free interval is

    { t in [0, 1] : ||p + t (q - p) - c|| <= delta }

which is convex, so it is empty or a closed interval ``[lo, hi]``. Writing
``f(t) = A t^2 + 2 B t + C`` with ``A = |q-p|^2``, ``B = (q-p).(p-c)`` and
``C = |p-c|^2 - delta^2``:

* ``lo`` is exactly ``0`` when ``p`` itself is free, and otherwise the smaller
  root ``(-B - sqrt(D)) / A`` with ``D = B^2 - A C``;
* ``hi`` is exactly ``1`` when ``q`` is free, and otherwise the larger root;
* when neither endpoint is free the interval is non-empty iff ``D >= 0`` and
  the vertex ``-B / A`` lies strictly inside ``(0, 1)``;
* a degenerate segment (``p == q``) is free everywhere or nowhere.

The solver never needs the VALUE of an endpoint, only its ORDER relative to
other endpoints on the same segment. So each segment's endpoints are ranked
once, here, and the solver compares integers. The ranking is exact: float64
enclosures from :mod:`._interval` separate almost every pair, and any pair
whose enclosures overlap is ordered by :func:`._quadroot.compare` in rational
arithmetic. Equal values always receive equal ranks. See ``docs/continuous.md``
section 4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

import numpy as np

from . import _interval as iv
from ._numerics import NumericPolicy, PredicateStats, within_row
from ._quadroot import ONE, ZERO, QuadNum, compare

__all__ = ["BIG", "FreeSpace", "IntervalFamily", "dense_ranks", "free_intervals"]

#: Rank sentinel meaning "empty / unreachable". Larger than any real rank.
BIG = np.int32(2**31 - 1)

FracPoint = tuple[Fraction, ...]


def _frac_points(points: np.ndarray) -> list[FracPoint]:
    return [tuple(Fraction(float(x)) for x in row) for row in points]


def _exact_quadratic(
    c: FracPoint, p: FracPoint, q: FracPoint, delta2: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    """Exact ``(A, B, D)`` of the free-interval quadratic, see module docstring."""
    d = [qk - pk for pk, qk in zip(p, q, strict=True)]
    w = [pk - ck for pk, ck in zip(p, c, strict=True)]
    a_coef = sum((x * x for x in d), Fraction(0))
    b_coef = sum((x * y for x, y in zip(d, w, strict=True)), Fraction(0))
    c_coef = sum((x * x for x in w), Fraction(0)) - delta2
    return a_coef, b_coef, b_coef * b_coef - a_coef * c_coef


@dataclass
class IntervalFamily:
    """Free intervals of every (point, segment) pair in a ``(P, S)`` grid.

    ``lo_zero`` / ``hi_one`` mark endpoints that are the exact constants 0 and
    1; every other endpoint of a non-empty interval is a quadratic root whose
    exact value :meth:`exact_endpoint` reconstructs on demand.
    """

    nonempty: np.ndarray
    lo_zero: np.ndarray
    hi_one: np.ndarray
    lo_enc: iv.Iv
    hi_enc: iv.Iv
    point_array: np.ndarray
    points: list[FracPoint]
    seg_start: list[FracPoint]
    seg_end: list[FracPoint]
    delta2: Fraction
    stats: PredicateStats

    def exact_endpoint(self, p: int, s: int, upper: bool) -> QuadNum:
        if upper and self.hi_one[p, s]:
            return ONE
        if not upper and self.lo_zero[p, s]:
            return ZERO
        self.stats.exact_fallbacks += 1
        a_coef, b_coef, disc = _exact_quadratic(
            self.points[p], self.seg_start[s], self.seg_end[s], self.delta2
        )
        return QuadNum(-b_coef / a_coef, 1 if upper else -1, disc / (a_coef * a_coef))


def free_intervals(
    points: np.ndarray,
    frac_points: list[FracPoint],
    seg_start: np.ndarray,
    frac_start: list[FracPoint],
    seg_end: np.ndarray,
    frac_end: list[FracPoint],
    delta: float,
    free_start: np.ndarray,
    free_end: np.ndarray,
    stats: PredicateStats,
) -> IntervalFamily:
    """Free intervals for every pair (``points[p]``, segment ``s``), shape ``(P, S)``.

    ``free_start[p, s]`` and ``free_end[p, s]`` must be the CERTIFIED answers to
    ``||seg_start[s] - points[p]|| <= delta`` and the same for ``seg_end``; the
    caller computes them once for the whole vertex grid.
    """
    dim = points.shape[1]
    d_iv = [iv.sub_exact(seg_end[None, :, k], seg_start[None, :, k]) for k in range(dim)]
    w_iv = [iv.sub_exact(seg_start[None, :, k], points[:, None, k]) for k in range(dim)]
    a_iv = iv.sumsq(d_iv)
    b_iv = iv.dot(d_iv, w_iv)
    c_iv = iv.sub(iv.sumsq(w_iv), iv.sqr(iv.exact(np.asarray(delta))))
    disc_iv = iv.sanitize(iv.sub(iv.sqr(b_iv), iv.mul(a_iv, c_iv)))
    negb = iv.sanitize(iv.neg(b_iv))
    a_iv = iv.sanitize(a_iv)
    root = iv.sqrt_nonneg(disc_iv)
    t_minus = iv.sanitize(iv.div_pos(iv.sub(negb, root), a_iv))
    t_plus = iv.sanitize(iv.div_pos(iv.add(negb, root), a_iv))

    shape = (points.shape[0], seg_start.shape[0])
    degenerate = np.broadcast_to(np.all(seg_start == seg_end, axis=1)[None, :], shape)
    either_free = free_start | free_end

    # Neither endpoint free: non-empty iff D >= 0 and 0 < -B < A (A > 0 here).
    gap = iv.sanitize(iv.sub(a_iv, negb))
    tests = [
        (disc_iv[0] >= 0.0, disc_iv[1] < 0.0),
        (negb[0] > 0.0, negb[1] <= 0.0),
        (gap[0] > 0.0, gap[1] <= 0.0),
    ]
    certainly_true = np.logical_and.reduce([t for t, _ in tests])
    certainly_false = np.logical_or.reduce([f for _, f in tests])
    interior = certainly_true & ~certainly_false
    undecided = ~either_free & ~degenerate & ~certainly_true & ~certainly_false

    delta2 = Fraction(float(delta)) ** 2
    for p, s in zip(*np.nonzero(undecided), strict=True):
        stats.exact_fallbacks += 1
        a_coef, b_coef, disc = _exact_quadratic(
            frac_points[p], frac_start[s], frac_end[s], delta2
        )
        interior[p, s] = disc >= 0 and 0 < -b_coef < a_coef

    nonempty = either_free | (~degenerate & interior)
    lo_zero = nonempty & free_start
    hi_one = nonempty & free_end
    zero = np.zeros(shape)
    one = np.ones(shape)
    lo_enc = (np.where(lo_zero, zero, t_minus[0]), np.where(lo_zero, zero, t_minus[1]))
    hi_enc = (np.where(hi_one, one, t_plus[0]), np.where(hi_one, one, t_plus[1]))
    return IntervalFamily(
        nonempty=nonempty,
        lo_zero=lo_zero,
        hi_one=hi_one,
        lo_enc=lo_enc,
        hi_enc=hi_enc,
        point_array=points,
        points=frac_points,
        seg_start=frac_start,
        seg_end=frac_end,
        delta2=delta2,
        stats=stats,
    )


def _exact_order(exact: dict[int, QuadNum]) -> Callable[[int, int], int]:
    def order(u: int, v: int) -> int:
        return compare(exact[u], exact[v])

    return order


def dense_ranks(
    flo: np.ndarray, fhi: np.ndarray, exact_of: Callable[[int], QuadNum]
) -> np.ndarray:
    """Exact dense ranks of values known only through enclosures ``[flo, fhi]``.

    Equal values receive equal ranks and a smaller value a strictly smaller
    rank. After sorting by lower bound, a new CLUSTER starts wherever an
    enclosure begins strictly above every enclosure before it; values in
    different clusters are therefore strictly ordered, and only a cluster with
    several members needs resolving. Clusters of zero-width enclosures hold
    exact float values and are grouped directly; any other cluster is sorted
    with the exact comparator.
    """
    count = len(flo)
    ranks = np.empty(count, dtype=np.int32)
    if count == 0:
        return ranks
    order = np.lexsort((fhi, flo))
    slo = flo[order]
    shi = fhi[order]
    runmax = np.maximum.accumulate(shi)
    starts_mask = np.empty(count, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = slo[1:] > runmax[:-1]
    starts = np.flatnonzero(starts_mask)
    sizes = np.diff(np.append(starts, count))

    distinct = np.ones(len(starts), dtype=np.int64)
    local = np.zeros(count, dtype=np.int64)
    for c in np.flatnonzero(sizes > 1):
        st = int(starts[c])
        en = st + int(sizes[c])
        if np.array_equal(slo[st:en], shi[st:en]):
            _, inverse = np.unique(slo[st:en], return_inverse=True)
            local[st:en] = inverse
            distinct[c] = int(inverse.max()) + 1
            continue
        members = [int(x) for x in order[st:en]]
        exact = {idx: exact_of(idx) for idx in members}
        members.sort(key=cmp_to_key(_exact_order(exact)))
        # `order[st:en]` must follow the exact order so `local` lines up.
        order[st:en] = members
        rank = 0
        local[st] = 0
        for pos in range(1, len(members)):
            if compare(exact[members[pos - 1]], exact[members[pos]]) != 0:
                rank += 1
            local[st + pos] = rank
        distinct[c] = rank + 1

    offsets = np.cumsum(distinct) - distinct
    cluster_of = np.cumsum(starts_mask) - 1
    ranks[order] = (offsets[cluster_of] + local).astype(np.int32)
    return ranks


def family_ranks(fam: IntervalFamily) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment exact ranks of every non-empty interval's endpoints.

    Returns ``(rank_lo, rank_hi)`` of shape ``(P, S)``; ranks are comparable
    only within one segment column, and empty entries hold :data:`BIG`.
    """
    num_points, num_segs = fam.nonempty.shape
    rank_lo = np.full((num_points, num_segs), BIG, dtype=np.int32)
    rank_hi = np.full((num_points, num_segs), BIG, dtype=np.int32)
    # Bit-identical points give bit-identical intervals on every segment, so
    # only one representative per distinct point is ranked; duplicates (laps,
    # a stationary receiver) copy its ranks. Without this every duplicate pair
    # overlaps and is sent to the exact tier just to be found equal.
    _, first, inverse = np.unique(
        fam.point_array, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    is_rep = np.zeros(num_points, dtype=bool)
    is_rep[first] = True
    for s in range(num_segs):
        rows = np.flatnonzero(fam.nonempty[:, s] & is_rep)
        if len(rows) == 0:
            continue
        flo = np.concatenate([fam.lo_enc[0][rows, s], fam.hi_enc[0][rows, s]])
        fhi = np.concatenate([fam.lo_enc[1][rows, s], fam.hi_enc[1][rows, s]])
        k = len(rows)

        def exact_of(idx: int, rows: np.ndarray = rows, s: int = s, k: int = k) -> QuadNum:
            return fam.exact_endpoint(int(rows[idx % k]), s, upper=idx >= k)

        ranks = dense_ranks(flo, fhi, exact_of)
        rank_lo[rows, s] = ranks[:k]
        rank_hi[rows, s] = ranks[k:]
    representative = first[inverse]
    return rank_lo[representative], rank_hi[representative]


class FreeSpace:
    """Everything the continuous solver needs about one ``(reference, observation, delta)``.

    Vertex freedom ``F[a, j]`` is decided by the certified point predicate.
    Horizontal edges are reference segment ``a`` against observation vertex
    ``j``; vertical edges are reference vertex ``a`` against the observation
    segment from ``i = j - g - 1`` to ``j`` (a "gap" of ``g`` deleted vertices).
    Gap tables are built lazily and cached, because the solver's budget - and
    so the largest gap it needs - grows between runs.
    """

    def __init__(
        self,
        reference: np.ndarray,
        observation: np.ndarray,
        delta: float,
        *,
        policy: NumericPolicy = "certified",
        stats: PredicateStats | None = None,
    ) -> None:
        self.reference = reference
        self.observation = observation
        self.delta = float(delta)
        self.m = len(reference)
        self.n = len(observation)
        self.stats = stats if stats is not None else PredicateStats()
        self.free = np.stack(
            [
                within_row(reference[a], observation, delta, policy=policy, stats=self.stats)
                for a in range(self.m)
            ]
        )
        self._frac_ref = _frac_points(reference)
        self._frac_obs = _frac_points(observation)
        self.h_lo, self.h_hi, self.h_ne = self._horizontal()
        self._gap_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _horizontal(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = self.m, self.n
        if m < 2:
            empty_i = np.full((0, n), BIG, dtype=np.int32)
            return empty_i, empty_i.copy(), np.zeros((0, n), dtype=bool)
        fam = free_intervals(
            self.observation,
            self._frac_obs,
            self.reference[:-1],
            self._frac_ref[:-1],
            self.reference[1:],
            self._frac_ref[1:],
            self.delta,
            self.free[:-1].T,
            self.free[1:].T,
            self.stats,
        )
        rank_lo, rank_hi = family_ranks(fam)
        return rank_lo.T.copy(), rank_hi.T.copy(), fam.nonempty.T.copy()

    def _gap(self, g: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ranks for vertical edges with gap ``g``, as ``(m, n - g - 1)`` arrays."""
        if g not in self._gap_cache:
            n = self.n
            starts = slice(0, n - g - 1)
            ends = slice(g + 1, n)
            fam = free_intervals(
                self.reference,
                self._frac_ref,
                self.observation[starts],
                self._frac_obs[starts],
                self.observation[ends],
                self._frac_obs[ends],
                self.delta,
                self.free[:, starts],
                self.free[:, ends],
                self.stats,
            )
            rank_lo, rank_hi = family_ranks(fam)
            self._gap_cache[g] = (rank_lo, rank_hi, fam.nonempty)
        return self._gap_cache[g]

    def vertical(self, num_gaps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(v_lo, v_hi, v_ne)`` of shape ``(m, n, num_gaps)``.

        Entry ``[a, j, g]`` describes the edge from observation vertex
        ``j - g - 1`` to ``j`` against reference vertex ``a``; entries with
        ``j - g - 1 < 0`` are empty.
        """
        m, n = self.m, self.n
        v_lo = np.full((m, n, num_gaps), BIG, dtype=np.int32)
        v_hi = np.full((m, n, num_gaps), BIG, dtype=np.int32)
        v_ne = np.zeros((m, n, num_gaps), dtype=bool)
        for g in range(num_gaps):
            rank_lo, rank_hi, nonempty = self._gap(g)
            v_lo[:, g + 1 :, g] = rank_lo
            v_hi[:, g + 1 :, g] = rank_hi
            v_ne[:, g + 1 :, g] = nonempty
        return v_lo, v_hi, v_ne
