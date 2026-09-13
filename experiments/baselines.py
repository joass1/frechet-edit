"""Competing baselines for the Frechet Edit Distance evaluation.

This module is deliberately self-contained. It does NOT import anything from
`src/frechet_edit/`; every routine here is written directly from its published
definition so that agreement with the production code is real evidence.

Conventions shared by every function in this file
-------------------------------------------------
* A *curve* is a `(k, d)` array of finite float64 coordinates, `k >= 1`,
  `d >= 1`. Both curves in a pairwise call must share `d`.
* Distances are EUCLIDEAN (L2) in coordinate units. Lat/lon must be projected
  by the caller; nothing here applies haversine.
* Comparisons against a threshold are CLOSED: `<= eps` matches, `> eps` does
  not. No hidden epsilon widens or narrows the threshold.
* Inputs are never mutated. Every function returns fresh objects.
* NaN or infinite coordinates raise `ValueError` rather than propagating.

Cross-measure warning
---------------------
The quantities produced here are NOT mutually comparable. `discrete_frechet`,
`continuous_frechet`, `dtw`, `dtw_normalized` and `erp` are in coordinate
units (or sums of them); `edr` and `lcss` are counts; `lcss_distance` is a
ratio in [0, 1]. The Frechet Edit Distance is an EDIT COUNT. Any table that
places these side by side must label the units per column and must never
average or difference across columns.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "continuous_frechet",
    "continuous_frechet_le",
    "discrete_frechet",
    "dtw",
    "dtw_normalized",
    "edr",
    "erp",
    "lcss",
    "lcss_distance",
    "median_filter",
    "moving_average_filter",
]

# Maximum bisection steps for `continuous_frechet`. A guard against a `tol`
# finer than float64 can separate; never reached for sane tolerances.
_MAX_BISECTION_STEPS = 200

# Maximum relative nudges applied to the initial upper bound of the bisection
# if the (mathematically valid) bound is rejected by the decision procedure
# because of floating point round-off.
_MAX_UPPER_BOUND_NUDGES = 60


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _as_curve(curve, name: str) -> np.ndarray:
    """Validate and normalise a curve argument to a `(k, d)` float64 array."""
    try:
        arr = np.asarray(curve, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # non-numeric input
        raise ValueError(f"{name} must be a numeric (k, d) array") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (k, d) array, got ndim={arr.ndim}")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} must have at least one point")
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must have at least one coordinate per point")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite coordinates")
    return arr


def _as_curve_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Validate two curves and check that their dimensions agree."""
    arr_a = _as_curve(a, "A")
    arr_b = _as_curve(b, "B")
    if arr_a.shape[1] != arr_b.shape[1]:
        raise ValueError(
            f"A and B must share a dimension, got {arr_a.shape[1]} and {arr_b.shape[1]}"
        )
    return arr_a, arr_b


def _as_threshold(eps, name: str = "eps") -> float:
    """Validate a non-negative finite matching threshold."""
    value = float(eps)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {eps!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _row_distances(point: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Euclidean distance from one point to every vertex of `curve`."""
    return np.sqrt(np.sum((curve - point) ** 2, axis=1))


def _point_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return math.sqrt(float(np.sum((p - q) ** 2)))


# --------------------------------------------------------------------------
# discrete Frechet
# --------------------------------------------------------------------------


def discrete_frechet(A, B) -> float:
    """Discrete (strong / standard) Frechet distance between two curves.

    Definition
    ----------
    A *coupling* of A (length m >= 1) and B (length n >= 1) is a sequence of
    index pairs beginning at (0, 0), ending at (m-1, n-1), where each
    successive pair advances the first index by 1, the second by 1, or both by
    1. Its cost is the MAXIMUM Euclidean distance over its pairs, and
    `discrete_frechet` is the MINIMUM cost over all couplings. This is the
    definition in `docs/definition.md` section 2.

    Normalisation
    -------------
    None. The value is in coordinate units and is a max, not a sum, so it does
    NOT grow with curve length and must never be divided by a path length.

    Endpoint rule
    -------------
    Every coupling contains (0, 0) and (m-1, n-1), so the result is always
    `>= max(||A[0]-B[0]||, ||A[-1]-B[-1]||)`. Endpoints always couple; there is
    no free prefix or suffix.

    Complexity
    ----------
    O(m*n) time, O(min(m, n)) memory. The curves are swapped internally when
    that shortens the rolling row; the discrete Frechet distance is symmetric,
    so this cannot change the value.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    # Roll over the shorter curve so the retained row is O(min(m, n)).
    if curve_b.shape[0] > curve_a.shape[0]:
        curve_a, curve_b = curve_b, curve_a

    n = curve_b.shape[0]
    prev = np.empty(n, dtype=np.float64)
    cur = np.empty(n, dtype=np.float64)

    # Row i = 0: only rightward steps exist, so the cost is the running maximum
    # of the distances from A[0].
    prev[:] = np.maximum.accumulate(_row_distances(curve_a[0], curve_b))

    for i in range(1, curve_a.shape[0]):
        row = _row_distances(curve_a[i], curve_b)
        # Column j = 0: only downward steps exist.
        cur[0] = max(prev[0], row[0])
        for j in range(1, n):
            best_predecessor = prev[j]
            if cur[j - 1] < best_predecessor:
                best_predecessor = cur[j - 1]
            if prev[j - 1] < best_predecessor:
                best_predecessor = prev[j - 1]
            cur[j] = best_predecessor if best_predecessor > row[j] else row[j]
        prev, cur = cur, prev

    return float(prev[n - 1])


# --------------------------------------------------------------------------
# continuous (polygonal) Frechet
# --------------------------------------------------------------------------


def _free_interval(
    p: np.ndarray, q: np.ndarray, c: np.ndarray, eps: float
) -> tuple[float, float] | None:
    """Free-space interval of one free-space-diagram cell edge.

    Returns the closed interval of `u` in [0, 1] with
    `|| p + u*(q - p) - c || <= eps`, or None when it is empty. The set is an
    interval because the squared distance is a convex quadratic in `u`.

    Numerical guard: whichever of `u = 0`, `u = 1` satisfies the closed test on
    the raw distance is snapped into the interval. That keeps the endpoint rule
    exact when the discriminant loses precision, and it can only ever agree
    with the mathematically correct answer.
    """
    v = q - p
    w = p - c
    eps_sq = eps * eps
    c0 = float(np.sum(w * w))
    w1 = q - c
    c1 = float(np.sum(w1 * w1))
    start_free = c0 <= eps_sq
    end_free = c1 <= eps_sq

    a = float(np.sum(v * v))
    b = 2.0 * float(np.sum(v * w))
    cc = c0 - eps_sq

    if a == 0.0:  # degenerate (repeated) vertex: the edge collapses to a point
        return (0.0, 1.0) if cc <= 0.0 else None

    disc = b * b - 4.0 * a * cc
    if disc < 0.0:
        # Empty, unless round-off contradicts a demonstrably free endpoint.
        if start_free and end_free:
            return (0.0, 1.0)
        if start_free:
            return (0.0, 0.0)
        if end_free:
            return (1.0, 1.0)
        return None

    root = math.sqrt(disc)
    lo = (-b - root) / (2.0 * a)
    hi = (-b + root) / (2.0 * a)
    if lo < 0.0:
        lo = 0.0
    if hi > 1.0:
        hi = 1.0
    if start_free:
        lo = 0.0
    if end_free:
        hi = 1.0
    if lo > hi:
        return None
    return (lo, hi)


def continuous_frechet_le(A, B, eps) -> bool:
    """Alt-Godau decision: is the continuous Frechet distance `<= eps`?

    Definition
    ----------
    A and B are read as POLYGONAL CURVES (vertices joined by straight segments,
    each traversed with the natural linear parameterisation). The continuous
    Frechet distance is
    `inf over reparameterisations a, b of max_t || A(a(t)) - B(b(t)) ||`, the
    infimum taken over continuous non-decreasing surjections of [0, 1].

    Method
    ------
    Alt & Godau's free-space diagram. For cell `(i, j)` the free set is the
    intersection of an ellipse with the unit square and is therefore convex, so
    the reachable subset of every cell edge is a SUFFIX of that edge's free
    interval. Propagation rules, using convexity inside the cell:

        right edge  <- all of its free interval when the bottom edge is
                       reachable; otherwise the part at or above the lowest
                       reachable point of the left edge;
        top edge    <- all of its free interval when the left edge is
                       reachable; otherwise the part at or right of the
                       leftmost reachable point of the bottom edge.

    The boundary row and column are seeded from the prefix conditions
    `||A[k] - B[0]|| <= eps` and `||A[0] - B[k]|| <= eps`. Those are sufficient
    for the whole boundary segments because the distance from a fixed point to
    a moving point on a segment is convex, hence maximal at a vertex.

    Endpoint rule
    -------------
    `(0, 0)` and `(m-1, n-1)` must both be free, so `eps` is rejected whenever
    `max(||A[0]-B[0]||, ||A[-1]-B[-1]||) > eps`. Reparameterisations are
    surjective: no prefix or suffix may be skipped.

    Complexity
    ----------
    O(m*n) time, O(n) memory. The threshold test is closed (`<=`).
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    threshold = _as_threshold(eps)
    m = curve_a.shape[0]
    n = curve_b.shape[0]

    if _point_distance(curve_a[0], curve_b[0]) > threshold:
        return False
    if _point_distance(curve_a[-1], curve_b[-1]) > threshold:
        return False

    # Degenerate curves: a single point must stay within eps of the whole other
    # polyline. Distance from a fixed point to a point moving along a segment is
    # convex, so checking the other curve's VERTICES is exactly equivalent.
    if m == 1:
        return bool(np.all(_row_distances(curve_a[0], curve_b) <= threshold))
    if n == 1:
        return bool(np.all(_row_distances(curve_b[0], curve_a) <= threshold))

    # Boundary reachability: corner (i, 0) is reachable iff every A[k], k <= i,
    # is within eps of B[0]; symmetrically for corner (0, j).
    bottom_ok = np.logical_and.accumulate(
        _row_distances(curve_b[0], curve_a) <= threshold
    )
    left_ok = np.logical_and.accumulate(
        _row_distances(curve_a[0], curve_b) <= threshold
    )

    # left_row[j] is the reachable subset of the left edge of cell (i, j):
    # the edge {a = i} x [j, j+1], parameterised along B's segment j.
    left_row: list[tuple[float, float] | None] = [
        _free_interval(curve_b[j], curve_b[j + 1], curve_a[0], threshold)
        if left_ok[j]
        else None
        for j in range(n - 1)
    ]

    for i in range(m - 1):
        # bottom_row[j] is the reachable subset of the bottom edge of cell
        # (i, j): the edge [i, i+1] x {b = j}, along A's segment i.
        bottom_row: list[tuple[float, float] | None] = [None] * n
        if bottom_ok[i]:
            bottom_row[0] = _free_interval(
                curve_a[i], curve_a[i + 1], curve_b[0], threshold
            )
        next_left_row: list[tuple[float, float] | None] = [None] * (n - 1)

        for j in range(n - 1):
            left = left_row[j]
            bottom = bottom_row[j]
            if left is None and bottom is None:
                continue

            # right edge of cell (i, j) == left edge of cell (i+1, j)
            free = _free_interval(curve_b[j], curve_b[j + 1], curve_a[i + 1], threshold)
            if free is not None:
                if bottom is not None:
                    next_left_row[j] = free
                else:
                    lo = free[0] if free[0] > left[0] else left[0]
                    if lo <= free[1]:
                        next_left_row[j] = (lo, free[1])

            # top edge of cell (i, j) == bottom edge of cell (i, j+1)
            free = _free_interval(curve_a[i], curve_a[i + 1], curve_b[j + 1], threshold)
            if free is not None:
                if left is not None:
                    bottom_row[j + 1] = free
                else:
                    lo = free[0] if free[0] > bottom[0] else bottom[0]
                    if lo <= free[1]:
                        bottom_row[j + 1] = (lo, free[1])

        if i == m - 2:
            # The corner (m-1, n-1) is reachable iff the top end of the last
            # right edge, or the right end of the last top edge, is reachable.
            last_left = next_left_row[n - 2]
            if last_left is not None and last_left[1] >= 1.0:
                return True
            last_bottom = bottom_row[n - 1]
            return last_bottom is not None and last_bottom[1] >= 1.0

        left_row = next_left_row

    return False


def continuous_frechet(A, B, tol: float = 1e-9) -> float:
    """Continuous (polygonal) Frechet distance, accurate to `tol`.

    Definition
    ----------
    As in `continuous_frechet_le`: the curves are polylines and the infimum is
    over continuous non-decreasing surjective reparameterisations.

    Method (the documented choice)
    ------------------------------
    BISECTION ON `eps` between a valid lower and a valid upper bound, using the
    Alt-Godau decision procedure. It is NOT the exact critical-value parametric
    search of Alt & Godau (over the O(m*n^2 + m^2*n) critical values); that
    returns an exact value, this returns one accurate to `tol`.

    * lower bound `max(||A[0]-B[0]||, ||A[-1]-B[-1]||)`, forced by the endpoint
      rule;
    * upper bound `discrete_frechet(A, B)`, because a discrete coupling induces
      a valid pair of reparameterisations, so the continuous distance can never
      exceed the discrete one.

    The returned value is always a FEASIBLE eps (the upper end of the shrinking
    bracket). Consequently `continuous_frechet(A, B) <= discrete_frechet(A, B)`
    holds exactly, with no tolerance slack, and the true value always lies in
    `[result - tol, result]`.

    Normalisation
    -------------
    None. Coordinate units, a max rather than a sum.

    Endpoint rule
    -------------
    Endpoints are pinned, exactly as in the discrete case.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    tolerance = float(tol)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tol must be finite and strictly positive, got {tol!r}")

    lo = max(
        _point_distance(curve_a[0], curve_b[0]),
        _point_distance(curve_a[-1], curve_b[-1]),
    )
    hi = discrete_frechet(curve_a, curve_b)
    if hi < lo:  # only reachable through round-off
        hi = lo
    if hi <= lo:
        return float(hi)

    # `hi` is mathematically feasible; nudge it only if round-off says otherwise.
    nudges = 0
    while not continuous_frechet_le(curve_a, curve_b, hi):
        if nudges >= _MAX_UPPER_BOUND_NUDGES:
            raise RuntimeError(
                "continuous_frechet: could not establish a feasible upper bound"
            )
        hi = hi * (1.0 + 1e-12) + tolerance
        nudges += 1

    steps = 0
    while hi - lo > tolerance and steps < _MAX_BISECTION_STEPS:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:  # float64 resolution exhausted
            break
        if continuous_frechet_le(curve_a, curve_b, mid):
            hi = mid
        else:
            lo = mid
        steps += 1

    return float(hi)


# --------------------------------------------------------------------------
# dynamic time warping
# --------------------------------------------------------------------------


def _dtw_tables(curve_a: np.ndarray, curve_b: np.ndarray) -> tuple[float, int]:
    """Rolling DP returning (optimal DTW cost, length of the chosen path).

    Tie-break for the path length: when several predecessors attain the same
    minimum accumulated cost, the DIAGONAL (i-1, j-1) is preferred, then
    (i-1, j), then (i, j-1). The COST is unaffected by this choice; the
    reported path LENGTH is not, which is why the rule is fixed here and
    documented in `dtw_normalized`.
    """
    m = curve_a.shape[0]
    n = curve_b.shape[0]

    prev_cost = np.empty(n, dtype=np.float64)
    prev_len = np.empty(n, dtype=np.int64)
    cur_cost = np.empty(n, dtype=np.float64)
    cur_len = np.empty(n, dtype=np.int64)

    # Row i = 0: the only path runs rightwards, accumulating every cell.
    prev_cost[:] = np.cumsum(_row_distances(curve_a[0], curve_b))
    prev_len[:] = np.arange(1, n + 1, dtype=np.int64)

    for i in range(1, m):
        row = _row_distances(curve_a[i], curve_b)
        # Column j = 0: the only path runs downwards.
        cur_cost[0] = prev_cost[0] + row[0]
        cur_len[0] = prev_len[0] + 1
        for j in range(1, n):
            best = prev_cost[j - 1]  # diagonal (i-1, j-1)
            best_len = prev_len[j - 1]
            if prev_cost[j] < best:  # (i-1, j)
                best = prev_cost[j]
                best_len = prev_len[j]
            if cur_cost[j - 1] < best:  # (i, j-1)
                best = cur_cost[j - 1]
                best_len = cur_len[j - 1]
            cur_cost[j] = best + row[j]
            cur_len[j] = best_len + 1
        prev_cost, cur_cost = cur_cost, prev_cost
        prev_len, cur_len = cur_len, prev_len

    return float(prev_cost[n - 1]), int(prev_len[n - 1])


def dtw(A, B) -> float:
    """Dynamic Time Warping distance with a SUM path cost.

    Definition
    ----------
    A warping path is a sequence of index pairs from (0, 0) to (m-1, n-1)
    advancing the first index, the second, or both by one at each step (the
    same three moves as a discrete Frechet coupling). The local cost of a pair
    is the EUCLIDEAN distance between the two points, the path cost is the SUM
    of the local costs, and DTW is the minimum path cost over all paths.

    Normalisation
    -------------
    NONE: this is the raw sum. Being a sum it grows with curve length, so it is
    NOT comparable with any Frechet value, nor across curve pairs of different
    lengths. `dtw` and `dtw_normalized` are DIFFERENT measures and must be
    reported in separate, separately labelled columns - never mixed, averaged
    or substituted for one another.

    Endpoint rule
    -------------
    Endpoints are pinned: (0, 0) and (m-1, n-1) always lie on the path. No
    open-begin / open-end (subsequence) variant is implemented here.

    Complexity
    ----------
    O(m*n) time, O(n) memory.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    cost, _ = _dtw_tables(curve_a, curve_b)
    return cost


def dtw_normalized(A, B) -> float:
    """DTW divided by the number of pairs on the chosen optimal warping path.

    Definition
    ----------
    `dtw(A, B) / L`, where `L` is the number of index pairs on the optimal
    warping path (between `max(m, n)` and `m + n - 1` inclusive).

    Normalisation
    -------------
    Path length, as above: a length-normalised AVERAGE local cost. It is a
    DIFFERENT measure from `dtw`, not a monotone transform of it - the ranking
    of two curve pairs can differ between the two, and neither dominates the
    other. They must be REPORTED SEPARATELY, in their own columns. Neither is
    a metric (DTW violates the triangle inequality).

    When the optimal cost is attained by several paths of different lengths,
    `L` is the length of the path picked by a fixed tie-break: diagonal first,
    then (i-1, j), then (i, j-1). See `_dtw_tables`.

    Endpoint rule
    -------------
    Same pinned endpoints as `dtw`.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    cost, path_len = _dtw_tables(curve_a, curve_b)
    return cost / float(path_len)


# --------------------------------------------------------------------------
# EDR
# --------------------------------------------------------------------------


def edr(A, B, eps) -> int:
    """Edit Distance on Real sequences (Chen, Ozsu & Oria, SIGMOD 2005).

    Definition
    ----------
    A Levenshtein-shaped recurrence over prefix lengths `i`, `j`:

        E(i, 0) = i
        E(0, j) = j
        E(i, j) = min( E(i-1, j-1) + subcost(A[i-1], B[j-1]),
                       E(i-1, j)   + 1,
                       E(i,   j-1) + 1 )

    with `subcost = 0` when the two points MATCH and `1` otherwise; every gap
    (insertion or deletion) costs 1. The result is an integer count of edits.

    Match predicate (ambiguity resolved)
    ------------------------------------
    The original paper matches PER COORDINATE (`max_k |a_k - b_k| <= eps`, an
    L-infinity ball). This implementation uses the closed EUCLIDEAN ball
    `||a - b|| <= eps`, so that every baseline in this module uses one
    distance. For the same `eps` the Euclidean ball is contained in the
    L-infinity ball, so this variant is never MORE permissive than the paper's;
    for `d = 1` the two coincide. Report the choice alongside the numbers.

    Normalisation
    -------------
    NONE. The value is a raw count in `[0, max(m, n)]`. Divide by `max(m, n)`
    only if the table header says so explicitly.

    Endpoint rule
    -------------
    No endpoint constraint at all: EDR may gap either end at cost 1 per point.
    That is a real difference from the Frechet measures, where endpoints are
    pinned, and it is why EDR can be small for curves with distant endpoints.

    Complexity
    ----------
    O(m*n) time, O(n) memory.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    threshold = _as_threshold(eps)
    m = curve_a.shape[0]
    n = curve_b.shape[0]

    prev = np.arange(n + 1, dtype=np.int64)
    cur = np.empty(n + 1, dtype=np.int64)

    for i in range(1, m + 1):
        subcost = (_row_distances(curve_a[i - 1], curve_b) > threshold).astype(np.int64)
        cur[0] = i
        for j in range(1, n + 1):
            best = prev[j - 1] + subcost[j - 1]
            if prev[j] + 1 < best:
                best = prev[j] + 1
            if cur[j - 1] + 1 < best:
                best = cur[j - 1] + 1
            cur[j] = best
        prev, cur = cur, prev

    return int(prev[n])


# --------------------------------------------------------------------------
# LCSS
# --------------------------------------------------------------------------


def lcss(A, B, eps) -> int:
    """Length of the longest common subsequence under an `eps` match.

    Definition
    ----------
        L(i, 0) = L(0, j) = 0
        L(i, j) = L(i-1, j-1) + 1                  if ||A[i-1] - B[j-1]|| <= eps
                = max( L(i-1, j), L(i, j-1) )      otherwise

    The match predicate is the closed EUCLIDEAN ball, for the same reason as in
    `edr`. (The original LCSS trajectory work of Vlachos, Kollios & Gunopulos
    uses a per-coordinate box plus an optional warping window `delta`; the
    window is NOT applied here - this is the unwindowed variant.)

    Normalisation
    -------------
    NONE: a COUNT of matched pairs, in `[0, min(m, n)]`. Note that it is a
    SIMILARITY, larger being better, the opposite direction from every other
    function in this module. See `lcss_distance` for the dissimilarity form.

    Endpoint rule
    -------------
    No endpoint constraint: unmatched prefixes and suffixes are FREE, costing
    nothing at all (unlike `edr`, where they cost 1 per point).

    Complexity
    ----------
    O(m*n) time, O(n) memory.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    threshold = _as_threshold(eps)
    m = curve_a.shape[0]
    n = curve_b.shape[0]

    prev = np.zeros(n + 1, dtype=np.int64)
    cur = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, m + 1):
        matches = _row_distances(curve_a[i - 1], curve_b) <= threshold
        cur[0] = 0
        for j in range(1, n + 1):
            if matches[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev, cur = cur, prev

    return int(prev[n])


def lcss_distance(A, B, eps) -> float:
    """LCSS turned into a dissimilarity in [0, 1].

    Normalisation (stated explicitly)
    ---------------------------------
        lcss_distance = 1 - lcss(A, B, eps) / min(m, n)

    The denominator is `min(m, n)`, the largest achievable LCSS length, so the
    value is exactly 0 when one curve is fully matched inside the other and
    exactly 1 when nothing matches. The alternative denominator `max(m, n)`
    (which additionally penalises a length mismatch) is NOT used here; quote
    this formula whenever the number is reported.

    This is not a metric: it violates the triangle inequality, and it is 0 for
    distinct curves whenever one is an eps-subsequence of the other.

    Endpoint rule
    -------------
    Inherited from `lcss`: none, free prefixes and suffixes.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    matched = lcss(curve_a, curve_b, eps)
    denominator = min(curve_a.shape[0], curve_b.shape[0])
    return 1.0 - matched / float(denominator)


# --------------------------------------------------------------------------
# ERP
# --------------------------------------------------------------------------


def erp(A, B, g=None) -> float:
    """Edit distance with Real Penalty (Chen & Ng, VLDB 2004).

    Definition
    ----------
    With a fixed gap-reference point `g`:

        P(i, 0) = sum_{k<i} dist(A[k], g)
        P(0, j) = sum_{k<j} dist(B[k], g)
        P(i, j) = min( P(i-1, j-1) + dist(A[i-1], B[j-1]),
                       P(i-1, j)   + dist(A[i-1], g),
                       P(i,   j-1) + dist(B[j-1], g) )

    A gap is charged the distance from the skipped point to `g`, which is what
    restores the triangle inequality that DTW lacks.

    Gap distance (ambiguity resolved)
    ---------------------------------
    `dist` is the EUCLIDEAN (L2) distance throughout, for matches and for gaps
    alike, consistent with the rest of this module. The original paper works
    with L1 on per-coordinate series; for `d = 1` the two agree, for `d > 1`
    they do not. ERP is still a metric under L2, because L2 is a metric, which
    is the only property the construction needs.

    Gap reference
    -------------
    `g` defaults to the ORIGIN of the coordinate system (`zeros(d)`). The
    caller may pass any point of `R^d` - for example the pooled mean of the two
    curves, which makes the measure translation-invariant but also data
    dependent. Record which was used: results under different `g` are not
    comparable.

    Normalisation
    -------------
    NONE. A SUM in coordinate units; it grows with curve length.

    Endpoint rule
    -------------
    No endpoint constraint: leading and trailing points may be gapped, at the
    cost of their distance to `g`.

    Complexity
    ----------
    O(m*n) time, O(n) memory.
    """
    curve_a, curve_b = _as_curve_pair(A, B)
    d = curve_a.shape[1]
    if g is None:
        gap = np.zeros(d, dtype=np.float64)
    else:
        gap = np.asarray(g, dtype=np.float64).reshape(-1)
        if gap.shape[0] != d:
            raise ValueError(
                f"g must have {d} coordinates to match the curves, got {gap.shape[0]}"
            )
        if not np.all(np.isfinite(gap)):
            raise ValueError("g contains NaN or infinite coordinates")

    m = curve_a.shape[0]
    n = curve_b.shape[0]
    gap_a = _row_distances(gap, curve_a)  # dist(A[k], g)
    gap_b = _row_distances(gap, curve_b)  # dist(B[k], g)

    prev = np.empty(n + 1, dtype=np.float64)
    cur = np.empty(n + 1, dtype=np.float64)
    prev[0] = 0.0
    prev[1:] = np.cumsum(gap_b)

    for i in range(1, m + 1):
        row = _row_distances(curve_a[i - 1], curve_b)
        cur[0] = prev[0] + gap_a[i - 1]
        for j in range(1, n + 1):
            best = prev[j - 1] + row[j - 1]
            candidate = prev[j] + gap_a[i - 1]
            if candidate < best:
                best = candidate
            candidate = cur[j - 1] + gap_b[j - 1]
            if candidate < best:
                best = candidate
            cur[j] = best
        prev, cur = cur, prev

    return float(prev[n])


# --------------------------------------------------------------------------
# preregistered GPS smoothers
# --------------------------------------------------------------------------


def _validate_window(window) -> int:
    """Windows are odd positive integers so the filter stays centred."""
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"window must be an integer, got {window!r}")
    size = int(window)
    if size < 1:
        raise ValueError(f"window must be >= 1, got {size}")
    if size % 2 == 0:
        raise ValueError(
            f"window must be odd so the filter is centred and phase-free, got {size}"
        )
    return size


def _window_bounds(index: int, radius: int, length: int) -> tuple[int, int]:
    """Symmetrically SHRUNK window bounds (half-open) around `index`.

    Edge policy for both filters: the window is shrunk SYMMETRICALLY, never
    padded, reflected or extended. At index `i` the effective radius is
    `min(radius, i, length-1-i)`, so the window stays centred and odd-sized
    everywhere and the FIRST and LAST points come back unchanged. No fabricated
    sample ever enters the filter, and the endpoints - which every Frechet
    coupling pins - are preserved exactly.
    """
    effective = min(radius, index, length - 1 - index)
    return index - effective, index + effective + 1


def moving_average_filter(A, window) -> np.ndarray:
    """Centred moving-average smoother over a curve's coordinates.

    Definition
    ----------
    Output point `i` is the coordinate-wise ARITHMETIC MEAN of the input points
    inside a centred window of radius `(window - 1) // 2`.

    Window / edge policy
    --------------------
    `window` must be an ODD positive integer. At the boundaries the window is
    SHRUNK SYMMETRICALLY (see `_window_bounds`), never padded; `window = 1` is
    the identity. The first and last points are always returned unchanged.

    Guarantees
    ----------
    * The number of points is preserved exactly: output shape == input shape.
    * The input array is never mutated; a fresh float64 array is returned.
    * NaN or infinite coordinates raise `ValueError` instead of spreading
      across the window.
    """
    curve = _as_curve(A, "A")
    size = _validate_window(window)
    length = curve.shape[0]
    if size == 1:
        return curve.copy()

    radius = (size - 1) // 2
    out = np.empty_like(curve)
    for i in range(length):
        start, stop = _window_bounds(i, radius, length)
        out[i] = curve[start:stop].mean(axis=0)
    return out


def median_filter(A, window) -> np.ndarray:
    """Centred running-median smoother over a curve's coordinates.

    Definition
    ----------
    Output point `i` is the coordinate-wise MEDIAN of the input points inside a
    centred window of radius `(window - 1) // 2`. The median is taken PER
    COORDINATE, so for `d > 1` the result need not be one of the input points;
    this is the usual componentwise median filter, not a geometric median.

    Window / edge policy
    --------------------
    `window` must be an ODD positive integer. At the boundaries the window is
    SHRUNK SYMMETRICALLY (see `_window_bounds`), so every window holds an odd
    number of samples and the median is always an actual coordinate value, with
    no averaging of two middle elements. `window = 1` is the identity, and the
    first and last points are always returned unchanged.

    Guarantees
    ----------
    * The number of points is preserved exactly: output shape == input shape.
    * The input array is never mutated; a fresh float64 array is returned.
    * NaN or infinite coordinates raise `ValueError`.
    """
    curve = _as_curve(A, "A")
    size = _validate_window(window)
    length = curve.shape[0]
    if size == 1:
        return curve.copy()

    radius = (size - 1) // 2
    out = np.empty_like(curve)
    for i in range(length):
        start, stop = _window_bounds(i, radius, length)
        out[i] = np.median(curve[start:stop], axis=0)
    return out
