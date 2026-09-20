"""Independent brute force for discrete Frechet edit distance.

Written from the DEFINITION only, with no dynamic program anywhere: it
enumerates edited curves and monotone couplings directly. This exists so that
the paper's recurrence and the frechet_edit package can each be checked against
something that shares no recurrence with either.

d_dF(A,B) <= delta iff a monotone staircase path exists in the grid from (1,1)
to (m,M) with every visited cell within delta. In such a path, column l covers a
contiguous non-empty row range [lo_l, hi_l], with lo_1 = 1, hi_M = m, and
lo_{l+1} in {hi_l, hi_l + 1}. An inserted point is free in R^d, so its column is
satisfiable exactly when the minimum enclosing ball of its row range fits in
delta; a kept point must itself be within delta of every row it covers.
"""
import itertools
from fractions import Fraction

INF = float("inf")

# Deliberately self-contained. An earlier revision imported `dist` and
# `meb_within` from `published_recurrence`, which meant the two oracles the
# erratum calls independent shared their geometric primitives. That never
# touched the recurrence logic under test, so it could not have manufactured
# the erratum's result - but "shares no recurrence with either" read as a
# stronger independence claim than was actually true. These primitives are now
# written from scratch here, so the claim is literal.


def _sq_dist(a, b):
    """Exact squared distance. Rational throughout: no tolerance, no rounding."""
    return sum((Fraction(x) - Fraction(y)) ** 2 for x, y in zip(a, b, strict=True))


def dist(a, b):
    """Euclidean distance as a float, for comparisons against a float delta."""
    return float(_sq_dist(a, b)) ** 0.5


def _covering_point_exists(block, delta):
    """Is there a point within ``delta`` of every point of ``block``?

    Equivalent to "the minimum enclosing ball has radius <= delta", but derived
    here the other way round - directly as the feasibility question the
    insertion step actually asks - so that it shares no code with the enclosing
    ball routines elsewhere in the tree.

    Complete by the same defining-subset argument used in `_meb`: the smallest
    covering ball is pinned by at most ``d + 1`` points of the block, so
    enumerating those subsets, taking the ball each one determines, and keeping
    any that covers the whole block is exhaustive. Solved in exact rationals by
    Gaussian elimination on the equidistance conditions.
    """
    points = [tuple(Fraction(c) for c in p) for p in block]
    unique = list(dict.fromkeys(points))
    if len(unique) == 1:
        return True
    dim = len(unique[0])
    limit = Fraction(delta) ** 2

    def covers(centre):
        return all(
            sum((c - p) ** 2 for c, p in zip(centre, point, strict=True)) <= limit
            for point in unique
        )

    for size in range(1, min(dim + 1, len(unique)) + 1):
        for subset in itertools.combinations(unique, size):
            centre = _equidistant_point(subset)
            if centre is not None and covers(centre):
                return True
    return False


def _equidistant_point(subset):
    """The point equidistant from every member of ``subset``, in their affine
    hull, or ``None`` when they do not determine one."""
    base = subset[0]
    if len(subset) == 1:
        return base
    vectors = [
        tuple(a - b for a, b in zip(p, base, strict=True)) for p in subset[1:]
    ]
    size = len(vectors)
    rows = [
        [
            2 * sum((a * b for a, b in zip(u, v, strict=True)), Fraction(0))
            for v in vectors
        ]
        + [sum((c * c for c in u), Fraction(0))]
        for u in vectors
    ]
    # Gauss-Jordan, exact.
    for col in range(size):
        pivot = next((r for r in range(col, size) if rows[r][col] != 0), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        lead = rows[col][col]
        rows[col] = [v / lead for v in rows[col]]
        for r in range(size):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [
                    v - factor * w for v, w in zip(rows[r], rows[col], strict=True)
                ]
    centre = list(base)
    for coeff, vector in zip([rows[i][size] for i in range(size)], vectors, strict=True):
        for k, component in enumerate(vector):
            centre[k] += coeff * component
    return tuple(centre)


def meb_within(block, delta):
    """Kept for the call site below; now backed by this module's own geometry."""
    if len(block) <= 1:
        return True
    return _covering_point_exists(block, delta)


def _columns_feasible(pi, cols, delta):
    """cols: list of (kind, value) where kind is 'keep' (value=point) or 'ins'."""
    m = len(pi)
    M = len(cols)
    if M == 0:
        return m == 0

    def rec(l, lo):
        # assign column l (0-based) a range starting at row lo (1-based)
        if l == M - 1:
            hi = m
            if hi < lo:
                return False
            return _col_ok(pi, cols[l], lo, hi, delta)
        for hi in range(lo, m + 1):
            if not _col_ok(pi, cols[l], lo, hi, delta):
                continue
            for nxt in (hi, hi + 1):
                if nxt > m:
                    continue
                if rec(l + 1, nxt):
                    return True
        return False

    return rec(0, 1)


def _col_ok(pi, col, lo, hi, delta):
    kind, val = col
    block = pi[lo - 1:hi]
    if not block:
        return False
    if kind == "ins":
        return meb_within(block, delta)
    return all(dist(val, p) <= delta for p in block)


def brute_edit_distance(pi, sigma, delta, mode):
    """Minimum edits; INF if no edited curve reaches within delta."""
    m, n = len(pi), len(sigma)
    best = INF
    keep_sets = ([tuple(range(n))] if mode == "insert"
                 else [s for r in range(n + 1)
                       for s in itertools.combinations(range(n), r)])
    max_ins = 0 if mode == "delete" else m
    for keep in keep_sets:
        deleted = n - len(keep)
        if deleted >= best:
            continue
        kept_pts = [sigma[i] for i in keep]
        for ins in range(max_ins + 1):
            cost = deleted + ins
            if cost >= best:
                break
            M = len(kept_pts) + ins
            if M == 0:
                if m == 0:
                    best = min(best, cost)
                continue
            for ins_pos in itertools.combinations(range(M), ins):
                cols, it = [], iter(kept_pts)
                ins_set = set(ins_pos)
                for l in range(M):
                    cols.append(("ins", None) if l in ins_set
                                else ("keep", next(it)))
                if _columns_feasible(pi, cols, delta):
                    best = min(best, cost)
                    break
    return best
