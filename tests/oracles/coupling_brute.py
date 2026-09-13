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

from tests.oracles.published_recurrence import dist, meb_within

INF = float("inf")


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
