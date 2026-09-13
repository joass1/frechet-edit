"""VERBATIM transcription of the recurrences as DISPLAYED in the published paper.

Transcribed directly from the PDF text, section 5.2 (IedDP) and 5.3 (edDP).
Deliberately NOT informed by the frechet_edit implementation: the point is to
test what the paper literally says, so any deviation would defeat the purpose.
See docs/errata-insertion-recurrence.md for the finding this supports.

Paper notation: pi = <pi_1..pi_m> is the curve matched against; sigma =
<sigma_1..sigma_n> is the curve edited (points inserted into it).
IedDP(i,j) := IedDF(pi[1,i], sigma[1,j]).
"""
import itertools
from fractions import Fraction

INF = float("inf")


def meb_radius_exact(points):
    """Exact minimum enclosing ball radius for a small point set (1-D or 2-D).

    Brute force over the defining subsets: for <=3 points in the plane the MEB
    is determined by a diameter pair or a circumcircle, so checking every pair
    and every triple is complete. Uses Fraction so the radius comparison is
    exact and cannot be blamed on floating point.
    """
    pts = [tuple(Fraction(c) for c in p) for p in points]
    if not pts:
        return Fraction(0)
    if len(pts) == 1:
        return Fraction(0)
    d = len(pts[0])
    if d == 1:
        lo = min(p[0] for p in pts)
        hi = max(p[0] for p in pts)
        return (hi - lo) / 2

    def d2(a, b):
        return sum((a[k] - b[k]) ** 2 for k in range(d))

    best = None
    # diameter pairs
    for a, b in itertools.combinations(pts, 2):
        c = tuple((a[k] + b[k]) / 2 for k in range(d))
        r2 = d2(a, c)
        if all(d2(p, c) <= r2 for p in pts) and (best is None or r2 < best):
            best = r2
    # circumcircles of triples
    for a, b, c in itertools.combinations(pts, 3):
        ax, ay = a
        bx, by = b
        cx, cy = c
        dd = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if dd == 0:
            continue
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
              + (cx**2 + cy**2) * (ay - by)) / dd
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
              + (cx**2 + cy**2) * (bx - ax)) / dd
        cen = (ux, uy)
        r2 = d2(a, cen)
        if all(d2(p, cen) <= r2 for p in pts) and (best is None or r2 < best):
            best = r2
    return best  # squared radius in 2-D; compared as squared below


def meb_within(points, delta):
    """Is the minimum enclosing ball radius of `points` at most delta?"""
    if len(points) <= 1:
        return True
    d = len(points[0])
    r = meb_radius_exact(points)
    if d == 1:
        return r <= Fraction(delta)
    return r <= Fraction(delta) ** 2  # squared comparison in 2-D


def mu_indices(pi, delta):
    """mu(i) = smallest t in {1..i} with MEB radius of <pi_t..pi_i> <= delta.

    Paper, section 5.2. 1-indexed; returns a dict i -> mu(i).
    """
    mu = {}
    for i in range(1, len(pi) + 1):
        t = i
        while t >= 1 and meb_within(pi[t - 1:i], delta):
            t -= 1
        mu[i] = t + 1
    return mu


def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def paper_insertion_dp(pi, sigma, delta):
    """IedDP exactly as displayed in section 5.2 of arXiv:2403.12878v1."""
    m, n = len(pi), len(sigma)
    mu = mu_indices(pi, delta)
    F = [[INF] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                F[i][j] = 0
            elif i == 0 and j >= 1:
                F[i][j] = INF
            elif (i >= 1 and j == 0) or (
                i >= 1 and j >= 1 and dist(sigma[j - 1], pi[i - 1]) > delta
            ):
                F[i][j] = 1 + min(F[k - 1][j] for k in range(mu[i], i + 1))
            else:
                F[i][j] = min(
                    F[i][j - 1],
                    F[i - 1][j],
                    F[i - 1][j - 1],
                    1 + min(F[k - 1][j] for k in range(mu[i], i + 1)),
                )
    return F[m][n]


def paper_mixed_dp(pi, sigma, delta):
    """edDP exactly as displayed in section 5.3 of arXiv:2403.12878v1."""
    m, n = len(pi), len(sigma)
    mu = mu_indices(pi, delta)
    F = [[INF] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                F[i][j] = 0
            elif i == 0 and j >= 1:
                F[i][j] = 1 + F[i][j - 1]
            elif i >= 1 and j == 0:
                F[i][j] = 1 + min(F[k - 1][j] for k in range(mu[i], i + 1))
            elif dist(sigma[j - 1], pi[i - 1]) > delta:
                F[i][j] = min(
                    1 + F[i][j - 1],
                    1 + min(F[k - 1][j] for k in range(mu[i], i + 1)),
                )
            else:
                F[i][j] = min(
                    F[i][j - 1],
                    F[i - 1][j],
                    F[i - 1][j - 1],
                    1 + min(F[k - 1][j] for k in range(mu[i], i + 1)),
                )
    return F[m][n]


def paper_deletion_dp(pi, sigma, delta):
    """DedDP exactly as displayed in section 5.1.

    Note there is no insertion branch here, so no predecessor can ever "end
    with an inserted point". That is precisely why this form is sound while
    the insertion and mixed forms are not.
    """
    m, n = len(pi), len(sigma)
    F = [[INF] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                F[i][j] = 0
            elif i >= 1 and j == 0:
                F[i][j] = INF
            elif (i == 0 and j >= 1) or (
                i >= 1 and j >= 1 and dist(sigma[j - 1], pi[i - 1]) > delta
            ):
                F[i][j] = 1 + F[i][j - 1]
            else:
                F[i][j] = min(F[i][j - 1], F[i - 1][j], F[i - 1][j - 1])
    return F[m][n]


def repaired_insertion_dp(pi, sigma, delta):
    """The published recurrence with ONE term changed, to isolate the cause.

    K(i,j) is the best solution whose edited prefix ends with the kept point
    sigma_j; P(i,j) the best ending with an inserted point; X = min(K, P) is the
    paper's IedDP. Only the vertical predecessor of the keep branch differs from
    the published form: K(i-1, j) rather than X(i-1, j). Coupling pi_i to sigma_j
    requires sigma_j to still be the last element of the edited prefix, which a
    predecessor ending in an inserted point cannot offer.

    See docs/errata-insertion-recurrence.md section 6.1.
    """
    m, n = len(pi), len(sigma)
    mu = mu_indices(pi, delta)
    K = [[INF] * (n + 1) for _ in range(m + 1)]
    P = [[INF] * (n + 1) for _ in range(m + 1)]
    X = [[INF] * (n + 1) for _ in range(m + 1)]
    X[0][0] = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                X[i][j] = 0 if j == 0 else INF
                continue
            P[i][j] = 1 + min(X[k - 1][j] for k in range(mu[i], i + 1))
            if j >= 1 and dist(sigma[j - 1], pi[i - 1]) <= delta:
                K[i][j] = min(X[i][j - 1], K[i - 1][j], X[i - 1][j - 1])
            X[i][j] = min(K[i][j], P[i][j])
    return X[m][n]
