"""Oracle C - exact minimum enclosing ball (``docs/oracle-protocol.md``).

Method, transcribed from the protocol:

    For ``d = 1``, the interval midpoint.  For ``d = 2``, enumerate every
    1-point, 2-point (diameter) and 3-point (circumcircle) candidate, keep the
    candidates that enclose all of S under exact rational comparison, and
    return the one of least squared radius.  All arithmetic is
    ``fractions.Fraction``, so both the decision and the centre are exact.

Correctness rests on the standard fact that the minimum enclosing ball of a
finite planar set is determined by a support set of at most 3 points, and is
the smallest enclosing candidate over that finite family.  Since the MEB of S
equals the MEB of its support set, and the MEB of a 1-, 2- or 3-point set is
one of the enumerated candidates (a point, a diameter circle, or a
circumcircle), the enumerated family contains the true MEB; every other
enumerated candidate that encloses S has radius at least that of the MEB, so
the minimum over enclosing candidates IS the MEB.

Notably the obtuse-triangle case is handled correctly for free: its
circumcircle is enumerated but so is its longest-side diameter, and the
diameter is smaller, so the minimum picks the diameter.  Naive
"circumcircle of the three extreme points" implementations get this wrong.

Cost ``O(|S|^4)`` rational operations; use is restricted to ``|S| <= 12``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from .exact import delta_sq, sq_dist, to_frac_point

__all__ = ["MAX_POINTS", "exact_meb", "meb_radius_le"]

MAX_POINTS = 12


def _normalise(points: Sequence[Any]) -> list[tuple[Fraction, ...]]:
    pts = [to_frac_point(p) for p in points]
    if not pts:
        raise ValueError("exact_meb requires at least one point")
    if len(pts) > MAX_POINTS:
        raise ValueError(
            f"exact_meb is restricted to |S| <= {MAX_POINTS}; got {len(pts)}"
        )
    dim = len(pts[0])
    for p in pts:
        if len(p) != dim:
            raise ValueError("all points must share the same dimension")
    if dim not in (1, 2):
        raise ValueError(
            f"exact_meb is implemented for d in (1, 2) only; got d={dim}"
        )
    return pts


def _encloses(
    centre: tuple[Fraction, ...],
    sq_radius: Fraction,
    pts: Sequence[tuple[Fraction, ...]],
) -> bool:
    return all(sq_dist(centre, p) <= sq_radius for p in pts)


def _circumcentre(
    p: tuple[Fraction, ...],
    q: tuple[Fraction, ...],
    r: tuple[Fraction, ...],
) -> tuple[Fraction, ...] | None:
    """Exact circumcentre of three planar rational points, or ``None``.

    ``None`` means the three points are collinear (this includes the case of
    duplicated points), so there is no circumcircle.  Such triples are simply
    skipped: the MEB of a collinear triple is the diameter of its two extreme
    points, and that 2-point candidate is enumerated separately.

    Solving the two perpendicular-bisector equations keeps every intermediate
    value rational, so the centre of a rational triple is itself rational.
    """
    (x1, y1), (x2, y2), (x3, y3) = p, q, r
    a1 = 2 * (x2 - x1)
    b1 = 2 * (y2 - y1)
    c1 = (x2 * x2 + y2 * y2) - (x1 * x1 + y1 * y1)
    a2 = 2 * (x3 - x1)
    b2 = 2 * (y3 - y1)
    c2 = (x3 * x3 + y3 * y3) - (x1 * x1 + y1 * y1)
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    cx = (c1 * b2 - c2 * b1) / det
    cy = (a1 * c2 - a2 * c1) / det
    return (cx, cy)


def _candidates_1d(
    pts: Sequence[tuple[Fraction, ...]],
) -> list[tuple[tuple[Fraction, ...], Fraction]]:
    lo = min(p[0] for p in pts)
    hi = max(p[0] for p in pts)
    centre = (lo + hi) / 2
    radius = (hi - lo) / 2
    return [((centre,), radius * radius)]


def _candidates_2d(
    pts: Sequence[tuple[Fraction, ...]],
) -> list[tuple[tuple[Fraction, ...], Fraction]]:
    out: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for p in pts:
        out.append((p, Fraction(0)))
    for p, q in itertools.combinations(pts, 2):
        centre = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        out.append((centre, sq_dist(centre, p)))
    for p, q, r in itertools.combinations(pts, 3):
        centre = _circumcentre(p, q, r)
        if centre is None:
            continue  # collinear or duplicated: covered by a diameter candidate
        out.append((centre, sq_dist(centre, p)))
    return out


def exact_meb(points: Sequence[Any]) -> tuple[tuple[Fraction, ...], Fraction]:
    """Return ``(centre, sq_radius)`` of the exact minimum enclosing ball.

    ``centre`` is a tuple of :class:`Fraction`, ``sq_radius`` is a
    :class:`Fraction`.  The radius itself is generally irrational, so only its
    square is reported and every downstream comparison is done squared.
    """
    pts = _normalise(points)
    dim = len(pts[0])
    candidates = _candidates_1d(pts) if dim == 1 else _candidates_2d(pts)

    best: tuple[tuple[Fraction, ...], Fraction] | None = None
    for centre, sq_radius in candidates:
        if not _encloses(centre, sq_radius, pts):
            continue
        if best is None or (sq_radius, centre) < (best[1], best[0]):
            # deterministic tie-break: least squared radius, then lexicographic
            # centre.  Coincident candidates have identical centres anyway.
            best = (centre, sq_radius)
    if best is None:  # pragma: no cover - the 1-point candidates always enclose
        raise AssertionError("no enclosing candidate found; enumeration is wrong")
    return best


def meb_radius_le(points: Sequence[Any], delta: Any) -> bool:
    """Exact CLOSED decision: MEB radius of ``points`` is ``<= delta``."""
    _, sq_radius = exact_meb(points)
    return sq_radius <= delta_sq(delta)
