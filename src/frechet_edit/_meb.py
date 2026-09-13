"""Exact minimum enclosing ball in arbitrary dimension.

``_geometry`` keeps fast hand-written paths for d = 1 and d = 2, which is where
almost all real use sits. This module supplies the general case, so that
insertion-capable modes are not restricted to the plane.

Method, and why this one
------------------------
Not Welzl. Welzl's recursion assumes the boundary set stays affinely
independent, and its base case needs the ball with a given set ON its boundary;
both are awkward exactly where this package is most careful - collinear,
cocircular and cospherical blocks, which are deliberate test fixtures here.

Instead this is primal support refinement, which rests on a one-line fact:

    If ``S`` is a subset of ``P`` and ``MEB(S)`` already encloses all of ``P``,
    then ``MEB(S) = MEB(P)``.

    Proof. ``MEB(P)`` must enclose ``S``, so ``radius(MEB(P)) >= radius(MEB(S))``.
    ``MEB(S)`` encloses ``P``, so ``radius(MEB(P)) <= radius(MEB(S))``. The two
    radii are equal, and the minimum enclosing ball is unique, so the balls are
    equal. []

So: maintain a small ``S``, compute ``MEB(S)`` exactly by enumerating the
subsets that could define it, and if some point of ``P`` lies outside, add it to
``S`` and repeat. Each round strictly increases ``radius(MEB(S))`` - the point
added was outside the previous ball - so no support set can recur and the loop
terminates. ``S`` is pruned back to the support of ``MEB(S)`` each round, so it
never exceeds ``d + 2`` points and the enumeration stays small.

Why skipping affinely dependent subsets is safe
-----------------------------------------------
``circumball`` returns ``None`` for an affinely dependent set, because such a
set determines no unique ball, and ``meb_of_small`` then skips it. That loses
nothing. The centre of ``MEB(P)`` lies in the convex hull of the points on its
boundary; by Carathéodory it lies in the hull of at most ``d + 1`` of them, and
such a minimal subset ``T`` is affinely independent. Every point of ``T`` is on
the boundary and the centre lies in ``conv(T) subset of aff(T)``, so the ball is
exactly ``circumball(T)`` - which the enumeration does examine. So the minimum
is always realised by a subset that is not skipped.

Everything here is exact rational arithmetic. There is no tolerance anywhere,
so degeneracy is ordinary rather than special.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from fractions import Fraction
from typing import NamedTuple

FracPoint = tuple[Fraction, ...]


class FracBall(NamedTuple):
    """An exactly represented ball."""

    centre: tuple[Fraction, ...]
    sq_radius: Fraction


def sq_dist(a: Sequence[Fraction], b: Sequence[Fraction]) -> Fraction:
    return sum(((x - y) * (x - y) for x, y in zip(a, b, strict=True)), Fraction(0))


def encloses(ball: FracBall, points: Sequence[FracPoint]) -> bool:
    return all(sq_dist(p, ball.centre) <= ball.sq_radius for p in points)


def ball_from_one(p: FracPoint) -> FracBall:
    return FracBall(tuple(p), Fraction(0))


def ball_from_two(a: FracPoint, b: FracPoint) -> FracBall:
    centre = tuple((x + y) / 2 for x, y in zip(a, b, strict=True))
    return FracBall(centre, sq_dist(a, centre))


def solve_exact(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction] | None:
    """Solve a square rational system by Gauss-Jordan. ``None`` when singular.

    Exact throughout, so "singular" means genuinely singular rather than
    ill-conditioned, and an affinely dependent point set is detected rather than
    producing a wildly wrong centre.
    """
    n = len(matrix)
    aug = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = aug[col][col]
        aug[col] = [v / inv for v in aug[col]]
        for r in range(n):
            if r == col or aug[r][col] == 0:
                continue
            factor = aug[r][col]
            aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col], strict=True)]
    return [aug[i][n] for i in range(n)]


def circumball(points: Sequence[FracPoint]) -> FracBall | None:
    """The ball with every point of ``points`` on its boundary.

    The centre is constrained to the affine hull of the points, which makes it
    unique when they are affinely independent. Returns ``None`` when they are
    not, since then no such ball is determined.

    With ``p0`` as origin and ``v_j = p_j - p0``, a centre ``x = p0 + sum a_i v_i``
    is equidistant from every ``p_j`` exactly when
    ``2 * sum_i a_i (v_j . v_i) = |v_j|^2`` for each ``j``, which is the square
    rational system solved here.
    """
    if not points:
        return None
    if len(points) == 1:
        return ball_from_one(points[0])
    if len(points) == 2:
        return ball_from_two(points[0], points[1])

    base = points[0]
    vectors = [
        tuple(x - y for x, y in zip(p, base, strict=True)) for p in points[1:]
    ]
    gram = [
        [2 * sum((a * b for a, b in zip(vi, vj, strict=True)), Fraction(0))
         for vj in vectors]
        for vi in vectors
    ]
    rhs = [sum((c * c for c in vi), Fraction(0)) for vi in vectors]
    solution = solve_exact(gram, rhs)
    if solution is None:
        return None

    offset = [Fraction(0)] * len(base)
    for coeff, vec in zip(solution, vectors, strict=True):
        for k, component in enumerate(vec):
            offset[k] += coeff * component
    centre = tuple(b + o for b, o in zip(base, offset, strict=True))
    return FracBall(centre, sq_dist(points[0], centre))


def meb_of_small(points: Sequence[FracPoint], dim: int) -> FracBall:
    """Exact minimum enclosing ball of a small set, by defining-subset search.

    The minimum enclosing ball of any set is determined by at most ``d + 1`` of
    its points, so enumerating subsets up to that size and keeping the smallest
    ball that encloses everything is complete. Intended for sets of at most
    ``d + 2`` points, where this is cheap.
    """
    if not points:
        raise ValueError("meb_of_small requires at least one point")
    unique = list(dict.fromkeys(points))
    if len(unique) == 1:
        return ball_from_one(unique[0])

    best: FracBall | None = None
    for size in range(1, min(dim + 1, len(unique)) + 1):
        for subset in itertools.combinations(unique, size):
            candidate = circumball(subset)
            if candidate is None:
                continue
            if not encloses(candidate, unique):
                continue
            if best is None or candidate.sq_radius < best.sq_radius:
                best = candidate
    if best is None:  # pragma: no cover - some subset always encloses
        raise AssertionError("no enclosing candidate found")
    return best


def support_of(ball: FracBall, points: Sequence[FracPoint]) -> list[FracPoint]:
    """Points lying exactly on the ball's boundary."""
    return [p for p in points if sq_dist(p, ball.centre) == ball.sq_radius]


def exact_meb_nd(
    points: Sequence[FracPoint],
    dim: int,
    *,
    seed_indices: Sequence[int] | None = None,
    max_rounds: int = 256,
) -> FracBall:
    """Exact minimum enclosing ball in dimension ``dim``.

    ``seed_indices`` optionally supplies a starting support set, normally the
    one proposed by a float64 pass. A good seed saves rounds; a bad one costs
    rounds and never costs correctness, because every ball is recomputed
    exactly and the termination test is exact.

    Raises ``RuntimeError`` if ``max_rounds`` is exhausted, which the caller
    should surface as an abstention rather than a guess.
    """
    unique = list(dict.fromkeys(points))
    if not unique:
        raise ValueError("exact_meb_nd requires at least one point")
    if len(unique) == 1:
        return ball_from_one(unique[0])

    if seed_indices:
        seen = {points[i] for i in seed_indices if 0 <= i < len(points)}
        working = list(seen) or [unique[0]]
    else:
        working = [unique[0]]

    for _ in range(max_rounds):
        ball = meb_of_small(working, dim)
        outlier = next(
            (p for p in unique if sq_dist(p, ball.centre) > ball.sq_radius), None
        )
        if outlier is None:
            # ball = MEB(working), working subset of points, and it encloses
            # every point, so by the lemma in the module docstring it is MEB.
            return ball
        working = support_of(ball, working)
        working.append(outlier)
    raise RuntimeError(
        f"exact minimum enclosing ball did not converge in {max_rounds} rounds"
    )
