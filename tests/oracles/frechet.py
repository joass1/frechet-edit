"""Independent discrete Frechet oracle, written from the coupling definition.

Source of truth: ``docs/definition.md`` section 2.

    A *coupling* of A (length a >= 1) and B (length b >= 1) is a sequence of
    index pairs starting at (0,0), ending at (a-1,b-1), where each successive
    pair advances the first index by 1, the second by 1, or both by 1.  Its
    cost is the maximum Euclidean distance over its pairs.  ``discrete_frechet``
    is the minimum cost over couplings.

Two implementations live here:

* :func:`enumerate_couplings` / :func:`brute_force_sq` enumerate every coupling
  literally.  Exponential, usable only for tiny sizes, but it is a direct
  transcription of the definition above.
* :func:`discrete_frechet` / :func:`discrete_frechet_le` are naive O(a*b)
  dynamic programs over the same monotone lattice.

``tests/oracles/test_oracles_selfcheck.py`` proves the two agree for all
``a, b <= 4``.  Everything else in this package depends on this file, so that
self-check is load bearing.

UNITS.  :func:`discrete_frechet` returns the exact SQUARED discrete Frechet
distance as a :class:`fractions.Fraction`.  Squaring is monotone on
non-negative reals, so the minimax over couplings of squared distances is the
square of the minimax over couplings of distances, and no square root (which
would be irrational and therefore inexact) is ever taken.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Any

from .exact import delta_sq, sq_dist

__all__ = [
    "brute_force_sq",
    "discrete_frechet",
    "discrete_frechet_le",
    "enumerate_couplings",
]

_STEPS = ((1, 0), (0, 1), (1, 1))


def _check_pair(A: Sequence[Any], B: Sequence[Any]) -> tuple[int, int]:
    a, b = len(A), len(B)
    if a == 0 and b == 0:
        # docs/definition.md section 2: empty vs empty is 0 but "occurs only as
        # an internal DP state, never as public input".  The oracle refuses it
        # so a caller bug cannot masquerade as a feasible answer.
        raise ValueError("discrete Frechet of two empty curves is not a public input")
    return a, b


def enumerate_couplings(a: int, b: int) -> Iterator[tuple[tuple[int, int], ...]]:
    """Yield every coupling of index grids of size ``a`` x ``b``.

    A coupling is a tuple of ``(i, j)`` pairs starting at ``(0, 0)``, ending at
    ``(a-1, b-1)``, each step advancing ``i``, ``j``, or both by exactly 1.
    Yields nothing when either length is 0 (infeasible, see definition).
    """
    if a <= 0 or b <= 0:
        return

    def walk(path: list[tuple[int, int]]) -> Iterator[tuple[tuple[int, int], ...]]:
        i, j = path[-1]
        if i == a - 1 and j == b - 1:
            yield tuple(path)
            return
        for di, dj in _STEPS:
            ni, nj = i + di, j + dj
            if ni < a and nj < b:
                path.append((ni, nj))
                yield from walk(path)
                path.pop()

    yield from walk([(0, 0)])


def brute_force_sq(A: Sequence[Any], B: Sequence[Any]) -> Fraction | None:
    """Exact squared discrete Frechet by literal enumeration of couplings.

    Returns ``None`` when infeasible (one curve empty and the other not).
    """
    a, b = _check_pair(A, B)
    if a == 0 or b == 0:
        return None
    best: Fraction | None = None
    for coupling in enumerate_couplings(a, b):
        cost = max(sq_dist(A[i], B[j]) for i, j in coupling)
        if best is None or cost < best:
            best = cost
    return best


def discrete_frechet(A: Sequence[Any], B: Sequence[Any]) -> Fraction | None:
    """Exact SQUARED discrete Frechet distance, or ``None`` when infeasible.

    Naive O(a*b) dynamic program over the monotone lattice of couplings:
    ``D[i][j]`` is the minimum over couplings from ``(0,0)`` to ``(i,j)`` of
    the maximum squared distance, which by the definition of the three legal
    steps satisfies

        D[i][j] = max( sq_dist(A[i], B[j]),
                       min(D[i-1][j], D[i][j-1], D[i-1][j-1]) )

    with ``D[0][0] = sq_dist(A[0], B[0])`` and out-of-range states absent.

    ``None`` is returned when exactly one curve is empty: ``discrete_frechet``
    of an empty curve against a non-empty curve is undefined and is treated as
    INFEASIBLE (``docs/definition.md`` section 2).
    """
    a, b = _check_pair(A, B)
    if a == 0 or b == 0:
        return None

    prev: list[Fraction] = []
    for i in range(a):
        cur: list[Fraction] = []
        for j in range(b):
            here = sq_dist(A[i], B[j])
            if i == 0 and j == 0:
                cur.append(here)
                continue
            best: Fraction | None = None
            if i > 0:
                best = prev[j]
            if j > 0 and (best is None or cur[j - 1] < best):
                best = cur[j - 1]
            if i > 0 and j > 0 and prev[j - 1] < best:
                best = prev[j - 1]
            assert best is not None
            cur.append(here if here > best else best)
        prev = cur
    return prev[b - 1]


def discrete_frechet_le(A: Sequence[Any], B: Sequence[Any], delta: Any) -> bool:
    """Exact CLOSED decision ``discrete_frechet(A, B) <= delta``.

    Naive O(a*b) reachability DP: ``reach[i][j]`` is true when some coupling
    prefix reaches ``(i, j)`` using only pairs within ``delta``.  Empty against
    non-empty is INFEASIBLE and returns ``False``.
    """
    a, b = _check_pair(A, B)
    if a == 0 or b == 0:
        return False
    limit = delta_sq(delta)

    prev: list[bool] = []
    for i in range(a):
        cur: list[bool] = []
        for j in range(b):
            if i == 0 and j == 0:
                reachable = True
            else:
                reachable = (
                    (i > 0 and prev[j])
                    or (j > 0 and cur[j - 1])
                    or (i > 0 and j > 0 and prev[j - 1])
                )
            cur.append(reachable and sq_dist(A[i], B[j]) <= limit)
        prev = cur
    return prev[b - 1]
