"""Independent verification of an :class:`~frechet_edit._types.EditResult`.

This module deliberately shares NOTHING with the solver beyond input
validation and the certified point predicate. It does not import the dynamic
program, the minimum queue, or the traceback, and it never trusts a stored
value: it replays the edits itself, recounts them, re-checks every coupling
step, and recomputes ordinary discrete Frechet on the edited curve with its own
plain implementation.

See ``docs/witness-invariants.md`` section 5.

Continuous results (``backend="continuous"``) are checked by
:func:`verify_continuous_witness`, which decides ``d_F <= delta`` on the edited
curve with :func:`continuous_frechet_le`: a plain Alt-Godau free-space sweep in
exact rational arithmetic, written separately from ``_continuous`` and
``_freespace`` and using a different exact comparison technique (see its
docstring). It shares no code with the continuous solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ._numerics import NumericPolicy, exact_within, within
from ._types import Deletion, EditResult, Insertion


@dataclass(slots=True)
class VerificationReport:
    """Outcome of verification. ``ok`` is true only if ``violations`` is empty."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    replayed_curve: np.ndarray | None = None
    ordinary_frechet: float | None = None

    def __bool__(self) -> bool:
        return self.ok

    def raise_for_status(self) -> None:
        if not self.ok:
            raise AssertionError("witness verification failed:\n  " + "\n  ".join(self.violations))


def replay(
    observation: np.ndarray,
    edits: tuple[Deletion | Insertion, ...],
) -> np.ndarray:
    """Apply ``edits`` to the ORIGINAL observation.

    Indices never shift: deleting index 3 does not renumber index 4. Insertion
    gap ``g`` places a point immediately before original index ``g``; ``g == n``
    places it after the last original vertex. Ordering is ascending gap, then
    ascending order within a gap, and that is retained even when every original
    vertex is deleted.
    """
    observation = np.asarray(observation, dtype=np.float64)
    n = len(observation)

    deleted: set[int] = set()
    for edit in edits:
        if isinstance(edit, Deletion):
            if not 0 <= edit.index < n:
                raise ValueError(f"deletion index {edit.index} out of range 0..{n - 1}")
            if edit.index in deleted:
                raise ValueError(f"observation index {edit.index} deleted more than once")
            deleted.add(edit.index)

    by_gap: dict[int, list[Insertion]] = {}
    for edit in edits:
        if isinstance(edit, Insertion):
            if not 0 <= edit.gap <= n:
                raise ValueError(f"insertion gap {edit.gap} out of range 0..{n}")
            by_gap.setdefault(edit.gap, []).append(edit)
    for gap, group in by_gap.items():
        orders = sorted(ins.order for ins in group)
        if orders != list(range(len(orders))):
            raise ValueError(f"insertions in gap {gap} have non-contiguous order values {orders}")

    out: list[np.ndarray] = []
    dim = observation.shape[1]
    for gap in range(n + 1):
        for ins in sorted(by_gap.get(gap, []), key=lambda e: e.order):
            point = np.asarray(ins.point, dtype=np.float64)
            if point.shape != (dim,):
                raise ValueError(f"inserted point {ins.point!r} is not {dim}-dimensional")
            out.append(point)
        if gap < n and gap not in deleted:
            out.append(observation[gap])
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, dim), dtype=np.float64)


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Ordinary strong discrete Frechet distance, written straight from the definition.

    A separate, deliberately simple implementation used only to cross-check
    witnesses. Empty against non-empty is ``inf``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0 if len(a) == len(b) else float("inf")

    prev = [float("inf")] * len(b)
    for i in range(len(a)):
        cur = [float("inf")] * len(b)
        for j in range(len(b)):
            d = float(np.linalg.norm(a[i] - b[j]))
            if i == 0 and j == 0:
                cur[j] = d
            elif i == 0:
                cur[j] = max(cur[j - 1], d)
            elif j == 0:
                cur[j] = max(prev[j], d)
            else:
                cur[j] = max(min(prev[j], prev[j - 1], cur[j - 1]), d)
        prev = cur
    return prev[-1]


def _check_coupling(
    reference: np.ndarray,
    edited: np.ndarray,
    coupling: tuple[tuple[int, int], ...],
    delta: float,
    policy: NumericPolicy,
    violations: list[str],
) -> None:
    m, k = len(reference), len(edited)
    if not coupling:
        violations.append("coupling is empty")
        return
    if coupling[0] != (0, 0):
        violations.append(f"coupling starts at {coupling[0]}, expected (0, 0)")
    if coupling[-1] != (m - 1, k - 1):
        violations.append(f"coupling ends at {coupling[-1]}, expected {(m - 1, k - 1)}")

    for idx, (ri, ei) in enumerate(coupling):
        if not (0 <= ri < m and 0 <= ei < k):
            violations.append(f"coupling pair {idx} = {(ri, ei)} is out of range")
            return

    for idx in range(1, len(coupling)):
        pri, pei = coupling[idx - 1]
        ri, ei = coupling[idx]
        step = (ri - pri, ei - pei)
        if step not in {(1, 0), (0, 1), (1, 1)}:
            violations.append(
                f"coupling step {idx} is {step} from {(pri, pei)} to {(ri, ei)}; "
                "only (1,0), (0,1) and (1,1) are legal"
            )

    for idx, (ri, ei) in enumerate(coupling):
        if not within(reference[ri], edited[ei], delta, policy=policy):
            dist = float(np.linalg.norm(reference[ri] - edited[ei]))
            violations.append(
                f"coupling pair {idx} = {(ri, ei)} has distance {dist!r} > delta {delta!r}"
            )


def verify_witness(
    reference: np.ndarray,
    observation: np.ndarray,
    result: EditResult,
    *,
    policy: NumericPolicy = "certified",
    check_ordinary_frechet: bool = True,
) -> VerificationReport:
    """Re-derive and check every claim a witness makes.

    A passing report establishes FEASIBILITY at the reported cost. It does not
    establish minimality; that comes from oracle agreement.

    A result from :func:`~frechet_edit.continuous_edit_distance` is routed to
    :func:`verify_continuous_witness`, since its witness has no discrete
    coupling to check.
    """
    if result.backend == "continuous":
        return verify_continuous_witness(reference, observation, result)
    violations: list[str] = []
    reference = np.asarray(reference, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)

    if result.witness_status != "certified":
        return VerificationReport(
            ok=False,
            violations=[f"witness_status is {result.witness_status!r}, nothing to verify"],
        )
    if result.edits is None or result.edited_curve is None or result.coupling is None:
        return VerificationReport(ok=False, violations=["witness fields are incomplete"])

    try:
        replayed = replay(observation, result.edits)
    except ValueError as exc:
        return VerificationReport(ok=False, violations=[f"replay failed: {exc}"])

    if replayed.shape != result.edited_curve.shape or not np.array_equal(
        replayed, result.edited_curve
    ):
        violations.append(
            f"replayed curve (shape {replayed.shape}) does not match the reported "
            f"edited_curve (shape {result.edited_curve.shape})"
        )

    if result.cost is None or len(result.edits) != result.cost:
        violations.append(f"witness has {len(result.edits)} edits but cost is {result.cost!r}")

    n_ins = sum(isinstance(e, Insertion) for e in result.edits)
    n_del = sum(isinstance(e, Deletion) for e in result.edits)
    if result.mode == "delete" and n_ins:
        violations.append(f"mode 'delete' but witness contains {n_ins} insertion(s)")
    if result.mode == "insert" and n_del:
        violations.append(f"mode 'insert' but witness contains {n_del} deletion(s)")

    deleted = {e.index for e in result.edits if isinstance(e, Deletion)}
    kept = [observation[idx] for idx in range(len(observation)) if idx not in deleted]
    pos = 0
    for point in replayed:
        if pos < len(kept) and np.array_equal(point, kept[pos]):
            pos += 1
    if pos != len(kept):
        violations.append("retained original vertices do not appear unchanged and in order")

    if len(replayed) == 0:
        violations.append("edited curve is empty, so no coupling with a non-empty reference exists")
    else:
        _check_coupling(reference, replayed, result.coupling, result.delta, policy, violations)

    ordinary: float | None = None
    if check_ordinary_frechet and len(replayed) > 0:
        ordinary = discrete_frechet(reference, replayed)
        if not exact_within([ordinary], [0.0], result.delta):
            violations.append(
                f"independent discrete Frechet of the edited curve is {ordinary!r}, "
                f"which exceeds delta {result.delta!r}"
            )

    return VerificationReport(
        ok=not violations,
        violations=violations,
        replayed_curve=replayed,
        ordinary_frechet=ordinary,
    )


# ---------------------------------------------------------------------------
# Continuous Frechet: an exact decision procedure independent of the solver.
# ---------------------------------------------------------------------------

_Pt = tuple[Fraction, ...]
#: The real number ``x + s * sqrt(y)``, as the triple ``(x, s, y)``.
_Alg = tuple[Fraction, int, Fraction]
_Span = tuple[_Alg, _Alg]
_A_ZERO: _Alg = (Fraction(0), 0, Fraction(0))
_A_ONE: _Alg = (Fraction(1), 0, Fraction(0))
_REFINE_LIMIT_BITS = 16384


def _alg_normal(r: _Alg) -> _Alg:
    """Fold a rational square root into ``x``, so irrational roots stay distinct."""
    x, s, y = r
    if s == 0 or y == 0:
        return (x, 0, Fraction(0))
    num, den = y.numerator, y.denominator
    rn, rd = math.isqrt(num), math.isqrt(den)
    if rn * rn == num and rd * rd == den:
        return (x + s * Fraction(rn, rd), 0, Fraction(0))
    return r


def _alg_bounds(r: _Alg, bits: int) -> tuple[Fraction, Fraction]:
    x, s, y = r
    if s == 0:
        return x, x
    root = math.isqrt((y.numerator * y.denominator) << (2 * bits))
    scale = y.denominator << bits
    lo, hi = Fraction(root, scale), Fraction(root + 1, scale)
    return (x + lo, x + hi) if s > 0 else (x - hi, x - lo)


def _alg_cmp(r1: _Alg, r2: _Alg) -> int:
    """Exact order of two ``x + s*sqrt(y)`` numbers.

    Equality: with irrational ``sqrt(y1)`` and ``sqrt(y2)``, ``x1 + s1 sqrt(y1)
    == x2 + s2 sqrt(y2)`` forces ``x1 == x2``, ``s1 == s2`` and ``y1 == y2``
    (otherwise one root would be rational), and an irrational never equals a
    rational. Order: unequal reals separate under rational enclosures built
    from integer square roots at increasing precision.
    """
    a, b = _alg_normal(r1), _alg_normal(r2)
    if a == b:
        return 0
    if a[1] == 0 and b[1] == 0:
        return (a[0] > b[0]) - (a[0] < b[0])
    bits = 64
    while bits <= _REFINE_LIMIT_BITS:
        alo, ahi = _alg_bounds(a, bits)
        blo, bhi = _alg_bounds(b, bits)
        if ahi < blo:
            return -1
        if bhi < alo:
            return 1
        bits *= 2
    raise ArithmeticError(f"verifier could not separate {a} from {b}")


def _free_part(c: _Pt, u: _Pt, v: _Pt, d2: Fraction) -> _Span | None:
    """Exact ``{t in [0, 1] : ||u + t(v - u) - c|| <= delta}``, or ``None`` if empty."""
    d = [vk - uk for uk, vk in zip(u, v, strict=True)]
    w = [uk - ck for uk, ck in zip(u, c, strict=True)]
    qa = sum((x * x for x in d), Fraction(0))
    qb = sum((x * y for x, y in zip(d, w, strict=True)), Fraction(0))
    qc = sum((x * x for x in w), Fraction(0)) - d2
    if qa == 0:
        return (_A_ZERO, _A_ONE) if qc <= 0 else None
    at_start = qc <= 0
    at_end = qa + 2 * qb + qc <= 0
    disc = qb * qb - qa * qc
    if not (at_start or at_end) and (disc < 0 or not 0 < -qb < qa):
        return None
    centre, spread = -qb / qa, disc / (qa * qa)
    return (
        _A_ZERO if at_start else (centre, -1, spread),
        _A_ONE if at_end else (centre, 1, spread),
    )


def _advance(free: _Span | None, straight: _Span | None, across: _Span | None) -> _Span | None:
    """Reachable part of a cell's exit edge.

    ``straight`` is the reachable part of the parallel entry edge, whose lowest
    point must not be undercut; ``across`` is the reachable part of the
    perpendicular entry edge, from which every free exit point is reachable by
    convexity of the free space inside a cell.
    """
    if free is None:
        return None
    if across is not None:
        return free
    if straight is None or _alg_cmp(free[1], straight[0]) < 0:
        return None
    low = free[0] if _alg_cmp(free[0], straight[0]) >= 0 else straight[0]
    return (low, free[1])


def continuous_frechet_le(a: np.ndarray, b: np.ndarray, delta: float) -> bool:
    """Exact ``d_F(a, b) <= delta`` for polygonal curves, by Alt-Godau.

    Shares no code with the solver: exact rationals throughout, and endpoint
    comparisons by normalisation plus integer-square-root refinement rather
    than by the solver's squaring method. ``O(m n)`` cells of a few rational
    operations each, so it is meant for checking, not for search.
    """
    pa = [tuple(Fraction(float(x)) for x in row) for row in _as_points(a)]
    pb = [tuple(Fraction(float(x)) for x in row) for row in _as_points(b)]
    if not pa or not pb:
        raise ValueError("continuous Frechet needs two non-empty curves")
    d2 = Fraction(float(delta)) ** 2

    def close(u: _Pt, v: _Pt) -> bool:
        return sum(((x - y) ** 2 for x, y in zip(u, v, strict=True)), Fraction(0)) <= d2

    m, n = len(pa), len(pb)
    if m == 1 or n == 1:
        # One curve is a single point; by convexity the farthest point of the
        # other curve from it is one of its vertices.
        return all(close(u, v) for u in pa for v in pb)
    if not (close(pa[0], pb[0]) and close(pa[-1], pb[-1])):
        return False

    # up[i][j]: reachable part of {i} x [j, j+1]; side[i][j]: of [i, i+1] x {j}.
    up: list[list[_Span | None]] = [[None] * (n - 1) for _ in range(m)]
    side: list[list[_Span | None]] = [[None] * n for _ in range(m - 1)]
    for j in range(n - 1):
        if not close(pa[0], pb[j]):
            break
        up[0][j] = _free_part(pa[0], pb[j], pb[j + 1], d2)
    for i in range(m - 1):
        if not close(pa[i], pb[0]):
            break
        side[i][0] = _free_part(pb[0], pa[i], pa[i + 1], d2)

    for i in range(m - 1):
        for j in range(n - 1):
            up[i + 1][j] = _advance(
                _free_part(pa[i + 1], pb[j], pb[j + 1], d2), up[i][j], side[i][j]
            )
            side[i][j + 1] = _advance(
                _free_part(pb[j + 1], pa[i], pa[i + 1], d2), side[i][j], up[i][j]
            )
    return up[m - 1][n - 2] is not None or side[m - 2][n - 1] is not None


def _as_points(curve: np.ndarray) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def verify_continuous_witness(
    reference: np.ndarray,
    observation: np.ndarray,
    result: EditResult,
) -> VerificationReport:
    """Check a continuous deletion witness: replay, recount, and decide exactly.

    A passing report establishes FEASIBILITY at the reported cost - the edited
    curve is the observation with exactly ``cost`` vertices deleted, and its
    continuous Frechet distance to the reference is at most ``delta`` - by
    exact arithmetic that shares no code with the solver. Minimality comes from
    oracle agreement, not from here.
    """
    reference = _as_points(reference)
    observation = _as_points(observation)

    if result.witness_status != "certified":
        return VerificationReport(
            ok=False,
            violations=[f"witness_status is {result.witness_status!r}, nothing to verify"],
        )
    if result.edits is None or result.edited_curve is None:
        return VerificationReport(ok=False, violations=["witness fields are incomplete"])
    if result.mode != "delete" or any(not isinstance(e, Deletion) for e in result.edits):
        return VerificationReport(
            ok=False, violations=["a continuous witness may contain deletions only"]
        )

    try:
        replayed = replay(observation, result.edits)
    except ValueError as exc:
        return VerificationReport(ok=False, violations=[f"replay failed: {exc}"])

    violations: list[str] = []
    if replayed.shape != result.edited_curve.shape or not np.array_equal(
        replayed, result.edited_curve
    ):
        violations.append(
            f"replayed curve (shape {replayed.shape}) does not match the reported "
            f"edited_curve (shape {result.edited_curve.shape})"
        )
    if result.cost is None or len(result.edits) != result.cost:
        violations.append(f"witness has {len(result.edits)} edits but cost is {result.cost!r}")
    if len(replayed) == 0:
        violations.append("edited curve is empty; continuous Frechet needs a non-empty curve")
    elif not continuous_frechet_le(reference, replayed, result.delta):
        violations.append(
            "independent exact check: the edited curve's continuous Frechet distance "
            f"to the reference exceeds delta {result.delta!r}"
        )
    return VerificationReport(ok=not violations, violations=violations, replayed_curve=replayed)
