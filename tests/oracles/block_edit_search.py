"""Oracle B - insertion and mixed edit search (``docs/oracle-protocol.md``).

B.1 Finite candidate lemma.  If any feasible edited curve of cost ``c`` exists
then one of cost ``c`` exists in which every inserted point is the exact
minimum-enclosing-ball centre of a contiguous block of R.  Hence

    CANDIDATES = { MEB_centre(R[a..b]) : 0 <= a <= b < m, radius <= delta }

is complete.  Centres come from Oracle C (:mod:`.enclosing_circle`) and are
exact rationals.  A candidate may be reused any number of times and in any
gap; the lemma places no restriction on which block a candidate is later
coupled to.

B.2 Insertion count bound.  An optimal solution uses at most ``m`` insertions,
because in an optimal solution every inserted element exclusively covers at
least one of the ``m`` reference indices and those exclusive sets are disjoint.

B.3 Search.  Iterative deepening on the total cost ``c = 0, 1, 2, ...``:
enumerate retained subsequences of Q consistent with the deletion budget, then
ordered placements of the remaining budget of insertions into the ``n + 1``
original gaps drawing from CANDIDATES with repetition, build the final curve by
replay (``docs/witness-invariants.md`` section 2, re-implemented here), and
accept the first curve whose independent ``discrete_frechet(R, curve)`` is
``<= delta`` under exact rational comparison.

Budget: ``MAX_STATES`` curves evaluated and ``TIMEOUT_S`` seconds per call.
Exceeding either raises :class:`OracleBudgetExceeded`.  A test that hits the
budget is BLOCKED evidence and is NOT agreement; callers must never catch this
and turn it into a skip that reads as a pass.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Any

from .enclosing_circle import exact_meb
from .exact import delta_sq
from .frechet import discrete_frechet_le

__all__ = [
    "MAX_STATES",
    "OPERATIONS",
    "TIMEOUT_S",
    "OracleBudgetExceeded",
    "candidate_points",
    "edit_oracle",
    "replay",
]

MAX_STATES = 100_000
TIMEOUT_S = 10.0

OPERATIONS = ("delete", "insert", "both")

Point = tuple[Fraction, ...]


class OracleBudgetExceeded(RuntimeError):
    """Raised when the oracle search exceeds MAX_STATES or TIMEOUT_S.

    This is BLOCKED evidence, never a pass.  Do not catch it to produce a skip.
    """


def candidate_points(R: Sequence[Any], delta: Any) -> tuple[Point, ...]:
    """Exact MEB centres of every contiguous block of R with radius <= delta.

    Returned sorted and de-duplicated (two blocks can share a centre; keeping
    one copy loses nothing, since the lemma allows any candidate in any gap).
    """
    limit = delta_sq(delta)
    m = len(R)
    if m == 0:
        raise ValueError("reference must be non-empty")
    found: set[Point] = set()
    for a in range(m):
        for b in range(a, m):
            centre, sq_radius = exact_meb(R[a : b + 1])
            if sq_radius <= limit:
                found.add(centre)
    return tuple(sorted(found))


def replay(
    Q: Sequence[Any],
    deleted: frozenset[int],
    insertions: dict[int, tuple[Point, ...]],
) -> list[Any]:
    """Replay semantics of ``docs/witness-invariants.md`` section 2.

    Pure function of the ORIGINAL curve and the edit records::

        out = []
        for gap g in 0..n:
            for ins in insertions with gap == g, ordered by ins.order:
                out.append(ins.point)
            if g < n and g not in deleted:
                out.append(Q[g])

    Indices never shift; ``gap = n`` places points after the last original
    vertex; the procedure is well defined when every original index is deleted,
    in which case the output is the inserted points alone in gap-then-order
    sequence.
    """
    n = len(Q)
    out: list[Any] = []
    for gap in range(n + 1):
        out.extend(insertions.get(gap, ()))
        if gap < n and gap not in deleted:
            out.append(Q[gap])
    return out


def _insertion_plans(
    n_gaps: int, count: int, candidates: Sequence[Point]
) -> Iterator[dict[int, tuple[Point, ...]]]:
    """Yield every ordered placement of exactly ``count`` insertions.

    A placement assigns each of the ``n_gaps`` gaps an ordered tuple of
    candidate points (repetition allowed); the tuple lengths sum to ``count``.
    Each distinct edit script is produced exactly once.
    """
    if count == 0:
        yield {}
        return
    if not candidates or n_gaps <= 0:
        return

    def walk(gap: int, remaining: int) -> Iterator[dict[int, tuple[Point, ...]]]:
        if gap == n_gaps - 1:
            for combo in itertools.product(candidates, repeat=remaining):
                yield {gap: combo} if combo else {}
            return
        for taken in range(remaining + 1):
            for combo in itertools.product(candidates, repeat=taken):
                head = {gap: combo} if combo else {}
                for tail in walk(gap + 1, remaining - taken):
                    merged = dict(head)
                    merged.update(tail)
                    yield merged

    yield from walk(0, count)


def _deleted_sets(n: int, deletions: int) -> Iterator[frozenset[int]]:
    """Yield the deleted-index set of every subsequence of Q having exactly
    ``deletions`` deletions, in a deterministic order."""
    for combo in itertools.combinations(range(n), deletions):
        yield frozenset(combo)


def _splits(operations: str, cost: int, m: int, n: int) -> Iterator[tuple[int, int]]:
    """Yield ``(deletions, insertions)`` pairs summing to ``cost`` that the mode
    allows.  Insertions are capped at ``m`` by lemma B.2."""
    if operations == "delete":
        if cost <= n:
            yield (cost, 0)
        return
    if operations == "insert":
        if cost <= m:
            yield (0, cost)
        return
    for inserted in range(0, min(m, cost) + 1):
        deleted = cost - inserted
        if deleted <= n:
            yield (deleted, inserted)


def edit_oracle(
    R: Sequence[Any],
    Q: Sequence[Any],
    delta: Any,
    operations: str = "both",
    stats: dict[str, Any] | None = None,
    max_states: int | None = None,
    timeout_s: float | None = None,
) -> float:
    """Brute-force minimum edit count, or ``math.inf`` when infeasible.

    ``operations`` is one of ``"delete"``, ``"insert"``, ``"both"``.
    ``stats``, when given, is filled in with the number of curves evaluated and
    related counters, including when the call returns ``inf`` or raises
    :class:`OracleBudgetExceeded`.
    """
    if operations not in OPERATIONS:
        raise ValueError(f"operations must be one of {OPERATIONS}; got {operations!r}")
    m, n = len(R), len(Q)
    if m == 0:
        raise ValueError("reference must be non-empty")
    if n == 0:
        raise ValueError("observation must be non-empty")
    delta_sq(delta)  # validate delta before doing any search work

    budget_states = MAX_STATES if max_states is None else max_states
    budget_time = TIMEOUT_S if timeout_s is None else timeout_s

    candidates: tuple[Point, ...] = ()
    if operations in ("insert", "both"):
        candidates = candidate_points(R, delta)

    counters: dict[str, Any] = {
        "curves_evaluated": 0,
        "candidates": len(candidates),
        "max_cost_searched": 0,
        "elapsed_s": 0.0,
    }
    if stats is not None:
        stats.update(counters)

    started = time.perf_counter()

    def publish() -> None:
        counters["elapsed_s"] = time.perf_counter() - started
        if stats is not None:
            stats.update(counters)

    def spend() -> None:
        counters["curves_evaluated"] += 1
        if counters["curves_evaluated"] > budget_states:
            publish()
            raise OracleBudgetExceeded(
                f"oracle evaluated more than {budget_states} curves "
                f"(m={m}, n={n}, operations={operations!r})"
            )
        if time.perf_counter() - started > budget_time:
            publish()
            raise OracleBudgetExceeded(
                f"oracle exceeded {budget_time}s (m={m}, n={n}, operations={operations!r}, "
                f"curves={counters['curves_evaluated']})"
            )

    # Total cost is bounded by m + n: delete all of Q, then insert a cover of R
    # (each single reference vertex is its own zero-radius block centre).
    for cost in range(0, m + n + 1):
        counters["max_cost_searched"] = cost
        for deletions, insertions in _splits(operations, cost, m, n):
            for deleted in _deleted_sets(n, deletions):
                for plan in _insertion_plans(n + 1, insertions, candidates):
                    curve = replay(Q, deleted, plan)
                    spend()
                    if not curve:
                        # Every original vertex deleted and nothing inserted:
                        # empty against a non-empty R is INFEASIBLE by
                        # definition, so this curve simply never qualifies.
                        continue
                    if discrete_frechet_le(R, curve, delta):
                        publish()
                        return float(cost)
    publish()
    return math.inf
