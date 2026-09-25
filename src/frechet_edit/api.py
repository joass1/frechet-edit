"""Public entry point.

The contract is ``docs/definition.md``. Everything here is a thin, explicit
layer over validation, the dynamic program and witness reconstruction; the
mathematics lives in ``_reference_dp`` and ``_geometry``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import _continuous, _discrete, _reference_dp
from ._freespace import FreeSpace
from ._numerics import NumericallyAmbiguous, PredicateStats
from ._traceback import reconstruct
from ._types import Deletion, EditResult, Operations, UnsupportedOperationError
from ._validation import (
    as_curve,
    check_backend,
    check_delta,
    check_mode_dimension,
    check_numeric_policy,
    check_operations,
    check_pair,
)

__all__ = [
    "continuous_edit_distance",
    "continuous_frechet_within",
    "discrete_edit_distance",
    "ordinary_discrete_frechet",
]

#: Largest recorded state, in bytes, for which a continuous witness is built.
#: The witness needs the full ``(m, n, k+1, k+1)`` reachability table of the
#: final run; beyond this the cost is still returned, with the witness marked
#: unavailable rather than risking the host's memory.
CONTINUOUS_WITNESS_MAX_BYTES = 512 * 1024 * 1024


def discrete_edit_distance(
    reference: Any,
    observation: Any,
    delta: float,
    *,
    operations: Operations = "both",
    return_witness: bool = False,
    backend: str = "python",
    numeric_policy: str = "certified",
) -> EditResult:
    """Minimum number of edits to ``observation`` making it delta-close to ``reference``.

    Edits apply to ``observation`` ONLY, and the objective is a count of
    vertex insertions and deletions, not a distance in coordinate units. The
    measure is directed: swapping the two curves is a different question.

    Parameters
    ----------
    reference, observation:
        ``(k, d)`` arrays of finite coordinates, or 1-D arrays read as scalars
        in one dimension. Neither may be empty and neither is mutated.
    delta:
        Finite, strictly positive Euclidean threshold in coordinate units.
        Comparisons are closed (``distance <= delta``).
    operations:
        ``"delete"``, ``"insert"`` or ``"both"``. Insertion-capable modes are
        certified for dimensions 1 to 8; deletion has no dimension bound.
    return_witness:
        When true, also return a replayable edit script, the edited curve and a
        coupling. Costs the ``O(mn)`` parent tables of the reference backend.
    backend:
        ``"python"`` (rolling-storage DP with a monotone minimum queue) or
        ``"reference"`` (readable full-table DP). Both must return the same cost.
    numeric_policy:
        ``"certified"`` (default; exact fallback on boundary predicates, with
        explicit abstention) or ``"fast"`` (float64 only, NOT certified).

    Returns
    -------
    EditResult
        ``status="optimal"`` with an integer ``cost``, ``status="infeasible"``
        with ``cost=inf`` when the requested mode cannot reach ``delta`` at all,
        or ``status="numerically_ambiguous"`` with ``cost=None`` when the
        numerical policy was exhausted. Ambiguity is never rewritten as
        infinity or as a feasible answer.

    Raises
    ------
    ValueError
        For empty, ragged, non-finite or mismatched inputs, a non-positive or
        non-finite ``delta``, or an unknown option value.
    UnsupportedDimensionError
        For an insertion-capable mode in a dimension this backend does not
        certify.
    """
    ref = as_curve(reference, "reference")
    obs = as_curve(observation, "observation")
    dim = check_pair(ref, obs)
    delta_value = check_delta(delta)
    mode = check_operations(operations)
    backend_name = check_backend(backend)
    policy = check_numeric_policy(numeric_policy)
    check_mode_dimension(mode, dim)

    stats = PredicateStats()
    base: dict[str, Any] = {
        "mode": mode,
        "delta": delta_value,
        "dimension": dim,
        "backend": backend_name,
        "numeric_policy": policy,
    }

    use_tables = return_witness or backend_name == "reference"
    try:
        if use_tables:
            tables = _reference_dp.solve(ref, obs, delta_value, mode, policy=policy, stats=stats)
            cost = tables.cost
            counters: dict[str, Any] = {"states": tables.states}
        else:
            tables = None
            cost, counters = _discrete.solve_cost(
                ref, obs, delta_value, mode, policy=policy, stats=stats
            )
    except NumericallyAmbiguous as exc:
        stats.ambiguous += 1
        return EditResult(
            status="numerically_ambiguous",
            cost=None,
            witness_status="not_requested",
            stats=stats.as_dict(),
            detail=exc.detail,
            **base,
        )

    counters.update(stats.as_dict())
    counters["reference_length"] = len(ref)
    counters["observation_length"] = len(obs)

    if math.isinf(cost):
        return EditResult(
            status="infeasible",
            cost=math.inf,
            witness_status="not_requested",
            stats=counters,
            detail=(
                f"no sequence of {mode} edits makes the observation "
                f"delta-close to the reference at delta={delta_value!r}"
            ),
            **base,
        )

    if not return_witness:
        return EditResult(
            status="optimal", cost=int(cost), witness_status="not_requested",
            stats=counters, **base,
        )

    assert tables is not None
    try:
        witness = reconstruct(tables, ref, obs, delta_value)
    except NumericallyAmbiguous as exc:
        stats.ambiguous += 1
        counters.update(stats.as_dict())
        return EditResult(
            status="optimal",
            cost=int(cost),
            witness_status="unavailable",
            stats=counters,
            detail=f"cost is certified; witness abandoned: {exc.detail}",
            **base,
        )

    if witness is None:
        return EditResult(
            status="optimal",
            cost=int(cost),
            witness_status="unavailable",
            stats=counters,
            detail=(
                "cost is certified optimal, but no float64 point near an inserted "
                "centre could be certified within delta; delta was not adjusted"
            ),
            **base,
        )

    return EditResult(
        status="optimal",
        cost=int(cost),
        witness_status="certified",
        edited_curve=witness.edited_curve,
        edits=witness.edits,
        coupling=witness.coupling,
        stats=counters,
        **base,
    )


def ordinary_discrete_frechet(a: Any, b: Any) -> float:
    """Ordinary strong discrete Frechet distance, for comparison and testing.

    This is the residual geometric constraint of the edit objective, not the
    objective itself. Provided so callers do not have to reach into
    ``frechet_edit.verify``.
    """
    from .verify import discrete_frechet

    left = as_curve(a, "a")
    right = as_curve(b, "b")
    check_pair(left, right)
    return discrete_frechet(np.asarray(left), np.asarray(right))


def continuous_frechet_within(a: Any, b: Any, delta: float) -> bool:
    """Exact decision: is the ordinary CONTINUOUS Frechet distance ``<= delta``?

    Curves are polygonal (linear between consecutive vertices). This is the
    budget-0 case of the continuous deletion solver, so it shares that
    solver's certified predicates and exact ranking. Comparisons are closed.
    """
    left = as_curve(a, "a")
    right = as_curve(b, "b")
    check_pair(left, right)
    delta_value = check_delta(delta)
    return _continuous.run(FreeSpace(left, right, delta_value), 0).cost == 0


def _check_max_deletions(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"max_deletions must be a non-negative integer or None, got {value!r}")
    if value < 0:
        raise ValueError(f"max_deletions must be non-negative, got {value!r}")
    return int(value)


def continuous_edit_distance(
    reference: Any,
    observation: Any,
    delta: float,
    *,
    operations: Operations = "delete",
    return_witness: bool = False,
    max_deletions: int | None = None,
) -> EditResult:
    """Fewest vertices to delete from ``observation`` so its polygonal curve is
    within CONTINUOUS Frechet distance ``delta`` of ``reference``.

    Both inputs are read as polygonal curves, linear between consecutive
    vertices, which is what makes this measure robust to sampling density: a
    sparse reference polyline and a densely sampled observation of the same
    path are close, where discrete Frechet would force vertex-to-vertex
    matches. Implements Section 4.1 (Theorem 3) of Fox, Nayyeri, Perry and
    Raichel, SoCG 2024; see ``docs/continuous.md``.

    Parameters
    ----------
    reference, observation:
        ``(k, d)`` arrays of finite coordinates, any dimension, or 1-D arrays
        read as scalars. Neither is mutated. Only ``observation`` is edited.
    delta:
        Finite, strictly positive threshold in coordinate units; closed.
    operations:
        Only ``"delete"``. Continuous insertion needs the paper's minimum-link
        machinery and is not implemented; asking for it raises
        :class:`UnsupportedOperationError` rather than silently doing something
        else.
    return_witness:
        When true, also return which vertices to delete and the edited curve.
    max_deletions:
        Optional cap on the search. The running time is ``O(k^2 m n)`` for a
        budget ``k``, so a cap bounds it; if no solution uses at most this many
        deletions the result is ``status="budget_exceeded"``, which is NOT a
        claim of infeasibility.

    Returns
    -------
    EditResult
        ``backend="continuous"``, ``mode="delete"``. ``status`` is
        ``"optimal"``, ``"infeasible"`` (no non-empty subsequence works), or
        ``"budget_exceeded"``. The cost is a COUNT of deletions, never a
        distance. There is no discrete coupling; ``coupling`` is ``None``.
    """
    ref = as_curve(reference, "reference")
    obs = as_curve(observation, "observation")
    dim = check_pair(ref, obs)
    delta_value = check_delta(delta)
    mode = check_operations(operations)
    if mode != "delete":
        raise UnsupportedOperationError(
            f"continuous_edit_distance supports operations='delete' only, got {mode!r}. "
            "Continuous insertion needs minimum-link machinery that is not implemented; "
            "use discrete_edit_distance for insertions."
        )
    cap_request = _check_max_deletions(max_deletions)

    stats = PredicateStats()
    base: dict[str, Any] = {
        "mode": mode,
        "delta": delta_value,
        "dimension": dim,
        "backend": "continuous",
        "numeric_policy": "certified",
    }
    m, n = len(ref), len(obs)
    # Discrete deletion is O(mn), certified, and never cheaper: a subsequence
    # within discrete Frechet delta is within continuous Frechet delta.
    discrete_cost, _ = _discrete.solve_cost(ref, obs, delta_value, "delete", stats=stats)
    upper = None if math.isinf(discrete_cost) else int(discrete_cost)

    space = FreeSpace(ref, obs, delta_value, stats=stats)
    cap = n - 1 if cap_request is None else min(cap_request, n - 1)
    cost, budgets = _continuous.best_cost(space, cap, upper)

    counters: dict[str, Any] = {
        "reference_length": m,
        "observation_length": n,
        "budgets_tried": budgets,
        "discrete_deletion_upper_bound": upper,
    }
    counters.update(stats.as_dict())

    if cost is None:
        return _continuous_no_solution(base, counters, cap, n, cap_request, delta_value)

    if not return_witness:
        return EditResult(status="optimal", cost=cost, stats=counters, **base)

    table_bytes = 4 * m * n * min(cost + 1, max(n - 1, 0)) * (cost + 1)
    if table_bytes > CONTINUOUS_WITNESS_MAX_BYTES:
        return EditResult(
            status="optimal",
            cost=cost,
            witness_status="unavailable",
            stats=counters,
            detail=(
                f"cost is exact; the witness table would need {table_bytes} bytes, above "
                f"CONTINUOUS_WITNESS_MAX_BYTES={CONTINUOUS_WITNESS_MAX_BYTES}"
            ),
            **base,
        )

    kept = _continuous_witness(space, cost)
    deleted = sorted(set(range(n)) - set(kept))
    return EditResult(
        status="optimal",
        cost=cost,
        witness_status="certified",
        edited_curve=obs[kept].copy(),
        edits=tuple(Deletion(idx) for idx in deleted),
        coupling=None,
        stats=counters,
        **base,
    )


def _continuous_no_solution(
    base: dict[str, Any],
    counters: dict[str, Any],
    cap: int,
    n: int,
    cap_request: int | None,
    delta_value: float,
) -> EditResult:
    """``infeasible`` when the whole search space was covered, else ``budget_exceeded``."""
    if cap >= n - 1:
        return EditResult(
            status="infeasible",
            cost=math.inf,
            stats=counters,
            detail=(
                "no non-empty subsequence of the observation is within continuous "
                f"Frechet distance delta={delta_value!r} of the reference"
            ),
            **base,
        )
    return EditResult(
        status="budget_exceeded",
        cost=None,
        stats=counters,
        detail=(
            f"no solution deletes at most max_deletions={cap_request} vertices; "
            "a larger budget may succeed"
        ),
        **base,
    )


def _continuous_witness(space: FreeSpace, cost: int) -> list[int]:
    """Kept observation indices of one optimal solution, re-derived and cross-checked."""
    final = _continuous.run(space, cost, record=True)
    if (
        final.cost != cost
        or final.tables is None
        or final.end_vertex is None
        or final.end_copy is None
    ):
        raise AssertionError("continuous witness run disagreed with the cost search")
    kept = _continuous.traceback(space, final.tables, final.end_vertex, final.end_copy)
    if space.n - len(kept) != cost:
        raise AssertionError(
            f"continuous traceback kept {len(kept)} of {space.n} vertices, "
            f"inconsistent with cost {cost}"
        )
    return kept
