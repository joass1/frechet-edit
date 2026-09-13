"""Public entry point.

The contract is ``docs/definition.md``. Everything here is a thin, explicit
layer over validation, the dynamic program and witness reconstruction; the
mathematics lives in ``_reference_dp`` and ``_geometry``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import _discrete, _reference_dp
from ._numerics import NumericallyAmbiguous, PredicateStats
from ._traceback import reconstruct
from ._types import EditResult, Operations
from ._validation import (
    as_curve,
    check_backend,
    check_delta,
    check_mode_dimension,
    check_numeric_policy,
    check_operations,
    check_pair,
)

__all__ = ["discrete_edit_distance", "ordinary_discrete_frechet"]


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
        certified for dimensions 1 and 2 only.
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
