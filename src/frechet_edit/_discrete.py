"""Optimised score-only dynamic program.

Same recurrences as ``_reference_dp`` (``docs/recurrences.md`` section 2) with
two changes and no others:

* columns roll, so working storage is ``O(m)`` rather than ``O(mn)``. No dense
  ``m x n`` distance matrix is ever formed; each column materialises one
  boolean row of length ``m``.
* the insertion window minimum comes from :class:`~frechet_edit._minqueue.MinQueue`
  instead of an ``O(m)`` rescan, so the whole DP is ``O(mn)`` after the
  ``O(m^2)`` ``mu`` preprocessing.

It returns a cost only. Witnesses come from ``_reference_dp`` plus
``_traceback``, and ``tests/property/test_invariants.py`` checks that the two
backends always agree on cost.
"""

from __future__ import annotations

import numpy as np

from ._geometry import mu_indices
from ._minqueue import MinQueue
from ._numerics import NumericPolicy, PredicateStats, within_row
from ._reference_dp import INF
from ._types import Operations


def solve_cost(
    reference: np.ndarray,
    observation: np.ndarray,
    delta: float,
    mode: Operations,
    *,
    policy: NumericPolicy = "certified",
    stats: PredicateStats | None = None,
) -> tuple[float, dict[str, int]]:
    """Return ``(cost, counters)``; cost is ``inf`` when the mode is infeasible."""
    m = len(reference)
    n = len(observation)
    allow_delete = mode in ("delete", "both")
    allow_insert = mode in ("insert", "both")

    mu = mu_indices(reference, delta, policy=policy, stats=stats) if allow_insert else None

    fprev = [INF] * (m + 1)
    fcur = [INF] * (m + 1)
    queue = MinQueue()
    states = 0

    for j in range(n + 1):
        close: np.ndarray | None = None
        if j >= 1:
            close = within_row(
                observation[j - 1], reference, delta, policy=policy, stats=stats
            )

        if j == 0:
            fcur[0] = 0.0
        elif allow_delete and fprev[0] != INF:
            fcur[0] = fprev[0] + 1.0
        else:
            fcur[0] = INF

        if allow_insert:
            queue.clear()
            queue.push(0, fcur[0])

        k_prev_row = INF  # K(i-1, j), same column
        for i in range(1, m + 1):
            states += 1

            k_val = INF
            if j >= 1 and close is not None and bool(close[i - 1]):
                k_val = min(k_prev_row, fprev[i - 1], fprev[i])

            p_val = INF
            if allow_insert:
                assert mu is not None
                queue.expire_before(int(mu[i - 1]))
                best = queue.min_value()
                if best != INF:
                    p_val = best + 1.0

            x_val = INF
            if allow_delete and j >= 1 and fprev[i] != INF:
                x_val = fprev[i] + 1.0

            fcur[i] = min(k_val, p_val, x_val)
            k_prev_row = k_val
            # The queue is only consulted by the insertion branch, so in
            # deletion-only mode it is never built at all.
            if allow_insert:
                queue.push(i, fcur[i])

        fprev, fcur = fcur, fprev

    counters = {
        "states": states,
        "minqueue_pushes": queue.pushes,
        "minqueue_pops": queue.pops,
        "working_floats": 2 * (m + 1),
    }
    return fprev[m], counters
