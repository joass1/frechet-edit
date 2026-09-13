"""Oracle A - deletion-only edit distance (``docs/oracle-protocol.md``).

For every non-empty subsequence S of Q, decide ``discrete_frechet(R, S) <=
delta`` with the independent naive DP in :mod:`.frechet`, and keep the S with
the fewest deletions.  Return ``math.inf`` when no subsequence works.

This tests the production deletion DP without duplicating it: it never forms
the edit table, never uses the three predecessor cases, and decides
feasibility only through an ordinary discrete Frechet computation.

Enumeration size ``2^n - 1``; use is restricted to ``n <= 12``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .frechet import discrete_frechet_le

__all__ = ["MAX_OBSERVATION_LEN", "deletion_oracle"]

MAX_OBSERVATION_LEN = 12


def deletion_oracle(
    R: Sequence[Any],
    Q: Sequence[Any],
    delta: Any,
    stats: dict[str, Any] | None = None,
) -> float:
    """Minimum number of deletions from Q making it within ``delta`` of R.

    Returns a ``float``: the deletion count, or ``math.inf`` when
    deletion-only is mathematically INFEASIBLE (``docs/definition.md``
    section 6 - infeasibility is a reported value, not an exception).
    """
    n = len(Q)
    m = len(R)
    if m == 0:
        raise ValueError("reference must be non-empty")
    if n == 0:
        raise ValueError("observation must be non-empty")
    if n > MAX_OBSERVATION_LEN:
        raise ValueError(
            f"deletion_oracle is restricted to n <= {MAX_OBSERVATION_LEN}; got {n}"
        )

    evaluated = 0
    # Search by increasing deletion count so the first feasible subsequence is
    # optimal; the enumeration is still over all 2^n - 1 non-empty masks.
    masks = list(range(1, 1 << n))
    masks.sort(key=lambda mask: (n - bin(mask).count("1"), mask))

    result: float = math.inf
    for mask in masks:
        kept = [Q[i] for i in range(n) if mask & (1 << i)]
        evaluated += 1
        if discrete_frechet_le(R, kept, delta):
            result = float(n - len(kept))
            break

    if stats is not None:
        stats["subsequences_evaluated"] = evaluated
        stats["subsequences_total"] = (1 << n) - 1
    return result
