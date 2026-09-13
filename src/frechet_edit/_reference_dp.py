"""Readable reference dynamic program.

This is the permanent, deliberately plain translation of the recurrences in
``docs/recurrences.md`` section 2. It keeps full ``(m+1) x (n+1)`` tables and
parent records so that a witness can be reconstructed without ambiguity, and it
is the oracle that the optimised backend in ``_discrete.py`` must match.

Do not optimise this file. Its job is to be obviously correct.

Index convention: ``i`` and ``j`` are ONE-BASED prefix LENGTHS, so state
``(i, j)`` means "the first ``i`` reference vertices against the first ``j``
observation vertices". Zero-based vertex access is therefore ``reference[i-1]``
and ``observation[j-1]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from ._geometry import mu_indices
from ._numerics import NumericPolicy, PredicateStats, within_row
from ._types import Operations

INF: Final[float] = math.inf

# Which layer achieved F(i, j).
FROM_NONE: Final[int] = -1
FROM_K: Final[int] = 0
FROM_P: Final[int] = 1
FROM_X: Final[int] = 2

# Which incoming coupling step achieved K(i, j).
K_VERTICAL: Final[int] = 0
K_DIAGONAL: Final[int] = 1
K_HORIZONTAL: Final[int] = 2


@dataclass(slots=True)
class DPTables:
    """Filled tables plus the parent records traceback needs."""

    f: list[list[float]]
    k: list[list[float]]
    f_from: list[list[int]]
    k_from: list[list[int]]
    p_block: list[list[int]]
    mu: np.ndarray | None
    states: int

    @property
    def cost(self) -> float:
        return self.f[-1][-1]


def solve(
    reference: np.ndarray,
    observation: np.ndarray,
    delta: float,
    mode: Operations,
    *,
    policy: NumericPolicy = "certified",
    stats: PredicateStats | None = None,
) -> DPTables:
    """Fill the layered tables for the requested mode.

    Raises :class:`~frechet_edit._numerics.NumericallyAmbiguous` if a geometric
    predicate cannot be certified; the caller turns that into a reported status.
    """
    m = len(reference)
    n = len(observation)
    allow_delete = mode in ("delete", "both")
    allow_insert = mode in ("insert", "both")

    mu = mu_indices(reference, delta, policy=policy, stats=stats) if allow_insert else None

    # f[i][j], k[i][j] indexed by prefix lengths, hence (m+1) x (n+1).
    f = [[INF] * (n + 1) for _ in range(m + 1)]
    k = [[INF] * (n + 1) for _ in range(m + 1)]
    f_from = [[FROM_NONE] * (n + 1) for _ in range(m + 1)]
    k_from = [[FROM_NONE] * (n + 1) for _ in range(m + 1)]
    p_block = [[0] * (n + 1) for _ in range(m + 1)]

    f[0][0] = 0.0
    states = 0

    # Column-major: P(i, j) reads F(k-1, j) from the SAME column, and
    # K(i, j) reads K(i-1, j) from the same column, so j must be the outer loop.
    for j in range(n + 1):
        if j >= 1:
            close = within_row(
                observation[j - 1], reference, delta, policy=policy, stats=stats
            )
        else:
            close = None

        for i in range(m + 1):
            if i == 0 and j == 0:
                continue
            states += 1

            # X: delete observation vertex j.
            x_val = INF
            if allow_delete and j >= 1:
                prev = f[i][j - 1]
                if prev != INF:
                    x_val = prev + 1.0

            # K: retain observation vertex j and couple it to reference vertex i.
            k_val = INF
            k_src = FROM_NONE
            if i >= 1 and j >= 1 and close is not None and bool(close[i - 1]):
                candidates = (
                    (k[i - 1][j], K_VERTICAL),
                    (f[i - 1][j - 1], K_DIAGONAL),
                    (f[i][j - 1], K_HORIZONTAL),
                )
                for value, source in candidates:
                    if value < k_val:
                        k_val = value
                        k_src = source
            k[i][j] = k_val
            k_from[i][j] = k_src

            # P: append one inserted point covering the reference block k..i.
            p_val = INF
            p_k = 0
            if allow_insert and i >= 1:
                assert mu is not None
                lo = int(mu[i - 1]) + 1  # smallest one-based block start
                best = INF
                best_k = 0
                for kk in range(lo, i + 1):
                    value = f[kk - 1][j]
                    if value < best:
                        best = value
                        best_k = kk
                if best != INF:
                    p_val = best + 1.0
                    p_k = best_k
            p_block[i][j] = p_k

            # F: the best of the three layers. Order K, P, X fixes tie-breaking
            # towards keeping, then inserting, then deleting
            # (docs/witness-invariants.md section 4).
            best_val = k_val
            best_src = FROM_K
            if p_val < best_val:
                best_val, best_src = p_val, FROM_P
            if x_val < best_val:
                best_val, best_src = x_val, FROM_X
            f[i][j] = best_val
            f_from[i][j] = best_src if best_val != INF else FROM_NONE

    return DPTables(
        f=f, k=k, f_from=f_from, k_from=k_from, p_block=p_block, mu=mu, states=states
    )
