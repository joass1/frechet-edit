"""Witness reconstruction from the layered tables.

Implements ``docs/witness-invariants.md`` section 3. The walk is over the
LAYERED states, so the identity of the last edited element is known at every
step and a state can never be asserted both to keep and to delete the same
observation vertex. That is the whole reason the collapsed ``F(i-1, j)``
predecessor is not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._geometry import certified_block_centre
from ._reference_dp import (
    FROM_K,
    FROM_NONE,
    FROM_P,
    FROM_X,
    INF,
    K_DIAGONAL,
    K_HORIZONTAL,
    K_VERTICAL,
    DPTables,
)
from ._types import Deletion, Insertion


class TracebackError(RuntimeError):
    """Raised when the walk reaches an impossible state. Never swallowed."""


@dataclass(slots=True)
class Witness:
    edits: tuple[Deletion | Insertion, ...]
    edited_curve: np.ndarray
    coupling: tuple[tuple[int, int], ...]


def reconstruct(
    tables: DPTables,
    reference: np.ndarray,
    observation: np.ndarray,
    delta: float,
) -> Witness | None:
    """Build a witness, or return ``None`` if an inserted centre is not representable.

    ``None`` means ``witness_status="unavailable"``: the cost stays certified
    optimal and ``delta`` is not touched.
    """
    m = len(reference)
    n = len(observation)
    if tables.f[m][n] == INF:
        raise TracebackError("reconstruct called on an infeasible instance")

    deletions: list[int] = []
    # Elements of the edited curve, collected BACKWARDS. Each entry is
    # ("keep", observation_index, refs) or ("insert", gap, block_lo, block_hi, refs)
    # where refs are ONE-BASED reference indices, descending.
    elements: list[tuple[Any, ...]] = []

    i, j = m, n
    layer = tables.f_from[i][j]
    guard = 0
    limit = 4 * (m + 1) * (n + 1) + 16

    while not (i == 0 and j == 0):
        guard += 1
        if guard > limit:  # pragma: no cover - structural safety net
            raise TracebackError("traceback exceeded its step bound")
        if layer == FROM_NONE:
            raise TracebackError(f"traceback reached an unreachable state ({i}, {j})")

        if layer == FROM_X:
            if j < 1:
                raise TracebackError(f"deletion branch at j={j}")
            deletions.append(j - 1)
            j -= 1
            layer = tables.f_from[i][j]

        elif layer == FROM_K:
            refs: list[int] = []
            step = FROM_NONE
            while True:
                if i < 1 or j < 1:
                    raise TracebackError(f"keep branch at ({i}, {j})")
                refs.append(i)
                step = tables.k_from[i][j]
                if step == K_VERTICAL:
                    i -= 1
                    continue
                break
            elements.append(("keep", j - 1, refs))
            if step == K_DIAGONAL:
                i -= 1
                j -= 1
            elif step == K_HORIZONTAL:
                j -= 1
            else:
                raise TracebackError(f"keep branch at ({i}, {j}) has no recorded step")
            layer = tables.f_from[i][j]

        elif layer == FROM_P:
            block_lo = tables.p_block[i][j]
            if not 1 <= block_lo <= i:
                raise TracebackError(f"insertion branch at ({i}, {j}) has block {block_lo}")
            elements.append(("insert", j, block_lo, i, list(range(i, block_lo - 1, -1))))
            i = block_lo - 1
            layer = tables.f_from[i][j]

        else:  # pragma: no cover - defensive
            raise TracebackError(f"unknown layer code {layer}")

    elements.reverse()

    # Materialise the edited curve and the coupling in forward order.
    points: list[np.ndarray] = []
    coupling: list[tuple[int, int]] = []
    insertions: list[Insertion] = []
    gap_counts: dict[int, int] = {}

    for element in elements:
        slot = len(points)
        if element[0] == "keep":
            _, obs_index, refs = element
            points.append(observation[obs_index])
        else:
            _, gap, block_lo, block_hi, refs = element
            centre = certified_block_centre(reference[block_lo - 1 : block_hi], delta)
            if centre is None:
                return None
            order = gap_counts.get(gap, 0)
            gap_counts[gap] = order + 1
            point = tuple(float(c) for c in centre)
            insertions.append(Insertion(gap=gap, order=order, point=point))
            points.append(centre)
        for ref in reversed(refs):  # refs were collected descending
            coupling.append((ref - 1, slot))

    ordered_insertions: list[Insertion] = sorted(
        insertions, key=lambda ins: (ins.gap, ins.order)
    )
    edits: list[Deletion | Insertion] = [Deletion(index=d) for d in sorted(deletions)]
    edits.extend(ordered_insertions)

    edited = (
        np.asarray(points, dtype=np.float64)
        if points
        else np.empty((0, reference.shape[1]), dtype=np.float64)
    )
    return Witness(
        edits=tuple(edits),
        edited_curve=edited,
        coupling=tuple(coupling),
    )
