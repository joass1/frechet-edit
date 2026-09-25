"""Continuous (strong) Frechet edit distance, deletions only.

Implements Section 4.1 of Fox, Nayyeri, Perry and Raichel (SoCG 2024,
Theorem 3): the observation ``sigma`` becomes the *complete weighted DAG
complex* - vertex copies ``(j, l)`` meaning "at observation vertex ``j`` having
deleted ``l`` vertices so far", with an edge ``(i, l) -> (j, l + j - i - 1)``
for every ``i < j`` within the budget ``k`` - and reachability is propagated
through the free space of its product with the fixed reference ``pi``.

Start vertices are ``(j, j)`` at ``pi_1`` (the first ``j`` observation
vertices deleted); reaching ``(j, l)`` at ``pi_m`` costs ``l + (n - 1 - j)``
(the suffix after ``j`` deleted too). With budget ``k`` the complex has
``O(k^2 n)`` edges and the product ``O(k^2 m n)`` cells.

Deviations from the prose of the paper, all recorded in ``docs/continuous.md``:

* **Explicit vertex propagation.** The paper propagates reachability cell by
  cell. A path that waits at an observation vertex with NO outgoing edge (the
  copies of the last vertex) while the reference advances is carried by no
  cell, so reachability of product vertices is propagated explicitly here: a
  vertex is reachable if it is free and is a start, or ends a reachable
  horizontal or vertical edge, and a reachable vertex makes every incident
  outgoing edge fully reachable. For ``m <= 2`` or ``n`` vertices with
  outgoing edges this coincides with the paper; the pinned counterexample for
  a cells-only reading is in ``tests/unit/test_continuous.py``.
* **Anti-diagonal wavefronts.** Cell ``(a, j)`` depends only on ``(a, i < j)``
  and ``(a - 1, j)``, so all cells with equal ``a + j`` are independent and
  are processed as one vectorised step. Only ``k + 3`` wavefronts of history
  are kept unless a witness is requested.
* **Exact integer comparisons.** Interval endpoints are pre-ranked per segment
  by :mod:`._freespace`, so the propagation compares integers and is exact.

Ranks: horizontal-edge ranks are comparable within one reference segment, and
vertical-edge ranks within one observation segment - exactly the pairs the
propagation ever compares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._freespace import BIG, FreeSpace

__all__ = ["ReachTables", "best_cost", "run", "traceback"]


@dataclass
class ReachTables:
    """Full propagation state, recorded only when a witness is requested."""

    k: int
    v_reach: np.ndarray  # (m, n, G, L) lowest reachable rank per vertical edge
    h_reach: np.ndarray  # (m - 1, n, L) lowest reachable rank per horizontal edge
    vertex: np.ndarray  # (m, n, L) product-vertex reachability
    v_lo: np.ndarray
    v_hi: np.ndarray
    v_ne: np.ndarray


@dataclass
class RunResult:
    cost: int | None
    end_vertex: int | None
    end_copy: int | None
    wavefronts: int
    tables: ReachTables | None


def _cell_outputs(
    left: np.ndarray,
    bottom: np.ndarray,
    right_lo: np.ndarray,
    right_hi: np.ndarray,
    right_ne: np.ndarray,
    top_lo: np.ndarray,
    top_hi: np.ndarray,
    top_ne: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Alt-Godau propagation through a batch of cells, on ranks.

    ``left`` and ``bottom`` are the lowest reachable ranks on each cell's left
    and bottom edges (``BIG`` if unreachable). Free space inside a cell is
    convex, so any free point of the right edge is reachable from a reachable
    bottom point, and a right point at or above a reachable left point is
    reachable from it; symmetrically for the top edge. Each returned rank is
    the lowest reachable rank on that edge; the top of every reachable part is
    the free interval's upper end.
    """
    has_left = left < BIG
    has_bottom = bottom < BIG
    right = np.where(
        has_bottom,
        right_lo,
        np.where(has_left & (right_hi >= left), np.maximum(right_lo, left), BIG),
    )
    right = np.where(right_ne, right, BIG)
    top = np.where(
        has_left,
        top_lo,
        np.where(has_bottom & (top_hi >= bottom), np.maximum(top_lo, bottom), BIG),
    )
    top = np.where(top_ne, top, BIG)
    return right.astype(np.int32, copy=False), top.astype(np.int32, copy=False)


def run(fs: FreeSpace, k: int, *, record: bool = False) -> RunResult:
    """Propagate reachability with deletion budget ``k``.

    Returns the minimum cost ``<= k`` over reachable end vertices, or ``None``
    when no deletion set of size at most ``k`` works.
    """
    m, n = fs.m, fs.n
    k = max(0, min(k, n - 1))
    num_gaps = min(k + 1, n - 1)
    num_copies = min(k, n - 1) + 1
    v_lo, v_hi, v_ne = fs.vertical(num_gaps)
    h_lo, h_hi, h_ne = fs.h_lo, fs.h_hi, fs.h_ne
    free = fs.free

    hist = num_gaps + 2
    h_hist = np.full((hist, m, num_copies), BIG, dtype=np.int32)
    vert_hist = np.zeros((hist, m, num_copies), dtype=bool)
    v_right = np.full((2, m, num_gaps, num_copies), BIG, dtype=np.int32)

    gaps = np.arange(num_gaps)
    copies = np.arange(num_copies)
    src_copy = copies[None, :] - gaps[:, None]  # (G, L): copy at the edge's tail
    src_ok = src_copy >= 0
    src_copy_c = np.where(src_ok, src_copy, 0)

    tables: ReachTables | None = None
    if record:
        tables = ReachTables(
            k=k,
            v_reach=np.full((m, n, num_gaps, num_copies), BIG, dtype=np.int32),
            h_reach=np.full((max(m - 1, 0), n, num_copies), BIG, dtype=np.int32),
            vertex=np.zeros((m, n, num_copies), dtype=bool),
            v_lo=v_lo,
            v_hi=v_hi,
            v_ne=v_ne,
        )

    best: tuple[int, int, int] | None = None
    dead_run = 0
    wavefronts = 0
    for tau in range(m + n - 1):
        wavefronts += 1
        rows = np.arange(max(0, tau - (n - 1)), min(m - 1, tau) + 1)
        cols = tau - rows
        slot = tau % hist
        h_hist[slot] = BIG
        vert_hist[slot] = False
        tail_slots = (tau - 1 - gaps) % hist  # wavefront of each edge's tail vertex
        tail_ok = (cols[:, None] - 1 - gaps[None, :] >= 0)[:, :, None] & src_ok[None]

        # Vertical edges into (a, j): from the cell below-left, or fully free
        # when the edge's tail vertex (a, j - g - 1) is reachable.
        if num_gaps:
            below = v_right[(tau - 1) % 2][np.maximum(rows - 1, 0)]
            below = np.where((rows >= 1)[:, None, None], below, BIG)
            tail_reached = vert_hist[
                tail_slots[None, :, None], rows[:, None, None], src_copy_c[None, :, :]
            ] & tail_ok
            v_cur = np.where(tail_reached, np.minimum(below, v_lo[rows, cols][:, :, None]), below)
            v_cur = np.where(v_ne[rows, cols][:, :, None], v_cur, BIG).astype(np.int32)
        else:
            v_cur = np.full((len(rows), 0, num_copies), BIG, dtype=np.int32)

        # Product vertex (a, j, l).
        reach = (v_cur < BIG).any(axis=1)
        if tau >= 1:
            h_below = h_hist[(tau - 1) % hist][np.maximum(rows - 1, 0)]
            reach |= (rows >= 1)[:, None] & (h_below < BIG)
        reach |= (rows == 0)[:, None] & (copies[None, :] == cols[:, None])
        reach &= free[rows, cols][:, None]
        vert_hist[slot, rows] = reach

        last = rows == m - 1
        if last.any():
            j_end = int(cols[last][0])
            reached = np.flatnonzero(reach[last][0])
            if len(reached):
                costs = reached + (n - 1 - j_end)
                pick = int(np.argmin(costs))
                if costs[pick] <= k and (best is None or costs[pick] < best[0]):
                    best = (int(costs[pick]), j_end, int(reached[pick]))

        # Cells (a, j) for a < m - 1: emit right edges for wavefront tau + 1
        # and the horizontal edge (a, j) for later wavefronts.
        inner = rows < m - 1
        if inner.any():
            r_in = rows[inner]
            c_in = cols[inner]
            if num_gaps:
                bottom = h_hist[tail_slots[None, :, None], r_in[:, None, None], src_copy_c[None]]
                bottom = np.where(tail_ok[inner], bottom, BIG)
                right, top = _cell_outputs(
                    v_cur[inner],
                    bottom,
                    v_lo[r_in + 1, c_in][:, :, None],
                    v_hi[r_in + 1, c_in][:, :, None],
                    v_ne[r_in + 1, c_in][:, :, None],
                    h_lo[r_in, c_in][:, None, None],
                    h_hi[r_in, c_in][:, None, None],
                    h_ne[r_in, c_in][:, None, None],
                )
                v_right[tau % 2, r_in] = right
                h_top = top.min(axis=1)
            else:
                h_top = np.full((len(r_in), num_copies), BIG, dtype=np.int32)
            h_init = np.where(reach[inner], h_lo[r_in, c_in][:, None], BIG)
            h_hist[slot, r_in] = np.minimum(h_top, h_init)

        if tables is not None:
            tables.v_reach[rows, cols] = v_cur
            tables.vertex[rows, cols] = reach
            if inner.any():
                tables.h_reach[rows[inner], cols[inner]] = h_hist[slot, rows[inner]]

        # Nothing reachable in the last `hist` wavefronts means nothing ever
        # will be - but only once every start vertex (0, j), which first
        # appears at wavefront j, has had its turn.
        alive = reach.any() or (h_hist[slot] < BIG).any() or (v_right[tau % 2] < BIG).any()
        dead_run = 0 if alive else dead_run + 1
        if dead_run >= hist and tau >= num_copies - 1 and best is None and not record:
            break

    if best is None:
        return RunResult(None, None, None, wavefronts, tables)
    return RunResult(best[0], best[1], best[2], wavefronts, tables)


def best_cost(fs: FreeSpace, cap: int, upper: int | None) -> tuple[int | None, list[int]]:
    """Minimum deletion count, searching budgets ``0, 1, 2, 4, ...`` up to ``cap``.

    ``upper`` is a known feasible cost (the discrete deletion distance, which
    can only be larger), used to stop the doubling early. A budget-``k`` run
    returns the exact optimum whenever that optimum is at most ``k``, so no
    bisection is needed. Returns the cost (``None`` if nothing within ``cap``)
    and the budgets tried.
    """
    tried: list[int] = []
    k = 0
    while True:
        budget = min(k, cap)
        if upper is not None:
            budget = min(budget, upper)
        tried.append(budget)
        result = run(fs, budget)
        if result.cost is not None:
            return result.cost, tried
        if budget >= cap or (upper is not None and budget >= upper):
            return None, tried
        k = 1 if k == 0 else 2 * k


def traceback(fs: FreeSpace, tables: ReachTables, end_vertex: int, end_copy: int) -> list[int]:
    """Observation indices KEPT by one optimal solution, in increasing order.

    Walks the recorded reachability backwards from the end vertex. At every
    edge it moves to the contributor that produced the edge's LOWEST reachable
    rank; since every contribution to an edge reaches up to the same free upper
    end, that contributor alone covers the whole reachable part, so the walk
    never needs a specific point. See ``docs/continuous.md`` section 3.
    """
    m, n = fs.m, fs.n
    v_reach, h_reach, vertex = tables.v_reach, tables.h_reach, tables.vertex
    v_lo, v_hi, v_ne = tables.v_lo, tables.v_hi, tables.v_ne
    h_lo, h_hi, h_ne = fs.h_lo, fs.h_hi, fs.h_ne

    kept = {end_vertex}
    state: tuple[str, int, int, int, int] = ("vertex", m - 1, end_vertex, end_copy, -1)
    for _ in range(4 * (m + n) * (tables.k + 2) + 8):
        kind, a, j, copy, g = state
        if kind == "vertex":
            if a == 0 and copy == j:
                return sorted(kept)
            if a >= 1 and h_reach[a - 1, j, copy] < BIG:
                state = ("h", a - 1, j, copy, -1)
                continue
            gaps = np.flatnonzero(v_reach[a, j, :, copy] < BIG)
            if len(gaps) == 0:
                raise AssertionError(f"traceback: vertex {(a, j, copy)} has no source")
            state = ("v", a, j, copy, int(gaps[0]))
            continue

        if kind == "h":
            target = h_reach[a, j, copy]
            if vertex[a, j, copy] and h_lo[a, j] == target:
                state = ("vertex", a, j, copy, -1)
                continue
            found = False
            for gap in range(v_reach.shape[2]):
                i, src = j - gap - 1, copy - gap
                if i < 0 or src < 0:
                    continue
                left = v_reach[a, j, gap, copy]
                bottom = h_reach[a, i, src]
                _, top = _cell_outputs(
                    np.array(left), np.array(bottom),
                    np.array(0), np.array(0), np.array(False),
                    np.array(h_lo[a, j]), np.array(h_hi[a, j]), np.array(h_ne[a, j]),
                )
                if int(top) == target:
                    if left < BIG:
                        state = ("v", a, j, copy, gap)
                    else:
                        kept.add(i)
                        state = ("h", a, i, src, -1)
                    found = True
                    break
            if not found:
                raise AssertionError(f"traceback: horizontal edge {(a, j, copy)} has no source")
            continue

        # kind == "v": the observation edge (j - g - 1) -> j at reference vertex a.
        i, src = j - g - 1, copy - g
        kept.add(i)
        target = v_reach[a, j, g, copy]
        if vertex[a, i, src] and v_lo[a, j, g] == target:
            state = ("vertex", a, i, src, -1)
            continue
        if a >= 1:
            left = v_reach[a - 1, j, g, copy]
            bottom = h_reach[a - 1, i, src]
            right, _ = _cell_outputs(
                np.array(left), np.array(bottom),
                np.array(v_lo[a, j, g]), np.array(v_hi[a, j, g]), np.array(v_ne[a, j, g]),
                np.array(0), np.array(0), np.array(False),
            )
            if int(right) == target:
                state = ("h", a - 1, i, src, -1) if bottom < BIG else ("v", a - 1, j, copy, g)
                continue
        raise AssertionError(f"traceback: vertical edge {(a, j, g, copy)} has no source")
    raise AssertionError("traceback did not terminate")
