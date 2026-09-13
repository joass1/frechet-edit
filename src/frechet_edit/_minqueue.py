"""Monotone minimum queue over a sliding window with non-decreasing bounds.

The insertion branch of the recurrence needs

    P(i, j) = 1 + min { F(k-1, j) : mu(i) <= k <= i }

Both window ends are non-decreasing in ``i`` because ``mu`` is non-decreasing,
so the classic monotone deque applies: every index is pushed once and popped
at most once, giving amortised O(1) per state instead of an O(m) rescan.

Two details are easy to get wrong and are tested directly in
``tests/unit/test_minqueue.py``:

* the stored priority is ``F(k-1, j)``, NOT ``F(k, j)`` - the caller pushes the
  index ``k-1`` it will later want back;
* on equal values the EARLIER index is kept, so ``argmin`` is deterministic.
"""

from __future__ import annotations

import math
from collections import deque


class MinQueue:
    """Sliding-window minimum with argmin, over indices pushed in ascending order."""

    __slots__ = ("_dq", "_last_index", "pops", "pushes")

    def __init__(self) -> None:
        self._dq: deque[tuple[int, float]] = deque()
        self._last_index: int | None = None
        self.pushes = 0
        self.pops = 0

    def __len__(self) -> int:
        return len(self._dq)

    @property
    def empty(self) -> bool:
        return not self._dq

    def clear(self) -> None:
        self._dq.clear()
        self._last_index = None

    def push(self, index: int, value: float) -> None:
        """Append ``index`` with priority ``value``. Indices must ascend."""
        if self._last_index is not None and index <= self._last_index:
            raise ValueError(
                f"MinQueue indices must strictly ascend; pushed {index} after {self._last_index}"
            )
        self._last_index = index
        # Strict '>' keeps the earliest index among equal values.
        while self._dq and self._dq[-1][1] > value:
            self._dq.pop()
            self.pops += 1
        self._dq.append((index, value))
        self.pushes += 1

    def expire_before(self, lo: int) -> None:
        """Drop every entry whose index is below ``lo``."""
        while self._dq and self._dq[0][0] < lo:
            self._dq.popleft()
            self.pops += 1

    def min_value(self) -> float:
        """Smallest priority in the window, or ``inf`` when the window is empty."""
        return self._dq[0][1] if self._dq else math.inf

    def argmin(self) -> int | None:
        """Index attaining :meth:`min_value`, or ``None`` when the window is empty."""
        return self._dq[0][0] if self._dq else None
