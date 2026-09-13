"""The monotone minimum queue, against a plain list-scan model.

The queue is where an optimised sliding-window minimum usually goes wrong:
expiry off by one, argmin drifting on ties, or infinities poisoning the deque.
Every test here compares against an obviously-correct rescan.
"""

from __future__ import annotations

import math
import random

import pytest

from frechet_edit._minqueue import MinQueue


def scan_min(values, lo, hi):
    """Reference: minimum of ``values[lo..hi]`` inclusive, by rescanning."""
    window = values[lo : hi + 1]
    return min(window) if window else math.inf


def scan_argmin(values, lo, hi):
    """Reference argmin, earliest index on ties.

    An all-infinite window still has an argmin: the earliest index. Seeding the
    search with ``inf`` instead of ``None`` would wrongly report no minimum,
    because ``inf < inf`` is false.
    """
    best_i = None
    for i in range(lo, hi + 1):
        if best_i is None or values[i] < values[best_i]:
            best_i = i
    return best_i


class TestBasics:
    def test_empty_queue_reports_infinity(self):
        q = MinQueue()
        assert q.empty
        assert q.min_value() == math.inf
        assert q.argmin() is None

    def test_indices_must_ascend(self):
        q = MinQueue()
        q.push(3, 1.0)
        with pytest.raises(ValueError, match="ascend"):
            q.push(3, 0.0)
        with pytest.raises(ValueError, match="ascend"):
            q.push(2, 0.0)

    def test_clear_resets_the_index_guard(self):
        q = MinQueue()
        q.push(5, 1.0)
        q.clear()
        q.push(0, 2.0)  # must not raise
        assert q.min_value() == 2.0

    def test_expiry_removes_only_old_indices(self):
        q = MinQueue()
        for i, v in enumerate([5.0, 3.0, 4.0, 1.0]):
            q.push(i, v)
        assert q.min_value() == 1.0
        q.expire_before(4)
        assert q.empty and q.min_value() == math.inf

    def test_expiry_of_the_current_minimum_exposes_the_next(self):
        q = MinQueue()
        q.push(0, 1.0)
        q.push(1, 2.0)
        q.push(2, 3.0)
        assert q.argmin() == 0
        q.expire_before(1)
        assert q.min_value() == 2.0 and q.argmin() == 1
        q.expire_before(2)
        assert q.min_value() == 3.0 and q.argmin() == 2


class TestTiesAndInfinities:
    def test_equal_values_keep_the_earliest_index(self):
        q = MinQueue()
        q.push(0, 2.0)
        q.push(1, 2.0)
        q.push(2, 2.0)
        assert q.argmin() == 0
        q.expire_before(1)
        assert q.argmin() == 1

    def test_all_infinite_window(self):
        q = MinQueue()
        for i in range(5):
            q.push(i, math.inf)
        assert q.min_value() == math.inf
        assert q.argmin() == 0

    def test_infinity_never_masks_a_finite_value(self):
        q = MinQueue()
        q.push(0, math.inf)
        q.push(1, 7.0)
        q.push(2, math.inf)
        assert q.min_value() == 7.0 and q.argmin() == 1


class TestAgainstScanModel:
    @pytest.mark.parametrize("seed", range(12))
    def test_random_sliding_windows(self, seed):
        rng = random.Random(seed)
        n = rng.randint(1, 60)
        values = [
            math.inf if rng.random() < 0.2 else float(rng.randint(0, 8)) for _ in range(n)
        ]
        # Non-decreasing window lower bounds, as mu guarantees.
        lows = []
        lo = 0
        for i in range(n):
            lo = min(i, lo + (1 if rng.random() < 0.35 else 0))
            lows.append(lo)

        q = MinQueue()
        for i in range(n):
            q.push(i, values[i])
            q.expire_before(lows[i])
            assert q.min_value() == scan_min(values, lows[i], i), (i, lows[i])
            assert q.argmin() == scan_argmin(values, lows[i], i), (i, lows[i])

    def test_amortised_work_is_linear(self):
        """Each index is pushed once and popped at most once."""
        q = MinQueue()
        n = 2000
        for i in range(n):
            q.push(i, float(n - i))  # strictly decreasing: worst case for pops
            q.expire_before(max(0, i - 10))
        assert q.pushes == n
        assert q.pops <= n
