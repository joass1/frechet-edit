"""Self-checks for the continuous oracle, against facts that need no free space.

Everything in ``tests/unit/test_continuous.py`` trusts ``tests/oracles/continuous.py``,
so the oracle itself is held to closed-form results and to inequalities with the
(separately self-checked) discrete oracle:

* two segments: ``d_F`` is the larger endpoint distance (linear interpolation of
  both is optimal, and no coupling can do better than its endpoints);
* a point against a curve: the farthest vertex, by convexity;
* ``d_F <= d_DF`` for the vertex sequences, and ``d_F >= d_DF - h`` style
  bounds after densifying both curves at spacing ``h``;
* symmetry, and the exact comparator against 120-digit decimal arithmetic.
"""

from __future__ import annotations

import itertools
import random
from decimal import Decimal, getcontext
from fractions import Fraction

import pytest

from tests.oracles.continuous import _cmp, deletion_oracle, frechet_le
from tests.oracles.frechet import discrete_frechet_le

getcontext().prec = 120


def _dec(r):
    x, s, y = r
    val = Decimal(x.numerator) / Decimal(x.denominator)
    if s:
        val += s * (Decimal(y.numerator) / Decimal(y.denominator)).sqrt()
    return val


def _sq(u, v):
    return sum((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(u, v, strict=True))


class TestComparator:
    def test_constructed_equalities(self):
        cases = [
            ((Fraction(1), 1, Fraction(4)), (Fraction(3), 0, Fraction(0))),
            ((Fraction(0), 1, Fraction(2)), (Fraction(0), 1, Fraction(8, 4))),
            ((Fraction(1, 2), -1, Fraction(1, 4)), (Fraction(0), 0, Fraction(0))),
            ((Fraction(5), -1, Fraction(0)), (Fraction(5), 1, Fraction(0))),
        ]
        for a, b in cases:
            assert _cmp(a, b) == 0 and _cmp(b, a) == 0

    def test_random_against_decimal(self):
        rng = random.Random(11)
        for _ in range(3000):
            a = (Fraction(rng.randint(-9, 9), rng.randint(1, 7)), rng.choice([-1, 0, 1]),
                 Fraction(rng.randint(0, 30), rng.randint(1, 9)))
            b = (Fraction(rng.randint(-9, 9), rng.randint(1, 7)), rng.choice([-1, 0, 1]),
                 Fraction(rng.randint(0, 30), rng.randint(1, 9)))
            diff = _dec(a) - _dec(b)
            got = _cmp(a, b)
            if abs(diff) > Decimal(10) ** -100:
                assert got == (1 if diff > 0 else -1), (a, b)
            else:
                assert got == 0, (a, b)


class TestClosedForms:
    @pytest.mark.parametrize("seed", range(40))
    def test_two_segments_is_the_larger_endpoint_distance(self, seed):
        rng = random.Random(seed)
        P = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(2)]
        Q = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(2)]
        exact2 = max(_sq(P[0], Q[0]), _sq(P[1], Q[1]))
        for delta in (0.5, 1.0, 1.5, 2.0, 3.0, 4.5):
            assert frechet_le(P, Q, delta) == (exact2 <= Fraction(delta) ** 2)

    def test_point_against_curve_is_the_farthest_vertex(self):
        P = [(0, 0)]
        Q = [(1, 0), (0, 2), (-1, -1)]
        assert frechet_le(P, Q, 2.0) is True
        assert frechet_le(P, Q, 1.99) is False
        assert frechet_le(Q, P, 2.0) is True

    def test_backtracking_costs_the_detour(self):
        # Q walks to 1 and back to 0 before going on to 2; P goes straight.
        # Strong Frechet cannot rewind P, so the dip back to 0 must be matched
        # while P is at or beyond the point matched with the peak at 1.
        P = [(0,), (2,)]
        Q = [(0,), (1,), (0,), (2,)]
        assert frechet_le(P, Q, 0.5) is True
        assert frechet_le(P, Q, 0.49) is False


class TestInequalitiesWithDiscrete:
    ALPHABET = (0.0, 1.0, 2.5)

    def _curves(self, max_len):
        for length in range(1, max_len + 1):
            yield from itertools.product(self.ALPHABET, repeat=length)

    @pytest.mark.parametrize("delta", [0.5, 1.0, 1.25, 1.5])
    def test_continuous_never_exceeds_discrete(self, delta):
        for P in self._curves(3):
            for Q in self._curves(3):
                Pc, Qc = [(x,) for x in P], [(x,) for x in Q]
                if discrete_frechet_le(Pc, Qc, delta):
                    assert frechet_le(Pc, Qc, delta), (P, Q)

    @pytest.mark.parametrize("seed", range(30))
    def test_dense_resampling_converges_from_above(self, seed):
        """d_F <= d_DF(dense) <= d_F + h, where h bounds the resampling step."""
        rng = random.Random(seed)
        P = [(Fraction(rng.randint(-4, 4)), Fraction(rng.randint(-4, 4))) for _ in range(3)]
        Q = [(Fraction(rng.randint(-4, 4)), Fraction(rng.randint(-4, 4))) for _ in range(3)]
        steps = 24

        def dense(curve):
            out = []
            for a, b in itertools.pairwise(curve):
                for k in range(steps):
                    t = Fraction(k, steps)
                    out.append(tuple(a[d] + t * (b[d] - a[d]) for d in range(2)))
            out.append(curve[-1])
            return out

        seg = max(float(_sq(a, b)) ** 0.5 for c in (P, Q) for a, b in itertools.pairwise(c))
        h = seg / steps
        for delta in (1.0, 2.0, 3.0):
            if frechet_le(P, Q, delta):
                assert discrete_frechet_le(dense(P), dense(Q), delta + h + 1e-9)
            else:
                assert not discrete_frechet_le(dense(P), dense(Q), delta)

    @pytest.mark.parametrize("seed", range(20))
    def test_symmetry(self, seed):
        rng = random.Random(100 + seed)
        P = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(rng.randint(1, 4))]
        Q = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(rng.randint(1, 4))]
        for delta in (0.5, 1.0, 2.0, 3.0):
            assert frechet_le(P, Q, delta) == frechet_le(Q, P, delta)


class TestDeletionOracle:
    def test_deleting_a_spike(self):
        P = [(0.0, 0.0), (10.0, 0.0)]
        Q = [(0.0, 0.0), (4.0, 0.0), (5.0, 30.0), (6.0, 0.0), (10.0, 0.0)]
        assert deletion_oracle(P, Q, 1.0) == 1

    def test_everything_far_is_infeasible(self):
        assert deletion_oracle([(0.0,)], [(5.0,), (6.0,)], 1.0) == float("inf")

    def test_a_single_retained_vertex_counts(self):
        assert deletion_oracle([(0.0,), (1.0,), (0.0,)], [(9.0,), (0.5,)], 0.5) == 1
