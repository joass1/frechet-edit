"""Continuous deletion-only Frechet edit distance (paper Section 4.1, Theorem 3).

Optimality evidence is agreement with ``tests/oracles/continuous.py``, which
enumerates retained subsequences and decides continuous Frechet with its own
exact Alt-Godau sweep and a different exact comparison technique. Nothing in
this file lets the package check itself.
"""

from __future__ import annotations

import itertools
import math
import random
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest

from frechet_edit import (
    Deletion,
    EditResult,
    UnsupportedOperationError,
    continuous_edit_distance,
    continuous_frechet_le,
    continuous_frechet_within,
    discrete_edit_distance,
    verify_continuous_witness,
    verify_witness,
)
from frechet_edit import _interval as iv
from frechet_edit._freespace import dense_ranks
from frechet_edit._quadroot import ONE, ZERO, QuadNum, compare, sign_plus_root
from tests.oracles.continuous import deletion_oracle, frechet_le

getcontext().prec = 120

ALPHABET_1D = (0.0, 1.0, 2.5)


def _curves_1d(max_len, alphabet=ALPHABET_1D):
    for length in range(1, max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield np.array(combo, dtype=np.float64).reshape(-1, 1)


def _dec(q: QuadNum) -> Decimal:
    val = Decimal(q.x.numerator) / Decimal(q.x.denominator)
    if q.s:
        val += q.s * (Decimal(q.y.numerator) / Decimal(q.y.denominator)).sqrt()
    return val


def _grid_curve(rng: random.Random, span: int, dim: int, max_len: int) -> np.ndarray:
    """A random curve of 1..max_len integer points in [-span, span]^dim."""
    length = rng.randint(1, max_len)
    return np.array(
        [[rng.randint(-span, span) for _ in range(dim)] for _ in range(length)], dtype=float
    )


def _random_quad(rng: random.Random) -> QuadNum:
    return QuadNum(
        Fraction(rng.randint(-20, 20), rng.randint(1, 9)),
        rng.choice([-1, 0, 1]),
        Fraction(rng.randint(0, 50), rng.randint(1, 9)),
    )


def _cost(result: EditResult) -> float:
    if result.status == "infeasible":
        return math.inf
    assert result.status == "optimal", result
    return float(result.cost)


# ---------------------------------------------------------------------------
# Exact number kernel
# ---------------------------------------------------------------------------


class TestQuadRootComparison:
    def test_rationals(self):
        assert compare(QuadNum.rational(Fraction(1, 3)), QuadNum.rational(Fraction(2, 5))) == -1
        assert compare(ONE, ONE) == 0
        assert compare(ONE, ZERO) == 1

    def test_constructed_ties_are_exact_zeros(self):
        three = QuadNum.rational(3)
        assert compare(QuadNum(Fraction(1), 1, Fraction(4)), three) == 0
        root2 = QuadNum(Fraction(0), 1, Fraction(2))
        assert compare(root2, QuadNum(Fraction(0), 1, Fraction(2))) == 0
        # 1 + sqrt(2) == (1/2) + sqrt(2) + 1/2, written differently
        a = QuadNum(Fraction(1), 1, Fraction(2))
        b = QuadNum(Fraction(1, 2) + Fraction(1, 2), 1, Fraction(8, 4))
        assert compare(a, b) == 0
        # Both sides carry a root and share a sign, so `compare` must square
        # twice: 1 + sqrt(4) == sqrt(9) exactly, and sqrt(8) > 1 + sqrt(2).
        one_plus_root4 = QuadNum(Fraction(1), 1, Fraction(4))
        assert compare(one_plus_root4, QuadNum(Fraction(0), 1, Fraction(9))) == 0
        root8 = QuadNum(Fraction(0), 1, Fraction(8))
        assert compare(root8, QuadNum(Fraction(1), 1, Fraction(2))) == 1

    def test_near_ties_do_not_collapse(self):
        # sqrt(2) vs 1.41421356237309504880168872420969807856967187537694
        approx = QuadNum.rational(Fraction("1.41421356237309504880168872420969807856967187537694"))
        root2 = QuadNum(Fraction(0), 1, Fraction(2))
        assert compare(root2, approx) == 1
        assert compare(approx, root2) == -1

    def test_random_against_120_digit_decimal(self):
        rng = random.Random(5)
        for _ in range(4000):
            p, q = _random_quad(rng), _random_quad(rng)
            diff = _dec(p) - _dec(q)
            if abs(diff) > Decimal(10) ** -100:
                assert compare(p, q) == (1 if diff > 0 else -1), (p, q)
            else:
                assert compare(p, q) == 0, (p, q)

    def test_sign_plus_root_branches(self):
        assert sign_plus_root(Fraction(3), Fraction(-1), Fraction(9)) == 0
        assert sign_plus_root(Fraction(3), Fraction(-1), Fraction(10)) == -1
        assert sign_plus_root(Fraction(-3), Fraction(1), Fraction(8)) == -1
        assert sign_plus_root(Fraction(0), Fraction(-2), Fraction(1)) == -1
        assert sign_plus_root(Fraction(2), Fraction(0), Fraction(5)) == 1

    def test_invalid_triples_are_rejected(self):
        with pytest.raises(ValueError):
            QuadNum(Fraction(0), 2, Fraction(1))
        with pytest.raises(ValueError):
            QuadNum(Fraction(0), 1, Fraction(-1))


class TestIntervalEnclosures:
    """Every enclosure must contain the exact value, including near underflow."""

    @pytest.mark.parametrize("scale", [1.0, 1e-150, 1e-160, 1e150])
    def test_arithmetic_encloses_exact_results(self, scale):
        rng = np.random.default_rng(3)
        a = rng.uniform(-5, 5, 400) * scale
        b = rng.uniform(-5, 5, 400) * scale
        c = rng.uniform(0.1, 5, 400) * scale
        x = iv.sub_exact(a, b)
        prod = iv.mul(x, iv.exact(c))
        sq = iv.sqr(x)
        quo = iv.div_pos(iv.exact(a), iv.exact(c))
        root = iv.sqrt_nonneg(iv.exact(c))
        for k in range(400):
            fa, fb, fc = Fraction(a[k]), Fraction(b[k]), Fraction(c[k])
            for enc, exact in (
                (x, fa - fb),
                (prod, (fa - fb) * fc),
                (sq, (fa - fb) ** 2),
                (quo, fa / fc),
            ):
                lo, hi = enc[0][k], enc[1][k]
                if math.isfinite(lo):
                    assert Fraction(lo) <= exact
                if math.isfinite(hi):
                    assert exact <= Fraction(hi)
            lo, hi = root[0][k], root[1][k]
            assert Fraction(lo) ** 2 <= fc <= Fraction(hi) ** 2

    def test_nan_is_widened_to_the_whole_line(self):
        lo, hi = iv.sanitize((np.array([np.nan, 1.0]), np.array([2.0, np.nan])))
        assert lo[0] == -np.inf and hi[1] == np.inf

    def test_division_by_an_enclosure_touching_zero_is_unbounded(self):
        lo, hi = iv.div_pos(iv.exact(np.array([1.0])), (np.array([0.0]), np.array([1.0])))
        assert lo[0] == -np.inf and hi[0] == np.inf


class TestDenseRanks:
    def test_ties_share_ranks_and_order_is_exact(self):
        values = [
            ZERO,
            ONE,
            QuadNum(Fraction(1, 2), 1, Fraction(1, 16)),  # 3/4
            QuadNum.rational(Fraction(3, 4)),
            QuadNum(Fraction(0), 1, Fraction(1, 2)),  # 0.7071...
            ZERO,
            QuadNum(Fraction(1), -1, Fraction(0)),  # 1
        ]
        flo = np.array([float(_dec(v)) for v in values])
        # Deliberately wide, overlapping enclosures force the exact path.
        ranks = dense_ranks(flo - 0.3, flo + 0.3, lambda i: values[i])
        assert list(ranks) == [0, 3, 2, 2, 1, 0, 3]

    def test_zero_width_clusters_group_by_value(self):
        flo = np.array([1.0, 0.0, 1.0, 0.5, 0.0])
        ranks = dense_ranks(flo, flo.copy(), lambda i: pytest.fail("exact path not expected"))
        assert list(ranks) == [2, 0, 2, 1, 0]

    def test_random_against_exact_sort(self):
        rng = random.Random(9)
        for _ in range(60):
            vals = [
                QuadNum(Fraction(rng.randint(0, 6), 4), rng.choice([-1, 0, 1]),
                        Fraction(rng.choice([0, 1, 2, 4, 9]), 16))
                for _ in range(12)
            ]
            mids = np.array([float(_dec(v)) for v in vals])
            width = rng.choice([0.0, 1e-9, 0.2])
            ranks = dense_ranks(mids - width, mids + width, lambda i, vals=vals: vals[i])
            for i, j in itertools.combinations(range(len(vals)), 2):
                assert np.sign(ranks[i] - ranks[j]) == compare(vals[i], vals[j])


# ---------------------------------------------------------------------------
# Ordinary continuous Frechet (budget 0)
# ---------------------------------------------------------------------------


class TestOrdinaryDecision:
    def test_sampling_density_does_not_matter(self):
        sparse = np.array([[0.0, 0.0], [10.0, 0.0]])
        dense = np.column_stack([np.linspace(0, 10, 41), np.zeros(41)])
        assert continuous_frechet_within(sparse, dense, 1e-9)
        assert continuous_frechet_within(dense, sparse, 1e-9)

    def test_closed_at_an_exact_tangency(self):
        # The segment y = 1 touches the unit circle around the origin at one
        # point: the free interval degenerates to a single parameter.
        ref = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        obs = np.array([[-1.0, 1.0], [1.0, 1.0]])
        assert continuous_frechet_within(ref, obs, 1.0)
        assert not continuous_frechet_within(ref, obs, float(np.nextafter(1.0, 0.0)))

    def test_a_path_through_a_single_point_needs_the_closed_cell_comparison(self):
        """At reference vertex 3 the observation's free window is t <= 0.2 and
        the lowest point still reachable from vertex 2 is exactly t = 0.2, with
        the cell's bottom edge unreachable: the only monotone path crosses the
        cell horizontally through that one point. A strict `>` in the cell
        rule loses it. Found by mutation testing, which the rest of this file
        did not catch; swapping the curves exercises the mirrored top-edge rule.
        """
        ref = np.array([[0.0], [3.0], [1.0], [10.0]])
        obs = np.array([[0.0], [10.0]])
        for a, b in ((ref, obs), (obs, ref)):
            assert frechet_le(a, b, 1.0)
            assert continuous_frechet_within(a, b, 1.0)
            assert not continuous_frechet_within(a, b, float(np.nextafter(1.0, 0.0)))
        # An overshoot to 30 must be deleted; a detour to 7 would not need to
        # be, since in one dimension it lies on the way from 0 to 10.
        spiked = np.array([[0.0], [30.0], [10.0]])
        assert continuous_edit_distance(ref, spiked, 1.0).cost == 1 == deletion_oracle(
            ref, spiked, 1.0
        )

    @pytest.mark.parametrize("delta", [0.4, 0.5, 1.0, 1.25, 2.0])
    def test_exhaustive_1d_against_two_independent_deciders(self, delta):
        for ref in _curves_1d(3):
            for obs in _curves_1d(3):
                expected = frechet_le(ref, obs, delta)
                assert continuous_frechet_within(ref, obs, delta) == expected, (ref, obs)
                assert continuous_frechet_le(ref, obs, delta) == expected, (ref, obs)

    def test_random_2d_integer_grid_with_tight_deltas(self):
        rng = random.Random(21)
        for _ in range(250):
            ref, obs = _grid_curve(rng, 3, 2, 5), _grid_curve(rng, 3, 2, 5)
            delta = rng.choice([0.5, 1.0, 2.0, 2.5, math.sqrt(2.0), math.sqrt(5.0)])
            expected = frechet_le(ref, obs, delta)
            assert continuous_frechet_within(ref, obs, delta) == expected, (ref, obs, delta)
            assert continuous_frechet_le(ref, obs, delta) == expected

    def test_symmetry(self):
        rng = np.random.default_rng(8)
        for _ in range(100):
            a = rng.integers(-3, 4, (int(rng.integers(1, 5)), 2)).astype(float)
            b = rng.integers(-3, 4, (int(rng.integers(1, 5)), 2)).astype(float)
            for delta in (1.0, 2.0):
                forward = continuous_frechet_within(a, b, delta)
                assert forward == continuous_frechet_within(b, a, delta)


# ---------------------------------------------------------------------------
# Continuous deletion edit distance
# ---------------------------------------------------------------------------


class TestDeletionAgainstTheOracle:
    @pytest.mark.parametrize("delta", [0.4, 1.0, 1.25])
    def test_exhaustive_1d_small(self, delta):
        for ref in _curves_1d(2):
            for obs in _curves_1d(4):
                got = continuous_edit_distance(ref, obs, delta)
                assert _cost(got) == deletion_oracle(ref, obs, delta), (ref.ravel(), obs.ravel())

    @pytest.mark.slow
    @pytest.mark.parametrize("delta", [0.4, 0.5, 1.0, 1.25, 2.0])
    def test_exhaustive_1d_larger(self, delta):
        for ref in _curves_1d(3):
            for obs in _curves_1d(4):
                got = continuous_edit_distance(ref, obs, delta)
                assert _cost(got) == deletion_oracle(ref, obs, delta), (ref.ravel(), obs.ravel())

    def test_random_2d_with_witnesses(self):
        rng = random.Random(4)
        for _ in range(160):
            ref, obs = _grid_curve(rng, 3, 2, 4), _grid_curve(rng, 3, 2, 6)
            delta = rng.choice([1.0, 1.5, 2.0, math.sqrt(2.0), 3.0])
            got = continuous_edit_distance(ref, obs, delta, return_witness=True)
            assert _cost(got) == deletion_oracle(ref, obs, delta), (ref, obs, delta)
            if got.status == "optimal":
                assert verify_continuous_witness(ref, obs, got).ok
                kept = got.edited_curve
                assert frechet_le(ref, kept, delta)

    def test_random_3d(self):
        rng = random.Random(33)
        for _ in range(40):
            ref, obs = _grid_curve(rng, 2, 3, 3), _grid_curve(rng, 2, 3, 5)
            delta = rng.choice([1.0, 2.0, 2.5])
            got = continuous_edit_distance(ref, obs, delta)
            assert _cost(got) == deletion_oracle(ref, obs, delta)


class TestPaperEdgeCases:
    def test_waiting_at_the_last_vertex_through_several_reference_vertices(self):
        """The only solution keeps just the LAST observation vertex while the
        reference visits three vertices. That path waits at a DAG vertex with
        no outgoing edge, which no cell of the product complex carries, so a
        cells-only reading of Section 4.1 reports this instance infeasible.
        The explicit vertex propagation in ``_continuous`` exists for it."""
        ref = np.array([[0.0], [1.0], [0.0]])
        obs = np.array([[9.0], [0.5]])
        assert deletion_oracle(ref, obs, 0.5) == 1
        result = continuous_edit_distance(ref, obs, 0.5, return_witness=True)
        assert (result.status, result.cost) == ("optimal", 1)
        assert result.edits == (Deletion(0),)
        assert verify_witness(ref, obs, result).ok

    def test_single_reference_vertex(self):
        ref = np.array([[0.0, 0.0]])
        obs = np.array([[5.0, 0.0], [0.5, 0.0], [0.0, 0.5], [7.0, 7.0]])
        result = continuous_edit_distance(ref, obs, 1.0, return_witness=True)
        assert result.cost == 2
        assert verify_witness(ref, obs, result).ok

    def test_single_observation_vertex(self):
        ref = np.array([[0.0], [2.0]])
        assert continuous_edit_distance(ref, np.array([[1.0]]), 1.0).cost == 0
        assert continuous_edit_distance(ref, np.array([[1.0]]), 0.9).status == "infeasible"

    def test_prefix_and_suffix_deletion(self):
        ref = np.array([[0.0, 0.0], [4.0, 0.0]])
        obs = np.array([[-9.0, 9.0], [0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [9.0, 9.0]])
        result = continuous_edit_distance(ref, obs, 0.5, return_witness=True)
        assert result.cost == 2
        assert {e.index for e in result.edits} == {0, 4}

    def test_duplicate_and_degenerate_segments(self):
        ref = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
        obs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [8.0, 8.0], [3.0, 0.0]])
        result = continuous_edit_distance(ref, obs, 0.25, return_witness=True)
        assert _cost(result) == deletion_oracle(ref, obs, 0.25) == 1
        assert verify_witness(ref, obs, result).ok


class TestProperties:
    def _random_pair(self, rng):
        ref = rng.integers(-4, 5, (int(rng.integers(1, 5)), 2)).astype(float)
        obs = rng.integers(-4, 5, (int(rng.integers(1, 7)), 2)).astype(float)
        return ref, obs

    def test_never_more_than_discrete_deletion(self):
        rng = np.random.default_rng(1)
        for _ in range(150):
            ref, obs = self._random_pair(rng)
            for delta in (1.0, 2.0, 3.0):
                cont = _cost(continuous_edit_distance(ref, obs, delta))
                disc = _cost(discrete_edit_distance(ref, obs, delta, operations="delete"))
                assert cont <= disc

    def test_zero_exactly_when_ordinary_continuous_frechet_is_within_delta(self):
        rng = np.random.default_rng(2)
        for _ in range(150):
            ref, obs = self._random_pair(rng)
            delta = 2.0
            zero = continuous_edit_distance(ref, obs, delta).cost == 0
            assert zero == continuous_frechet_within(ref, obs, delta)

    def test_non_increasing_in_delta(self):
        rng = np.random.default_rng(3)
        for _ in range(80):
            ref, obs = self._random_pair(rng)
            costs = [_cost(continuous_edit_distance(ref, obs, d)) for d in (0.5, 1.0, 2.0, 4.0)]
            assert costs == sorted(costs, reverse=True)

    def test_reversing_both_curves_preserves_the_cost(self):
        rng = np.random.default_rng(4)
        for _ in range(100):
            ref, obs = self._random_pair(rng)
            forward = _cost(continuous_edit_distance(ref, obs, 1.5))
            backward = _cost(continuous_edit_distance(ref[::-1], obs[::-1], 1.5))
            assert forward == backward

    def test_exact_power_of_two_rescaling_preserves_the_cost(self):
        rng = np.random.default_rng(5)
        for _ in range(60):
            ref, obs = self._random_pair(rng)
            base = _cost(continuous_edit_distance(ref, obs, 1.5))
            for scale in (2.0**-30, 2.0**40, 2.0**-535):
                scaled = continuous_edit_distance(ref * scale, obs * scale, 1.5 * scale)
                assert _cost(scaled) == base


class TestWitnessRejection:
    REF = np.array([[0.0, 0.0], [10.0, 0.0]])
    OBS = np.array([[0.0, 0.0], [4.0, 0.0], [5.0, 30.0], [6.0, 0.0], [10.0, 0.0]])

    def _result(self):
        result = continuous_edit_distance(self.REF, self.OBS, 1.0, return_witness=True)
        assert verify_witness(self.REF, self.OBS, result).ok
        return result

    def test_deleting_the_wrong_vertex_is_rejected(self):
        good = self._result()
        tampered = EditResult(
            **{**_fields(good), "edits": (Deletion(1),), "edited_curve": np.delete(self.OBS, 1, 0)}
        )
        report = verify_continuous_witness(self.REF, self.OBS, tampered)
        assert not report.ok and any("exceeds delta" in v for v in report.violations)

    def test_a_cost_that_does_not_match_the_edits_is_rejected(self):
        tampered = EditResult(**{**_fields(self._result()), "cost": 0})
        assert not verify_continuous_witness(self.REF, self.OBS, tampered).ok

    def test_an_edited_curve_that_is_not_the_replay_is_rejected(self):
        good = self._result()
        tampered = EditResult(**{**_fields(good), "edited_curve": good.edited_curve[::-1].copy()})
        assert not verify_continuous_witness(self.REF, self.OBS, tampered).ok

    def test_deleting_everything_is_rejected(self):
        good = self._result()
        edits = tuple(Deletion(i) for i in range(len(self.OBS)))
        tampered = EditResult(
            **{**_fields(good), "edits": edits, "cost": len(edits),
               "edited_curve": np.empty((0, 2))}
        )
        assert not verify_continuous_witness(self.REF, self.OBS, tampered).ok

    def test_no_witness_means_nothing_to_verify(self):
        result = continuous_edit_distance(self.REF, self.OBS, 1.0)
        assert not verify_continuous_witness(self.REF, self.OBS, result).ok


def _fields(result: EditResult) -> dict:
    return {
        name: getattr(result, name)
        for name in (
            "status", "cost", "mode", "delta", "dimension", "backend", "numeric_policy",
            "witness_status", "edited_curve", "edits", "coupling", "stats", "detail",
        )
    }


class TestApiContract:
    def test_insertions_are_refused_not_substituted(self):
        with pytest.raises(UnsupportedOperationError):
            continuous_edit_distance([[0.0]], [[0.0]], 1.0, operations="insert")
        with pytest.raises(UnsupportedOperationError):
            continuous_edit_distance([[0.0]], [[0.0]], 1.0, operations="both")

    @pytest.mark.parametrize("bad", [-1, 1.5, "3", True])
    def test_max_deletions_must_be_a_non_negative_integer(self, bad):
        with pytest.raises(ValueError):
            continuous_edit_distance([[0.0]], [[0.0]], 1.0, max_deletions=bad)

    def test_budget_exceeded_is_not_infeasible(self):
        ref = np.array([[0.0], [10.0]])
        obs = np.array([[0.0], [50.0], [-50.0], [60.0], [10.0]])
        full = continuous_edit_distance(ref, obs, 1.0)
        assert (full.status, full.cost) == ("optimal", 3)
        capped = continuous_edit_distance(ref, obs, 1.0, max_deletions=2)
        assert capped.status == "budget_exceeded" and capped.cost is None
        assert not capped.is_feasible
        payload = capped.to_json_obj()
        assert payload["status"] == "budget_exceeded" and payload["cost_is_infinite"] is False
        assert continuous_edit_distance(ref, obs, 1.0, max_deletions=3).cost == 3

    def test_infeasible_is_reported_with_infinite_cost(self):
        result = continuous_edit_distance([[0.0], [1.0]], [[5.0], [6.0]], 1.0)
        assert result.status == "infeasible" and math.isinf(result.cost)

    def test_result_metadata(self):
        result = continuous_edit_distance([[0.0], [1.0]], [[0.0], [1.0]], 0.5)
        assert (result.backend, result.mode, result.numeric_policy) == (
            "continuous", "delete", "certified",
        )
        assert result.coupling is None

    def test_inputs_are_validated_and_not_mutated(self):
        ref = np.array([[0.0, 0.0], [1.0, 0.0]])
        obs = np.array([[0.0, 0.0], [9.0, 9.0], [1.0, 0.0]])
        before = (ref.copy(), obs.copy())
        continuous_edit_distance(ref, obs, 0.5, return_witness=True)
        assert np.array_equal(ref, before[0]) and np.array_equal(obs, before[1])
        with pytest.raises(ValueError):
            continuous_edit_distance(ref, obs, 0.0)
        with pytest.raises(ValueError):
            continuous_edit_distance(ref, [[0.0, 0.0, 0.0]], 1.0)
        with pytest.raises(ValueError):
            continuous_frechet_within(ref, [[np.nan, 0.0]], 1.0)

    def test_one_dimensional_scalars_are_accepted(self):
        assert continuous_edit_distance([0.0, 2.0], [0.0, 5.0, 2.0], 0.5).cost == 1
