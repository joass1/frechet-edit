"""Property-based invariants from docs/definition.md.

Deliberately absent, because they are NOT properties of this measure:
symmetry, the triangle inequality, identity of indiscernibles, monotonicity
under resampling, and equality between the returned cost and the number of
corruptions that were injected.
"""

from __future__ import annotations

import math
import operator
from fractions import Fraction

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from frechet_edit import discrete_edit_distance, ordinary_discrete_frechet, verify_witness

SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

coords = st.floats(
    min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False, width=32
)
deltas = st.floats(min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False)

# Powers of two only: multiplying a float64 by one shifts the exponent and
# leaves the significand alone, so the scaling is exact and its invariance is a
# statement about the measure rather than about rounding. The range is chosen so
# that neither `coords` (|x| <= 50) nor `deltas` (<= 20) can overflow or go
# subnormal after scaling.
exact_scales = st.sampled_from([float(2.0**k) for k in range(-10, 11)])


def curves_1d(min_size=1, max_size=6):
    return st.lists(coords, min_size=min_size, max_size=max_size).map(
        lambda xs: np.array(xs, dtype=np.float64).reshape(-1, 1)
    )


def curves_2d(min_size=1, max_size=5):
    return st.lists(
        st.tuples(coords, coords), min_size=min_size, max_size=max_size
    ).map(lambda ps: np.array(ps, dtype=np.float64))


def _cost(result):
    return math.inf if result.status == "infeasible" else float(result.cost)


def _transform_is_exact(op, points, operand) -> bool:
    """Did applying ``op`` to every coordinate round at all?

    Compares the float64 result against the same operation carried out in exact
    rationals. When they agree everywhere, the transform moved the geometry
    without altering it, and an invariance assertion about it is meaningful.
    When they do not, float64 has quietly produced a different point set and
    any "invariance" would be a statement about rounding, not about the measure.
    """
    operand_array = np.broadcast_to(np.asarray(operand, dtype=np.float64), points.shape)
    for value, other in zip(points.ravel(), operand_array.ravel(), strict=True):
        if Fraction(float(op(value, other))) != op(Fraction(float(value)), Fraction(float(other))):
            return False
    return True


class TestCoreInvariants:
    @SETTINGS
    @given(ref=curves_1d(), obs=curves_1d(), delta=deltas)
    def test_mixed_is_feasible_and_bounded_by_m_plus_n(self, ref, obs, delta):
        result = discrete_edit_distance(ref, obs, delta, operations="both")
        assume(result.status != "numerically_ambiguous")
        assert result.status == "optimal"
        assert 0 <= result.cost <= len(ref) + len(obs)

    @SETTINGS
    @given(ref=curves_1d(), obs=curves_1d(), delta=deltas)
    def test_mixed_dominates_each_restricted_mode(self, ref, obs, delta):
        costs = {}
        for mode in ("delete", "insert", "both"):
            result = discrete_edit_distance(ref, obs, delta, operations=mode)
            assume(result.status != "numerically_ambiguous")
            costs[mode] = _cost(result)
        assert costs["both"] <= costs["delete"]
        assert costs["both"] <= costs["insert"]

    @SETTINGS
    @given(ref=curves_1d(), obs=curves_1d(), delta=deltas)
    def test_zero_cost_exactly_when_ordinary_frechet_fits(self, ref, obs, delta):
        result = discrete_edit_distance(ref, obs, delta, operations="both")
        assume(result.status == "optimal")
        fits = ordinary_discrete_frechet(ref, obs) <= delta
        assert (result.cost == 0) == fits

    @SETTINGS
    @given(
        ref=curves_1d(),
        obs=curves_1d(),
        delta=deltas,
        bump=st.floats(min_value=1.01, max_value=50.0),
    )
    def test_cost_is_non_increasing_in_delta(self, ref, obs, delta, bump):
        for mode in ("delete", "insert", "both"):
            small = discrete_edit_distance(ref, obs, delta, operations=mode)
            large = discrete_edit_distance(ref, obs, delta * bump, operations=mode)
            assume("numerically_ambiguous" not in (small.status, large.status))
            assert _cost(large) <= _cost(small), (mode, delta, bump)

    @SETTINGS
    @given(ref=curves_1d(), obs=curves_1d(), delta=deltas, scale=exact_scales)
    def test_common_scaling_of_coordinates_and_delta_preserves_cost(
        self, ref, obs, delta, scale
    ):
        """Scaling coordinates and delta together by an EXACT factor.

        The factors are powers of two on purpose. Multiplying a float64 by a
        power of two only shifts its exponent, so the scaled problem really is
        the original problem resized. An arbitrary factor rounds every
        coordinate independently, which changes the geometry rather than
        resizing it, exactly as the translation test below documents for
        addition. Filtering arbitrary factors down to the exact ones was tried
        and discarded: it threw away roughly 94 percent of inputs and
        Hypothesis rightly flagged the resulting distribution as distorted.
        """
        assume(_transform_is_exact(operator.mul, ref, scale))
        assume(_transform_is_exact(operator.mul, obs, scale))
        assume(_transform_is_exact(operator.mul, np.array([[delta]]), scale))
        base = discrete_edit_distance(ref, obs, delta, operations="both")
        scaled = discrete_edit_distance(ref * scale, obs * scale, delta * scale, operations="both")
        assume("numerically_ambiguous" not in (base.status, scaled.status))
        assert _cost(base) == _cost(scaled)

    @SETTINGS
    @given(ref=curves_2d(), obs=curves_2d(), delta=deltas, shift=st.tuples(coords, coords))
    def test_common_translation_preserves_cost(self, ref, obs, delta, shift):
        """Translating both curves by the same EXACTLY REPRESENTABLE offset.

        The exactness precondition is not a formality. float64 addition is
        lossy, so a large enough offset silently destroys a small coordinate,
        and the translated problem is then a genuinely different geometry
        rather than the same one moved. Hypothesis found this:

            ref = [(0, 0)],  obs = [(1.4e-45, 1)],  delta = 1,  shift = (1, 0)

        Exactly, ``dist^2 = 1 + (1.4e-45)^2 > 1``, so the points are NOT within
        delta and the cost is 2. After shifting, ``1.4e-45 + 1.0`` rounds to
        ``1.0``, the x-coordinates coincide, the distance is exactly 1 = delta,
        and the cost is 0. Both answers are correct for the input actually
        given; it is the premise that the two inputs describe the same problem
        that is false. Asserting invariance across a lossy transform would be
        asserting something untrue of floating point, so the transform is
        required to be exact first.
        """
        offset = np.array(shift, dtype=np.float64)
        assume(_transform_is_exact(operator.add, ref, offset))
        assume(_transform_is_exact(operator.add, obs, offset))
        base = discrete_edit_distance(ref, obs, delta, operations="both")
        moved = discrete_edit_distance(ref + offset, obs + offset, delta, operations="both")
        assume("numerically_ambiguous" not in (base.status, moved.status))
        assert _cost(base) == _cost(moved)


class TestBackendAndWitnessConsistency:
    @SETTINGS
    @given(ref=curves_1d(), obs=curves_1d(), delta=deltas)
    def test_backends_agree(self, ref, obs, delta):
        for mode in ("delete", "insert", "both"):
            fast = discrete_edit_distance(ref, obs, delta, operations=mode, backend="python")
            slow = discrete_edit_distance(ref, obs, delta, operations=mode, backend="reference")
            assume("numerically_ambiguous" not in (fast.status, slow.status))
            assert fast.status == slow.status
            assert _cost(fast) == _cost(slow), mode

    @SETTINGS
    @given(ref=curves_2d(), obs=curves_2d(), delta=deltas)
    def test_score_only_and_witness_calls_agree(self, ref, obs, delta):
        for mode in ("delete", "insert", "both"):
            plain = discrete_edit_distance(ref, obs, delta, operations=mode)
            witnessed = discrete_edit_distance(
                ref, obs, delta, operations=mode, return_witness=True
            )
            assume("numerically_ambiguous" not in (plain.status, witnessed.status))
            assert plain.status == witnessed.status
            assert _cost(plain) == _cost(witnessed), mode

    @SETTINGS
    @given(ref=curves_2d(), obs=curves_2d(), delta=deltas)
    def test_every_certified_witness_verifies(self, ref, obs, delta):
        for mode in ("delete", "insert", "both"):
            result = discrete_edit_distance(
                ref, obs, delta, operations=mode, return_witness=True
            )
            if result.witness_status != "certified":
                continue
            report = verify_witness(ref, obs, result)
            assert report.ok, (mode, delta, report.violations)
            assert len(result.edits) == result.cost

    @SETTINGS
    @given(ref=curves_2d(), obs=curves_2d(), delta=deltas)
    def test_status_and_witness_status_are_independent(self, ref, obs, delta):
        """An unavailable witness must never be reported as infeasibility."""
        result = discrete_edit_distance(ref, obs, delta, operations="both", return_witness=True)
        assume(result.status != "numerically_ambiguous")
        assert result.status == "optimal"
        assert result.witness_status in ("certified", "unavailable")
        if result.witness_status == "unavailable":
            assert result.cost is not None and not math.isinf(float(result.cost))
            assert result.detail and "delta was not adjusted" in result.detail


class TestNonProperties:
    """These must NOT be assumed. The tests record that they genuinely fail."""

    def test_measure_is_not_symmetric(self):
        ref = np.array([[0.0], [2.0], [4.0]])
        obs = np.array([[0.0]])
        forward = discrete_edit_distance(ref, obs, 1.0, operations="insert")
        backward = discrete_edit_distance(obs, ref, 1.0, operations="insert")
        assert _cost(forward) != _cost(backward)

    def test_cost_is_not_the_injected_corruption_count(self):
        """Repeated samples and delta permit cheaper repairs than the injection."""
        ref = np.array([[0.0], [1.0], [2.0]])
        obs = np.array([[0.0], [1.0], [1.0], [1.0], [2.0]])  # three "duplicated" samples
        result = discrete_edit_distance(ref, obs, 0.1, operations="both")
        assert result.cost == 0  # not 2, even though two vertices were "added"
