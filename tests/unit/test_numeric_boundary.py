"""The paths where a wrong answer would be SILENT.

Most of this package fails loudly when it fails. These are the exceptions: the
exact-arithmetic fallbacks, the abstention decisions and the witness-centre
search. A bug in any of them returns a plausible number rather than an error,
which is why they get their own file and why the fixtures here are pinned
constants rather than random draws.

Two of the tests inject a fault on purpose, by making the float64 pre-pass
propose a deliberately bad support set. That path is close to unreachable with
the real pre-pass, which is exactly why it needs forcing: an untested safety net
is not a safety net. Each is marked clearly where it happens.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from frechet_edit import (
    EditResult,
    NumericallyAmbiguous,
    discrete_edit_distance,
)
from frechet_edit import _geometry as geometry
from frechet_edit._geometry import (
    certified_block_centre,
    exact_meb,
    meb_radius_le,
)
from frechet_edit._meb import circumball
from frechet_edit._minqueue import MinQueue
from frechet_edit._numerics import (
    EXACT_BALL_MAX_POINTS,
    PredicateStats,
    frac_sq_dist,
    within,
)

# Found by search over random blocks: the exactly-rounded minimum-enclosing-ball
# centre is NOT within delta of every vertex here, but a neighbouring float64
# point one ulp away is. This is the only configuration class that reaches the
# neighbour search in `certified_block_centre`.
ULP_BLOCK = np.array([[5.08, -4.304], [4.677, -0.844], [4.459, -1.141]])
ULP_DELTA = 1.7416952230513814


class TestFastPolicySkipsTheExactPath:
    """`policy="fast"` must decide in float64 and never reach exact arithmetic."""

    def test_ball_predicate_decides_without_exact_fallback(self):
        stats = PredicateStats()
        block = np.array([[0.0, 0.0], [6.0, 0.0]])  # radius exactly 3
        assert meb_radius_le(block, 3.0, policy="fast", stats=stats) is True
        assert stats.exact_fallbacks == 0

    def test_distance_predicate_decides_without_exact_fallback(self):
        stats = PredicateStats()
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])  # distance exactly 5
        assert within(a, b, 5.0, policy="fast", stats=stats) is True
        assert stats.exact_fallbacks == 0

    def test_fast_and_certified_agree_away_from_the_boundary(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            block = np.round(rng.normal(size=(4, 2)) * 2, 2)
            # 0.37 is not near any exact radius of these blocks, so both
            # policies should be decided by the float test alone.
            assert meb_radius_le(block, 0.37, policy="fast") == meb_radius_le(
                block, 0.37, policy="certified"
            )


class TestCertifiedBlockCentre:
    def test_the_rounded_centre_is_used_when_it_already_works(self):
        block = np.array([[0.0, 0.0], [4.0, 0.0]])
        centre = certified_block_centre(block, 2.0)
        assert centre is not None
        assert np.allclose(centre, [2.0, 0.0])

    def test_the_neighbour_search_rescues_a_centre_rounding_spoiled(self):
        """The pinned ulp fixture: exact centre rounds badly, a neighbour works."""
        delta2 = Fraction(ULP_DELTA) ** 2
        exact_centre = np.array(
            [float(c) for c in exact_meb(ULP_BLOCK).centre], dtype=np.float64
        )
        # Precondition: the naive rounded centre really does fail.
        assert not all(frac_sq_dist(exact_centre, p) <= delta2 for p in ULP_BLOCK)

        centre = certified_block_centre(ULP_BLOCK, ULP_DELTA)
        assert centre is not None
        # Whatever it returned must be genuinely within delta of every vertex,
        # checked in exact arithmetic rather than in floats.
        assert all(frac_sq_dist(centre, p) <= delta2 for p in ULP_BLOCK)

    def test_returns_none_rather_than_a_wrong_centre_when_infeasible(self):
        block = np.array([[0.0, 0.0], [10.0, 0.0]])  # needs radius 5
        assert certified_block_centre(block, 1.0) is None

    def test_every_returned_centre_is_exactly_certified(self):
        """The contract, swept: a returned point is never merely close."""
        rng = np.random.default_rng(11)
        returned = 0
        for _ in range(300):
            block = np.round(rng.normal(size=(int(rng.integers(2, 5)), 2)) * 2, 2)
            radius = math.sqrt(float(exact_meb(block).sq_radius))
            delta = radius * 1.000001
            centre = certified_block_centre(block, delta)
            if centre is None:
                continue
            returned += 1
            delta2 = Fraction(float(delta)) ** 2
            assert all(frac_sq_dist(centre, p) <= delta2 for p in block)
        assert returned > 0, "the sweep never produced a centre to check"


class TestAbstentionIsReportedNotGuessed:
    def test_exact_ball_abstains_past_the_point_cap(self):
        with pytest.raises(NumericallyAmbiguous, match="capped"):
            exact_meb(np.zeros((EXACT_BALL_MAX_POINTS + 1, 2)))

    def test_boundary_block_past_the_cap_abstains(self, monkeypatch):
        """FAULT INJECTED: the float pre-pass is forced to propose a bad support.

        With the real pre-pass this path is close to unreachable, because the
        proposed support is almost always good enough to settle the decision
        before the cap matters. Forcing a single-point support drives a genuinely
        undecided large block into the capped exact branch, which must abstain
        rather than answer.
        """
        monkeypatch.setattr(geometry, "_support_indices", lambda pts, rng: [0])

        # A large block whose true enclosing radius is exactly delta, so no float
        # certificate can settle it either way.
        block = np.zeros((EXACT_BALL_MAX_POINTS + 2, 2))
        block[1] = [6.0, 0.0]
        stats = PredicateStats()
        with pytest.raises(NumericallyAmbiguous, match="exact cap"):
            meb_radius_le(block, 3.0, stats=stats)
        assert stats.ambiguous == 1

    def test_the_public_api_reports_ambiguity_as_a_status(self, monkeypatch):
        """FAULT INJECTED, as above. The API must not turn abstention into a cost."""
        def explode(*args, **kwargs):
            raise NumericallyAmbiguous("forced for test")

        monkeypatch.setattr(geometry, "meb_radius_le", explode)
        result = discrete_edit_distance(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            0.4,
            operations="insert",
        )
        assert isinstance(result, EditResult)
        assert result.status == "numerically_ambiguous"
        assert result.cost is None

    def test_an_abstaining_block_never_silently_becomes_infeasible(self, monkeypatch):
        """Abstention and infeasibility are different answers and must stay so."""
        def explode(*args, **kwargs):
            raise NumericallyAmbiguous("forced for test")

        monkeypatch.setattr(geometry, "meb_radius_le", explode)
        result = discrete_edit_distance(
            np.array([[0.0, 0.0], [50.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            0.1,
            operations="insert",
        )
        assert result.status != "infeasible"
        assert result.status == "numerically_ambiguous"


class TestClosedComparisonsAtTheExactBoundary:
    """`<= delta` is closed, and one ulp either side must flip the answer."""

    def test_ball_predicate_is_closed(self):
        block = np.array([[0.0, 0.0], [6.0, 0.0]])  # radius exactly 3
        assert meb_radius_le(block, 3.0) is True
        assert meb_radius_le(block, float(np.nextafter(3.0, 0.0))) is False

    def test_distance_predicate_is_closed(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])  # distance exactly 5
        assert within(a, b, 5.0) is True
        assert within(a, b, float(np.nextafter(5.0, 0.0))) is False

    def test_edit_distance_is_closed_at_delta(self):
        ref = np.array([[0.0], [1.0]])
        obs = np.array([[0.0], [1.0]])
        assert discrete_edit_distance(ref, obs, 0.0 + 1e-300).status in {
            "optimal",
            "infeasible",
        }
        exact = discrete_edit_distance(ref, obs, 1.0, operations="delete")
        assert exact.status == "optimal"


class TestSmallSurfacesThatStillMustWork:
    """Accessors and reprs. Cheap to test, and confusing when they break."""

    def test_edit_result_repr_mentions_status_and_cost(self):
        result = discrete_edit_distance(
            np.array([[0.0]]), np.array([[0.0]]), 1.0, operations="delete"
        )
        text = repr(result)
        assert "EditResult(" in text
        assert "status='optimal'" in text
        assert "cost=0" in text

    def test_edit_result_repr_renders_infinite_cost_as_inf(self):
        result = discrete_edit_distance(
            np.array([[0.0], [9.0]]), np.array([[0.0]]), 0.1, operations="delete"
        )
        assert result.status == "infeasible"
        assert "cost=inf" in repr(result)

    def test_min_queue_length_and_emptiness_track_pushes(self):
        queue = MinQueue()
        assert len(queue) == 0
        assert queue.empty is True
        assert queue.min_value() == math.inf
        assert queue.argmin() is None

        queue.push(0, 5.0)
        assert len(queue) == 1
        assert queue.empty is False
        queue.push(1, 2.0)
        # 5.0 is dominated by the later, smaller 2.0 and is dropped on push.
        assert queue.min_value() == 2.0
        assert queue.argmin() == 1

        queue.expire_before(2)
        assert len(queue) == 0
        assert queue.empty is True

    def test_min_queue_rejects_non_ascending_indices(self):
        queue = MinQueue()
        queue.push(3, 1.0)
        with pytest.raises(ValueError, match="strictly ascend"):
            queue.push(3, 0.5)

    def test_circumball_of_nothing_is_undetermined(self):
        assert circumball([]) is None


class TestALossyTranslationIsNotTheSameProblem:
    """Hypothesis found this while probing translation invariance.

    It is NOT a bug, and the point of pinning it is to stop anyone "fixing" it.
    Both answers below are correct for the input actually supplied; what is
    false is the assumption that a float64 translation carries a geometry across
    unchanged. `tests/property/test_invariants.py` now requires the transform to
    be exact before asserting invariance over it.
    """

    # The smallest float32 subnormal. Adding 1.0 to it rounds straight back to
    # 1.0, so the x-coordinate is annihilated by the shift.
    TINY = 1.40129846e-45

    def test_before_the_shift_the_points_are_outside_delta(self):
        ref = np.array([[0.0, 0.0]])
        obs = np.array([[self.TINY, 1.0]])
        # Exactly: dist^2 = 1 + TINY^2, which is strictly greater than 1.
        assert Fraction(self.TINY) ** 2 + 1 > 1
        result = discrete_edit_distance(ref, obs, 1.0, operations="both")
        assert result.cost == 2, "exact arithmetic must see past the float tie"

    def test_after_the_shift_they_are_exactly_on_delta(self):
        offset = np.array([1.0, 0.0])
        ref = np.array([[0.0, 0.0]]) + offset
        obs = np.array([[self.TINY, 1.0]]) + offset
        # The shift absorbed TINY, so the x-coordinates now coincide exactly.
        assert obs[0, 0] == ref[0, 0] == 1.0
        result = discrete_edit_distance(ref, obs, 1.0, operations="both")
        assert result.cost == 0, "distance is exactly delta, and <= is closed"

    def test_the_float_tie_is_what_makes_this_subtle(self):
        """In float64 both configurations look identical. Only exact arithmetic
        separates them, which is the whole reason the predicate policy exists."""
        assert self.TINY**2 + 1.0 == 1.0
        assert self.TINY + 1.0 == 1.0
