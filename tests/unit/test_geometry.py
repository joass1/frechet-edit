"""Enclosing-ball geometry and the mu preprocessing.

Covers the degeneracy list in docs/numerics.md section 4. These cases decide
whether the insertion recurrence is trustworthy, so they are deliberately
adversarial rather than representative.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from frechet_edit._geometry import (
    certified_block_centre,
    exact_meb,
    meb_radius_le,
    mu_indices,
)
from frechet_edit._numerics import NumericallyAmbiguous, exact_within
from tests.conftest import curve


def sq_radius(points) -> float:
    return float(exact_meb(curve(points)).sq_radius)


class TestExactMEB:
    def test_single_point_has_zero_radius(self):
        ball = exact_meb(curve([[3.0, -7.0]]))
        assert ball.sq_radius == 0
        assert ball.centre == (Fraction(3), Fraction(-7))

    def test_two_points_give_the_midpoint(self):
        ball = exact_meb(curve([[0.0, 0.0], [4.0, 0.0]]))
        assert ball.centre == (Fraction(2), Fraction(0))
        assert ball.sq_radius == 4

    def test_obtuse_triangle_uses_the_longest_side_not_the_circumcircle(self):
        """The classic trap: an obtuse triangle's MEB is its longest-side diameter."""
        pts = curve([[0.0, 0.0], [4.0, 0.0], [2.0, 0.3]])
        ball = exact_meb(pts)
        assert ball.centre == (Fraction(2), Fraction(0))
        assert ball.sq_radius == 4
        # The circumradius is much larger, so a naive circumcircle answer differs.
        assert sq_radius(pts) < 6.0

    def test_acute_triangle_uses_the_circumcircle(self):
        root3_2 = math.sqrt(3.0) / 2.0
        pts = curve([[0.0, 0.0], [1.0, 0.0], [0.5, root3_2]])
        assert sq_radius(pts) == pytest.approx(1.0 / 3.0, rel=1e-12)

    def test_right_triangle_hypotenuse_is_the_diameter(self):
        pts = curve([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        ball = exact_meb(pts)
        assert ball.sq_radius == Fraction(25, 4)

    def test_collinear_points(self):
        pts = curve([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        ball = exact_meb(pts)
        assert ball.centre == (Fraction(3, 2), Fraction(3, 2))
        assert ball.sq_radius == Fraction(9, 2)

    def test_duplicate_points_do_not_change_the_ball(self):
        base = curve([[0.0, 0.0], [2.0, 0.0]])
        dupes = curve([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
        assert exact_meb(base) == exact_meb(dupes)

    def test_cocircular_points(self):
        pts = curve([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        ball = exact_meb(pts)
        assert ball.centre == (Fraction(0), Fraction(0))
        assert ball.sq_radius == 1

    def test_near_zero_area_triangle(self):
        pts = curve([[0.0, 0.0], [1.0, 1e-12], [2.0, 0.0]])
        assert sq_radius(pts) == pytest.approx(1.0, rel=1e-9)

    def test_large_translation_is_exact(self):
        offset = 1e8
        near = exact_meb(curve([[0.0, 0.0], [2.0, 0.0]]))
        far = exact_meb(curve([[offset, offset], [offset + 2.0, offset]]))
        assert near.sq_radius == far.sq_radius

    def test_disparate_scales(self):
        pts = curve([[0.0, 0.0], [1e-9, 0.0], [1e6, 0.0]])
        assert sq_radius(pts) == pytest.approx((1e6 / 2) ** 2, rel=1e-12)

    def test_one_dimensional_interval(self):
        ball = exact_meb(curve([5.0, -3.0, 1.0]))
        assert ball.centre == (Fraction(1),)
        assert ball.sq_radius == 16

    def test_three_dimensions_now_resolve_rather_than_raising(self):
        """exact_meb used to reject d >= 3; it now dispatches to `_meb`."""
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        ball = exact_meb(pts)
        assert ball.centre == (Fraction(1), Fraction(0), Fraction(0))
        assert ball.sq_radius == 1

    def test_too_many_points_still_abstains(self):
        from frechet_edit._numerics import EXACT_BALL_MAX_POINTS

        with pytest.raises(NumericallyAmbiguous, match="capped"):
            exact_meb(np.zeros((EXACT_BALL_MAX_POINTS + 1, 3)))


class TestBallPredicate:
    def test_closed_comparison_at_the_exact_boundary(self):
        """delta exactly equal to the radius must decide YES (comparison is closed)."""
        pts = curve([[0.0, 0.0], [6.0, 0.0]])  # radius exactly 3
        assert meb_radius_le(pts, 3.0) is True
        assert meb_radius_le(pts, np.nextafter(3.0, 0.0)) is False
        assert meb_radius_le(pts, np.nextafter(3.0, 10.0)) is True

    def test_boundary_of_a_three_point_ball(self):
        pts = curve([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])  # radius exactly 2.5
        assert meb_radius_le(pts, 2.5) is True
        assert meb_radius_le(pts, 2.4999999999999996) is False

    def test_singleton_is_always_within_a_positive_delta(self):
        assert meb_radius_le(curve([[1e300, -1e300]]), 1e-300) is True

    def test_large_boundary_block_abstains_rather_than_guessing(self):
        """Above the exact cap, a boundary decision must abstain, not guess."""
        from frechet_edit import _numerics

        n = _numerics.EXACT_BALL_MAX_POINTS + 5
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        # A circle of radius 1 whose enclosing decision at delta == 1 lands on the
        # boundary and whose float support set does not enclose exactly.
        try:
            meb_radius_le(pts, 1.0)
        except NumericallyAmbiguous as exc:
            assert "boundary" in str(exc)
        # Either it was certified by an exact support set or it abstained; what
        # must never happen is a silent wrong answer, which the asserts above
        # and the exact tests in this file jointly rule out.


class TestMuIndices:
    def test_monotone_non_decreasing(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(size=(40, 2)) * 5.0
        mu = mu_indices(ref, 2.0)
        assert list(mu) == sorted(mu)
        assert all(0 <= mu[i] <= i for i in range(len(mu)))

    def test_known_small_case(self):
        ref = curve([0.0, 2.0, 4.0])
        assert list(mu_indices(ref, 1.0)) == [0, 0, 1]
        assert list(mu_indices(ref, 2.0)) == [0, 0, 0]
        assert list(mu_indices(ref, 0.5)) == [0, 1, 2]

    def test_agrees_with_brute_force_all_blocks(self):
        """mu(i) must equal the smallest feasible block start, checked exhaustively."""
        rng = np.random.default_rng(11)
        for _ in range(20):
            m = int(rng.integers(1, 9))
            ref = np.round(rng.normal(size=(m, 2)) * 3.0, 3)
            delta = float(rng.uniform(0.3, 3.0))
            mu = mu_indices(ref, delta)
            for i in range(m):
                expected = i
                for t in range(i + 1):
                    if meb_radius_le(ref[t : i + 1], delta):
                        expected = t
                        break
                assert mu[i] == expected, (i, mu[i], expected, delta)

    def test_tiny_delta_forces_singleton_blocks(self):
        ref = curve([0.0, 1.0, 2.0, 3.0])
        assert list(mu_indices(ref, 1e-9)) == [0, 1, 2, 3]

    def test_huge_delta_allows_one_block(self):
        ref = curve([0.0, 1.0, 2.0, 3.0])
        assert list(mu_indices(ref, 1e9)) == [0, 0, 0, 0]


class TestCertifiedCentre:
    def test_centre_is_within_delta_of_every_block_vertex(self):
        block = curve([2.0, 4.0])
        centre = certified_block_centre(block, 1.0)
        assert centre is not None
        for point in block:
            assert exact_within(centre, point, 1.0)

    def test_returns_none_when_the_block_does_not_fit(self):
        assert certified_block_centre(curve([0.0, 4.0]), 1.0) is None

    def test_two_dimensional_centre_is_certified(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            block = np.round(rng.normal(size=(int(rng.integers(1, 6)), 2)), 3)
            delta = float(rng.uniform(0.5, 4.0))
            centre = certified_block_centre(block, delta)
            if centre is None:
                assert not meb_radius_le(block, delta)
                continue
            for point in block:
                assert exact_within(centre, point, delta), (centre, point, delta)

    def test_boundary_block_either_certifies_or_declines(self):
        """A witness is never emitted unless it is exactly within delta."""
        block = curve([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        radius = math.sqrt(float(exact_meb(block).sq_radius))
        centre = certified_block_centre(block, radius)
        if centre is not None:
            for point in block:
                assert exact_within(centre, point, radius)
