"""Tests for the general-dimension exact minimum enclosing ball.

Three independent lines of check, because this kernel decides which insertions
are legal and a wrong answer here is silent:

1. **Against the planar enumeration.** `_geometry._exact_meb_2d` enumerates all
   defining subsets and was tested long before this module existed. For d = 2
   the two must agree exactly, which pits iterative support refinement against
   exhaustive enumeration.
2. **Against exhaustive enumeration in higher dimensions.** For small point sets
   the defining-subset search is tractable in 3-D and 4-D, so the same
   comparison can be made where no prior implementation exists. Note the two
   share the `circumball` primitive; what differs is the search strategy, and
   that is stated rather than glossed.
3. **Against the defining property itself.** Independently of any other
   implementation: the returned ball must enclose every point, and shrinking it
   by any amount must leave some point outside. That is checkable directly and
   assumes nothing about how the ball was found.

Degenerate configurations are not a special section here. They are drawn
constantly by using a coarse integer grid, which makes collinear, cocircular and
duplicate inputs ordinary rather than exotic.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

from frechet_edit._geometry import _exact_meb_2d, _frac_point
from frechet_edit._meb import (
    FracBall,
    circumball,
    encloses,
    exact_meb_nd,
    meb_of_small,
    solve_exact,
    sq_dist,
)


def _grid_points(rng: random.Random, count: int, dim: int, span: int = 4):
    return [
        tuple(Fraction(rng.randint(0, span)) for _ in range(dim))
        for _ in range(count)
    ]


def _brute_meb(points, dim: int) -> FracBall:
    """Exhaustive defining-subset search. Complete, and far too slow for real use."""
    unique = list(dict.fromkeys(points))
    best: FracBall | None = None
    for size in range(1, min(dim + 1, len(unique)) + 1):
        for subset in itertools.combinations(unique, size):
            candidate = circumball(subset)
            if candidate is None or not encloses(candidate, unique):
                continue
            if best is None or candidate.sq_radius < best.sq_radius:
                best = candidate
    assert best is not None
    return best


class TestSolveExact:
    def test_solves_a_known_system(self):
        matrix = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(3)]]
        rhs = [Fraction(5), Fraction(10)]
        solution = solve_exact(matrix, rhs)
        assert solution == [Fraction(1), Fraction(3)]

    def test_reports_singular_rather_than_guessing(self):
        matrix = [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]]
        assert solve_exact(matrix, [Fraction(1), Fraction(2)]) is None

    def test_is_exact_where_floats_would_drift(self):
        matrix = [[Fraction(1), Fraction(1, 3)], [Fraction(1, 7), Fraction(1)]]
        rhs = [Fraction(1), Fraction(1)]
        solution = solve_exact(matrix, rhs)
        assert solution is not None
        lhs0 = matrix[0][0] * solution[0] + matrix[0][1] * solution[1]
        assert lhs0 == rhs[0]  # exact equality, not approx


class TestCircumball:
    def test_two_points_give_the_diameter_ball(self):
        a, b = (Fraction(0), Fraction(0)), (Fraction(4), Fraction(0))
        ball = circumball([a, b])
        assert ball is not None
        assert ball.centre == (Fraction(2), Fraction(0))
        assert ball.sq_radius == Fraction(4)

    def test_right_triangle_circumcentre_is_the_hypotenuse_midpoint(self):
        pts = [(Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)),
               (Fraction(0), Fraction(3))]
        ball = circumball(pts)
        assert ball is not None
        assert ball.centre == (Fraction(2), Fraction(3, 2))

    def test_collinear_points_have_no_circumball(self):
        pts = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
               (Fraction(2), Fraction(0))]
        assert circumball(pts) is None

    def test_regular_tetrahedron_centre_is_equidistant(self):
        pts = [
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        ]
        ball = circumball(pts)
        assert ball is not None
        distances = {sq_dist(p, ball.centre) for p in pts}
        assert len(distances) == 1

    def test_affinely_dependent_set_has_no_determined_circumball(self):
        """Four coplanar points in 3-D determine no unique ball, and say so.

        This is the subtlety behind `meb_of_small` skipping such subsets. It is
        safe: the minimum enclosing ball is always realised by SOME affinely
        independent subset of at most d+1 points, so skipping the dependent ones
        cannot lose the answer. The next test pins that.
        """
        pts = [
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(2), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(2), Fraction(0)),
            (Fraction(2), Fraction(2), Fraction(0)),
        ]
        assert circumball(pts) is None

    def test_the_dependent_set_still_gets_the_right_enclosing_ball(self):
        """Same four coplanar points: the ball is found via an independent subset."""
        pts = [
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(2), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(2), Fraction(0)),
            (Fraction(2), Fraction(2), Fraction(0)),
        ]
        ball = exact_meb_nd(pts, 3)
        assert ball.centre == (Fraction(1), Fraction(1), Fraction(0))
        assert ball.sq_radius == Fraction(2)  # half-diagonal of a 2x2 square
        assert encloses(ball, pts)


class TestAgreesWithThePlanarEnumeration:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_planar_sets_agree_exactly(self, seed):
        rng = random.Random(seed)
        for _ in range(150):
            pts = _grid_points(rng, rng.randint(1, 9), 2)
            assert exact_meb_nd(pts, 2) == _exact_meb_2d(pts)

    def test_agrees_on_a_cocircular_set(self):
        pts = [(Fraction(1), Fraction(0)), (Fraction(-1), Fraction(0)),
               (Fraction(0), Fraction(1)), (Fraction(0), Fraction(-1))]
        assert exact_meb_nd(pts, 2) == _exact_meb_2d(pts)

    def test_agrees_when_every_point_is_identical(self):
        pts = [(Fraction(3), Fraction(4))] * 6
        assert exact_meb_nd(pts, 2) == _exact_meb_2d(pts)


class TestAgreesWithExhaustiveSearchInHigherDimensions:
    @pytest.mark.parametrize("dim", [3, 4])
    def test_random_sets_agree(self, dim):
        rng = random.Random(20 + dim)
        for _ in range(60):
            pts = _grid_points(rng, rng.randint(1, 6), dim, span=3)
            assert exact_meb_nd(pts, dim) == _brute_meb(pts, dim)


class TestTheDefiningProperty:
    """Checks that assume nothing about how the ball was computed."""

    @pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
    def test_ball_encloses_every_point(self, dim):
        rng = random.Random(700 + dim)
        for _ in range(40):
            pts = _grid_points(rng, rng.randint(1, 7), dim, span=3)
            assert encloses(exact_meb_nd(pts, dim), pts)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_no_smaller_ball_at_the_same_centre_encloses(self, dim):
        rng = random.Random(800 + dim)
        for _ in range(40):
            pts = _grid_points(rng, rng.randint(2, 6), dim, span=3)
            ball = exact_meb_nd(pts, dim)
            if ball.sq_radius == 0:
                continue
            shrunk = FracBall(ball.centre, ball.sq_radius * Fraction(99, 100))
            assert not encloses(shrunk, pts), "radius was not tight"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_at_least_two_points_sit_on_the_boundary(self, dim):
        rng = random.Random(900 + dim)
        for _ in range(40):
            pts = _grid_points(rng, rng.randint(2, 6), dim, span=3)
            if len(set(pts)) < 2:
                continue
            ball = exact_meb_nd(pts, dim)
            on_boundary = [p for p in pts if sq_dist(p, ball.centre) == ball.sq_radius]
            assert len(on_boundary) >= 2

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_translation_moves_the_centre_and_keeps_the_radius(self, dim):
        rng = random.Random(1000 + dim)
        shift = tuple(Fraction(rng.randint(-5, 5)) for _ in range(dim))
        pts = _grid_points(rng, 5, dim, span=3)
        before = exact_meb_nd(pts, dim)
        moved = [tuple(c + s for c, s in zip(p, shift, strict=True)) for p in pts]
        after = exact_meb_nd(moved, dim)
        assert after.sq_radius == before.sq_radius
        assert after.centre == tuple(
            c + s for c, s in zip(before.centre, shift, strict=True)
        )

    @pytest.mark.parametrize("dim", [2, 3])
    def test_scaling_scales_the_squared_radius_quadratically(self, dim):
        rng = random.Random(1100 + dim)
        pts = _grid_points(rng, 5, dim, span=3)
        before = exact_meb_nd(pts, dim)
        scaled = [tuple(c * 3 for c in p) for p in pts]
        assert exact_meb_nd(scaled, dim).sq_radius == before.sq_radius * 9


class TestSeedingAndEdgeCases:
    def test_a_wrong_seed_still_gives_the_right_ball(self):
        """The float pass only proposes; it must never be able to be believed."""
        rng = random.Random(5)
        for _ in range(40):
            pts = _grid_points(rng, 6, 3, span=3)
            honest = exact_meb_nd(pts, 3)
            misled = exact_meb_nd(pts, 3, seed_indices=[0])
            assert misled == honest

    def test_single_point_gives_a_zero_radius_ball(self):
        ball = exact_meb_nd([(Fraction(2), Fraction(5))], 2)
        assert ball.sq_radius == 0
        assert ball.centre == (Fraction(2), Fraction(5))

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match="at least one point"):
            exact_meb_nd([], 2)

    def test_meb_of_small_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one point"):
            meb_of_small([], 2)

    def test_round_cap_is_reported_not_hidden(self):
        rng = random.Random(7)
        pts = _grid_points(rng, 12, 3, span=6)
        with pytest.raises(RuntimeError, match="did not converge"):
            exact_meb_nd(pts, 3, max_rounds=1)


class TestMatchesFloatReality:
    """A sanity bridge to numpy, so the exact answer is not exactly wrong."""

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_radius_is_close_to_a_float_estimate(self, dim):
        rng = np.random.default_rng(dim)
        raw = rng.normal(size=(8, dim))
        pts = [_frac_point(p) for p in raw]
        exact = float(exact_meb_nd(pts, dim).sq_radius) ** 0.5
        centroid = raw.mean(axis=0)
        spread = float(np.linalg.norm(raw - centroid, axis=1).max())
        half_diameter = max(
            float(np.linalg.norm(a - b)) for a in raw for b in raw
        ) / 2.0
        # the MEB radius lies between the half-diameter and the centroid spread
        assert half_diameter - 1e-9 <= exact <= spread + 1e-9
