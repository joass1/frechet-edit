"""Self-checks for the independent oracles.

The oracles are code too, and everything they certify rests on them being
right.  These tests check the oracles against hand-computed values and against
each other's slowest, most literal formulation - never against the production
package, which this file must not import.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

import pytest

from .block_edit_search import (
    OracleBudgetExceeded,
    candidate_points,
    edit_oracle,
    replay,
)
from .enclosing_circle import exact_meb, meb_radius_le
from .exact import sq_dist, to_frac_point, within
from .frechet import (
    brute_force_sq,
    discrete_frechet,
    discrete_frechet_le,
    enumerate_couplings,
)
from .subsequence import deletion_oracle

SEED = 20240912


def _random_curve(rng: random.Random, length: int, dim: int, spread: int = 3):
    return [[rng.randrange(spread) for _ in range(dim)] for _ in range(length)]


def _all_curves_1d(max_len: int, alphabet):
    for length in range(1, max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield [[value] for value in combo]


# ---------------------------------------------------------------------------
# exact.py
# ---------------------------------------------------------------------------


def test_float_coordinates_convert_without_rounding():
    # 0.1 is not 1/10 in binary; the oracle must see the real binary value.
    (frac,) = to_frac_point([0.1])
    assert frac == Fraction(3602879701896397, 36028797018963968)
    assert frac != Fraction(1, 10)


def test_within_is_closed_and_has_no_hidden_epsilon():
    assert within([0.0], [1.0], 1.0) is True  # closed: <= delta
    # 1 + 2 * 2**-53 is the next float above 1.0 that is strictly greater;
    # no epsilon may enlarge delta to admit it.
    assert within([0.0], [1.0 + 2e-16], 1.0) is False
    assert sq_dist([0.0, 0.0], [3.0, 4.0]) == Fraction(25)


def test_sq_dist_rejects_dimension_mismatch_and_non_finite():
    with pytest.raises(ValueError):
        sq_dist([0.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        sq_dist([float("nan")], [0.0])
    with pytest.raises(ValueError):
        within([0.0], [1.0], 0.0)  # delta must be strictly positive


# ---------------------------------------------------------------------------
# frechet.py - the DP against literal coupling enumeration
# ---------------------------------------------------------------------------


def test_enumerate_couplings_produces_only_valid_couplings():
    for a in range(1, 5):
        for b in range(1, 5):
            seen = 0
            for coupling in enumerate_couplings(a, b):
                seen += 1
                assert coupling[0] == (0, 0)
                assert coupling[-1] == (a - 1, b - 1)
                for (i0, j0), (i1, j1) in itertools.pairwise(coupling):
                    assert (i1 - i0, j1 - j0) in ((1, 0), (0, 1), (1, 1))
            assert seen > 0
    # Delannoy numbers D(a-1, b-1) count monotone lattice paths with the three
    # legal steps, so they count couplings exactly.
    assert sum(1 for _ in enumerate_couplings(1, 1)) == 1
    assert sum(1 for _ in enumerate_couplings(2, 2)) == 3
    assert sum(1 for _ in enumerate_couplings(3, 3)) == 13
    assert sum(1 for _ in enumerate_couplings(4, 4)) == 63


def test_dp_agrees_with_brute_force_exhaustively_for_sizes_up_to_3():
    curves = list(_all_curves_1d(3, (0, 1, 2)))
    assert len(curves) == 39
    for A in curves:
        for B in curves:
            assert discrete_frechet(A, B) == brute_force_sq(A, B)


@pytest.mark.parametrize("dim", [1, 2])
def test_dp_agrees_with_brute_force_random_sizes_up_to_4(dim):
    rng = random.Random(SEED + dim)
    for a in range(1, 5):
        for b in range(1, 5):
            for _ in range(20):
                A = _random_curve(rng, a, dim)
                B = _random_curve(rng, b, dim)
                assert discrete_frechet(A, B) == brute_force_sq(A, B)


def test_decision_agrees_with_brute_force_value():
    rng = random.Random(SEED + 7)
    for a in range(1, 5):
        for b in range(1, 5):
            for _ in range(10):
                A = _random_curve(rng, a, 2)
                B = _random_curve(rng, b, 2)
                exact = brute_force_sq(A, B)
                for delta in (0.5, 1.0, 1.5, 2.0, 3.0):
                    expected = exact <= Fraction(delta) ** 2
                    assert discrete_frechet_le(A, B, delta) is expected


def test_hand_computed_frechet_values():
    A = [[0], [1], [2]]
    B = [[0], [1], [100], [2]]
    # B[2] = 100 must be coupled to some A[i]; the cheapest is A[2] = 2, and
    # the coupling (0,0),(1,1),(2,2),(2,3) attains it, so the value is 98.
    assert discrete_frechet(A, B) == Fraction(98) ** 2
    assert brute_force_sq(A, B) == Fraction(9604)
    assert discrete_frechet_le(A, B, 98.0) is True
    assert discrete_frechet_le(A, B, 97.999) is False

    assert discrete_frechet([[0], [1]], [[0], [1]]) == 0
    assert discrete_frechet([[0.0, 0.0]], [[3.0, 4.0]]) == Fraction(25)
    # repeated vertices are never deduplicated
    assert discrete_frechet([[0], [0], [0]], [[0]]) == 0


def test_empty_curve_is_infeasible_not_zero():
    assert discrete_frechet_le([[0]], [], 1.0) is False
    assert discrete_frechet_le([], [[0]], 1.0) is False
    assert discrete_frechet([[0]], []) is None
    with pytest.raises(ValueError):
        discrete_frechet_le([], [], 1.0)


# ---------------------------------------------------------------------------
# enclosing_circle.py - Oracle C
# ---------------------------------------------------------------------------


def test_meb_single_point_has_zero_radius():
    centre, sq_radius = exact_meb([[3.5]])
    assert centre == (Fraction(7, 2),)
    assert sq_radius == 0
    centre, sq_radius = exact_meb([[1.0, 2.0]])
    assert centre == (Fraction(1), Fraction(2))
    assert sq_radius == 0


def test_meb_two_points_is_the_midpoint_and_half_the_distance():
    centre, sq_radius = exact_meb([[0.0], [10.0]])
    assert centre == (Fraction(5),)
    assert sq_radius == Fraction(25)
    centre, sq_radius = exact_meb([[0.0, 0.0], [3.0, 4.0]])
    assert centre == (Fraction(3, 2), Fraction(2))
    assert sq_radius == Fraction(25, 4)  # (5/2)**2


def test_meb_acute_triangle_is_the_circumcircle():
    # (0,0), (4,0), (2,4): acute, so the MEB is the circumcircle.
    # Circumcentre (2, 3/2) by the perpendicular bisectors; r^2 = 4 + 9/4.
    centre, sq_radius = exact_meb([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
    assert centre == (Fraction(2), Fraction(3, 2))
    assert sq_radius == Fraction(25, 4)
    # It is strictly larger than every half-diameter, which proves no 2-point
    # diameter candidate encloses the set: this really is the circumcircle.
    pts = [[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]]
    widest = max(sq_dist(p, q) for p, q in itertools.combinations(pts, 2))
    assert sq_radius > widest / 4


def test_meb_equilateral_triangle_matches_the_known_circumradius():
    # An exactly equilateral triangle has an irrational coordinate, so it is
    # not representable in float64 or in the rationals; this checks the float
    # approximation lands on the known circumcentre and circumradius.
    side = 1.0
    pts = [[0.0, 0.0], [side, 0.0], [side / 2, math.sqrt(3) / 2]]
    centre, sq_radius = exact_meb(pts)
    assert float(centre[0]) == pytest.approx(0.5, abs=1e-12)
    assert float(centre[1]) == pytest.approx(math.sqrt(3) / 6, abs=1e-12)
    assert float(sq_radius) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_meb_obtuse_triangle_is_the_longest_side_diameter_not_the_circumcircle():
    # (0,0), (10,0), (5,1) is obtuse.  Its circumcentre is (5,-12) with
    # r^2 = 169; the MEB is the diameter of the longest side, centre (5,0),
    # r^2 = 25.  Implementations that just take the circumcircle fail here.
    pts = [[0.0, 0.0], [10.0, 0.0], [5.0, 1.0]]
    centre, sq_radius = exact_meb(pts)
    assert centre == (Fraction(5), Fraction(0))
    assert sq_radius == Fraction(25)
    assert sq_radius < Fraction(169)


def test_meb_collinear_triple_uses_the_extreme_pair():
    centre, sq_radius = exact_meb([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert centre == (Fraction(1), Fraction(1))
    assert sq_radius == Fraction(2)
    centre, sq_radius = exact_meb([[0.0], [1.0], [2.0]])
    assert centre == (Fraction(1),)
    assert sq_radius == Fraction(1)


def test_meb_handles_duplicate_points():
    centre, sq_radius = exact_meb([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    assert centre == (Fraction(1), Fraction(0))
    assert sq_radius == Fraction(1)
    centre, sq_radius = exact_meb([[1.0, 1.0]] * 5)
    assert centre == (Fraction(1), Fraction(1))
    assert sq_radius == 0


def test_meb_encloses_everything_and_respects_the_diameter_and_jung_bounds():
    rng = random.Random(SEED + 11)
    for _ in range(150):
        size = rng.randint(1, 8)
        pts = [[float(rng.randrange(-6, 7)), float(rng.randrange(-6, 7))] for _ in range(size)]
        centre, sq_radius = exact_meb(pts)
        assert all(sq_dist(centre, p) <= sq_radius for p in pts)
        diameter_sq = max(
            (sq_dist(p, q) for p, q in itertools.combinations(pts, 2)), default=Fraction(0)
        )
        # r >= d/2 (the two farthest points must both fit)
        assert 4 * sq_radius >= diameter_sq
        # Jung's theorem in the plane: r <= d/sqrt(3)
        assert 3 * sq_radius <= diameter_sq


def test_meb_input_guards():
    with pytest.raises(ValueError):
        exact_meb([])
    with pytest.raises(ValueError):
        exact_meb([[0.0]] * 13)  # |S| <= 12
    with pytest.raises(ValueError):
        exact_meb([[0.0, 0.0, 0.0]])  # d in (1, 2) only
    with pytest.raises(ValueError):
        exact_meb([[0.0], [0.0, 1.0]])  # ragged


def test_meb_radius_le_is_closed():
    pts = [[0.0], [2.0]]
    assert meb_radius_le(pts, 1.0) is True  # radius exactly 1
    assert meb_radius_le(pts, 0.999) is False


# ---------------------------------------------------------------------------
# subsequence.py - Oracle A
# ---------------------------------------------------------------------------


def test_deletion_oracle_removes_the_single_outlier():
    R = [[0], [1], [2]]
    Q = [[0], [1], [100], [2]]
    assert deletion_oracle(R, Q, 0.1) == 1


def test_deletion_oracle_returns_zero_when_already_close():
    R = [[0], [1], [2]]
    Q = [[0], [1], [2]]
    assert deletion_oracle(R, Q, 0.1) == 0


def test_deletion_oracle_is_infeasible_when_no_subsequence_covers_the_reference():
    # Q can never reach R[1] = 10 by deleting vertices, so deletion-only is
    # mathematically INFEASIBLE and that is a value, not an exception.
    R = [[0], [10]]
    Q = [[0]]
    assert deletion_oracle(R, Q, 1.0) == math.inf
    # Deleting cannot help when an unreachable vertex is in R, not in Q.
    R2 = [[0], [1], [2]]
    Q2 = [[0], [9], [2]]
    assert deletion_oracle(R2, Q2, 0.5) == math.inf


def test_deletion_oracle_reports_stats_and_guards_its_size():
    stats: dict = {}
    assert deletion_oracle([[0], [1]], [[0], [5], [1]], 0.5, stats=stats) == 1
    assert stats["subsequences_total"] == 7
    assert stats["subsequences_evaluated"] >= 1
    with pytest.raises(ValueError):
        deletion_oracle([[0]], [[0]] * 13, 1.0)
    with pytest.raises(ValueError):
        deletion_oracle([], [[0]], 1.0)


# ---------------------------------------------------------------------------
# block_edit_search.py - Oracle B
# ---------------------------------------------------------------------------


def test_replay_follows_the_witness_invariants():
    Q = [[0.0], [1.0], [2.0]]
    out = replay(Q, frozenset(), {0: ((Fraction(-1),),), 3: ((Fraction(9),),)})
    assert out == [(Fraction(-1),), [0.0], [1.0], [2.0], (Fraction(9),)]
    # deletions never renumber later indices
    out = replay(Q, frozenset({1}), {})
    assert out == [[0.0], [2.0]]
    # delete-all-and-rebuild: gap order survives, within a gap order survives
    out = replay(Q, frozenset({0, 1, 2}), {1: ((Fraction(5),), (Fraction(6),))})
    assert out == [(Fraction(5),), (Fraction(6),)]
    assert replay(Q, frozenset({0, 1, 2}), {}) == []


def test_candidate_set_contains_non_reference_points():
    R = [[0.0], [2.0], [4.0]]
    candidates = candidate_points(R, 1.0)
    assert candidates == (
        (Fraction(0),),
        (Fraction(1),),
        (Fraction(2),),
        (Fraction(3),),
        (Fraction(4),),
    )
    reference_vertices = {(Fraction(0),), (Fraction(2),), (Fraction(4),)}
    extra = [c for c in candidates if c not in reference_vertices]
    assert extra, "the candidate set must contain block centres that are not R vertices"
    assert (Fraction(3),) in extra  # centre of the block R[1..2]


def test_insert_oracle_uses_a_block_centre_instead_of_two_reference_vertices():
    R = [[0.0], [2.0], [4.0]]
    Q = [[0.0]]
    stats: dict = {}
    # A single inserted point at 3 covers both 2 and 4 (each exactly 1 away),
    # so one insertion suffices; inserting reference vertices would need two.
    assert edit_oracle(R, Q, 1.0, "insert", stats=stats) == 1
    assert stats["curves_evaluated"] >= 1
    assert stats["candidates"] == 5
    # deletion-only cannot reach R[2] = 4 at all
    assert edit_oracle(R, Q, 1.0, "delete") == math.inf
    assert edit_oracle(R, Q, 1.0, "both") == 1


def test_mixed_mode_needs_one_deletion_and_one_insertion():
    R = [[0.0], [2.0], [4.0]]
    Q = [[0.0], [9.0], [4.0]]
    assert edit_oracle(R, Q, 1.0, "delete") == math.inf  # 9 is uncoverable
    assert edit_oracle(R, Q, 1.0, "insert") == math.inf  # 9 cannot be removed
    assert edit_oracle(R, Q, 1.0, "both") == 2  # delete 9, insert 2


def test_deletion_only_case_where_insertion_cannot_help():
    R = [[0.0], [1.0]]
    Q = [[0.0], [9.0], [1.0]]
    assert edit_oracle(R, Q, 1.0, "delete") == 1
    assert edit_oracle(R, Q, 1.0, "insert") == math.inf
    assert edit_oracle(R, Q, 1.0, "both") == 1


@pytest.mark.parametrize(
    "R, Q, delta",
    [
        ([[0.0], [2.0], [4.0]], [[0.0]], 1.0),
        ([[0.0], [2.0], [4.0]], [[0.0], [9.0], [4.0]], 1.0),
        ([[0.0], [1.0]], [[0.0], [9.0], [1.0]], 1.0),
        ([[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 7.0]], 1.5),
        ([[0.0], [0.0], [0.0]], [[5.0], [5.0]], 1.0),
    ],
)
def test_mixed_mode_is_always_feasible_and_dominates_the_restricted_modes(R, Q, delta):
    m, n = len(R), len(Q)
    both = edit_oracle(R, Q, delta, "both")
    assert both <= m + n  # delete all of Q, then insert a cover of R
    assert math.isfinite(both)
    assert both <= edit_oracle(R, Q, delta, "delete")
    assert both <= edit_oracle(R, Q, delta, "insert")


def test_edit_oracle_delete_mode_agrees_with_the_subsequence_oracle():
    rng = random.Random(SEED + 23)
    for _ in range(40):
        m = rng.randint(1, 3)
        n = rng.randint(1, 3)
        R = [[float(rng.randrange(5))] for _ in range(m)]
        Q = [[float(rng.randrange(5))] for _ in range(n)]
        delta = 1.0
        assert edit_oracle(R, Q, delta, "delete") == deletion_oracle(R, Q, delta)


def _free_grid_cost(R, Q, delta, grid, max_cost):
    """Smallest edit cost <= ``max_cost`` when inserted points may be ANY point
    of ``grid``, not just a block centre.

    Deliberately written without :mod:`.block_edit_search` helpers so it also
    cross-checks the placement enumeration there.
    """
    m, n = len(R), len(Q)
    gaps = range(n + 1)
    for cost in range(max_cost + 1):
        for inserted in range(0, min(m, cost) + 1):
            deletions = cost - inserted
            if deletions > n:
                continue
            if inserted == 0:
                plans = [{}]
            elif inserted == 1:
                plans = [{g: (p,)} for g in gaps for p in grid]
            elif inserted == 2:
                plans = []
                for g1, g2 in itertools.combinations_with_replacement(gaps, 2):
                    for p1, p2 in itertools.product(grid, repeat=2):
                        plans.append(
                            {g1: (p1, p2)} if g1 == g2 else {g1: (p1,), g2: (p2,)}
                        )
            else:  # pragma: no cover - the caller keeps max_cost small
                raise AssertionError("grid cross-check only enumerates <= 2 insertions")
            for deleted in itertools.combinations(range(n), deletions):
                for plan in plans:
                    curve = replay(Q, frozenset(deleted), plan)
                    if curve and discrete_frechet_le(R, curve, delta):
                        return cost
    return math.inf


def test_block_centre_candidates_are_never_beaten_by_free_insertion_points():
    """Empirical check of lemma B.1.

    Insertions are allowed at ANY point of the space, so the oracle's
    restriction to exact block centres is only legitimate if a free choice of
    insertion point never yields a cheaper solution.
    """
    grid = [(Fraction(k, 2),) for k in range(-2, 11)]  # -1.0 .. 5.0, step 0.5
    values = (0.0, 1.0, 3.0)
    checked = 0
    for m in (1, 2):
        for n in (1, 2):
            for R in itertools.product(values, repeat=m):
                for Q in itertools.product(values, repeat=n):
                    curve_r = [[v] for v in R]
                    curve_q = [[v] for v in Q]
                    free = _free_grid_cost(curve_r, curve_q, 1.0, grid, max_cost=2)
                    if free is math.inf:
                        continue
                    assert edit_oracle(curve_r, curve_q, 1.0, "both") <= free
                    checked += 1
    assert checked > 50, "the cross-check must actually exercise feasible cases"


def test_edit_oracle_reports_state_counts_for_the_always_on_sizes():
    stats: dict = {}
    edit_oracle([[0.0], [2.0], [4.0]], [[0.0], [9.0], [4.0]], 1.0, "both", stats=stats)
    assert stats["curves_evaluated"] > 0
    assert stats["curves_evaluated"] < 100_000
    assert stats["elapsed_s"] >= 0.0


def test_budget_exceeded_is_raised_and_never_swallowed():
    R = [[0.0], [10.0]]
    Q = [[5.0]]
    stats: dict = {}
    with pytest.raises(OracleBudgetExceeded):
        edit_oracle(R, Q, 1.0, "both", stats=stats, max_states=1)
    assert stats["curves_evaluated"] >= 1  # counters survive the raise

    with pytest.raises(OracleBudgetExceeded):
        # a negative time budget is exceeded by construction, so this exercises
        # the timeout branch without depending on clock resolution
        edit_oracle(R, Q, 1.0, "both", timeout_s=-1.0)


def test_edit_oracle_input_guards():
    with pytest.raises(ValueError):
        edit_oracle([[0.0]], [[0.0]], 1.0, "substitute")
    with pytest.raises(ValueError):
        edit_oracle([], [[0.0]], 1.0)
    with pytest.raises(ValueError):
        edit_oracle([[0.0]], [], 1.0)
    with pytest.raises(ValueError):
        edit_oracle([[0.0]], [[0.0]], 0.0)
