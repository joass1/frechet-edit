"""Insertion-only mode: arbitrary inserted locations, not reference vertices.

The separating fixtures here are the ones that catch an implementation which
quietly restricts insertion to copies of reference vertices, or which skips the
insertion branch whenever the current observation vertex happens to be close.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from frechet_edit import discrete_edit_distance, verify_witness
from tests.conftest import ALL_BACKENDS, curve


def fed(ref, obs, delta, **kw):
    return discrete_edit_distance(curve(ref), curve(obs), delta, operations="insert", **kw)


class TestArbitraryInsertionLocations:
    def test_one_inserted_point_covers_two_reference_vertices(self):
        """Plan fixture 2. Inserting reference vertices alone would cost 2."""
        result = fed([0.0, 2.0, 4.0], [0.0], 1.0, return_witness=True)
        assert result.cost == 1
        inserted = [e for e in result.edits if e.as_dict()["op"] == "insert"]
        assert len(inserted) == 1
        point = inserted[0].point[0]
        assert point == pytest.approx(3.0)
        # And it is genuinely not a reference vertex.
        assert point not in {0.0, 2.0, 4.0}

    def test_one_inserted_point_covers_a_long_block(self):
        """Nine reference vertices, one insertion.

        Insert-only retains every observation vertex, so the observation must
        start delta-close to the reference; it is the REST of the reference
        that a single inserted point has to cover.
        """
        ref = curve(np.linspace(10.0, 12.0, 9))
        result = fed(ref, [10.0], 1.0, return_witness=True)
        assert result.cost == 1
        inserted = [e for e in result.edits if e.as_dict()["op"] == "insert"]
        assert len(inserted) == 1
        point = inserted[0].point[0]
        assert 10.0 < point < 12.0
        assert point not in set(ref.ravel().tolist())

    def test_two_dimensional_block_centre(self):
        ref = curve([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
        obs = curve([[0.0, 0.0]])
        result = discrete_edit_distance(
            ref, obs, 1.6, operations="insert", return_witness=True
        )
        assert result.cost == 1
        centre = result.edits[0].point
        assert centre[0] == pytest.approx(4.5) and centre[1] == pytest.approx(0.0)

    def test_shrinking_delta_forces_more_insertions(self):
        ref = curve([0.0, 2.0, 4.0, 6.0])
        assert fed(ref, [0.0], 3.0).cost == 1
        assert fed(ref, [0.0], 1.0).cost == 2
        assert fed(ref, [0.0], 0.1).cost == 3


class TestInsertionPositions:
    def test_insertion_before_the_first_observation_vertex(self):
        result = fed([0.0, 5.0], [5.0], 0.1, return_witness=True)
        assert result.cost == 1
        assert result.edits[0].gap == 0
        assert result.edited_curve.ravel().tolist() == [0.0, 5.0]

    def test_insertion_after_the_last_observation_vertex(self):
        result = fed([0.0, 5.0], [0.0], 0.1, return_witness=True)
        assert result.cost == 1
        assert result.edits[0].gap == 1
        assert result.edited_curve.ravel().tolist() == [0.0, 5.0]

    def test_insertion_into_an_internal_gap(self):
        result = fed([0.0, 5.0, 10.0], [0.0, 10.0], 0.1, return_witness=True)
        assert result.cost == 1
        assert result.edits[0].gap == 1
        assert result.edited_curve.ravel().tolist() == [0.0, 5.0, 10.0]

    def test_several_insertions_in_one_gap_keep_their_order(self):
        result = fed([0.0, 3.0, 6.0, 9.0], [0.0, 9.0], 0.6, return_witness=True)
        inserted = [e for e in result.edits if e.as_dict()["op"] == "insert"]
        assert len(inserted) >= 2
        gaps = [e.gap for e in inserted]
        orders = [e.order for e in inserted if e.gap == 1]
        assert gaps == sorted(gaps)
        assert orders == list(range(len(orders)))
        # Replayed points must ascend along the reference direction.
        points = [e.point[0] for e in inserted if e.gap == 1]
        assert points == sorted(points)


class TestCloseCellStillConsidersInsertion:
    def test_insertion_is_evaluated_even_when_the_vertex_is_close(self):
        """Closeness of Q[j] to R[i] must not suppress the insertion branch.

        Here the observation vertex sits exactly on the first reference vertex,
        yet the optimal prefix still has to insert to reach the rest.
        """
        result = fed([0.0, 4.0, 8.0], [0.0], 2.0, return_witness=True)
        assert result.cost == 1
        assert result.edits[0].point[0] == pytest.approx(6.0)

    def test_close_cell_whose_best_solution_ends_in_an_inserted_point(self):
        ref = curve([0.0, 1.0, 20.0])
        obs = curve([0.0, 1.0])
        result = fed(ref, obs, 0.5, return_witness=True)
        assert result.cost == 1
        assert result.edits[0].gap == 2
        assert result.edited_curve.ravel().tolist()[-1] == pytest.approx(20.0)


class TestInfeasibility:
    def test_insertion_cannot_remove_an_outlier(self):
        result = fed([0.0, 1.0, 2.0], [0.0, 1.0, 100.0, 2.0], 0.1)
        assert result.status == "infeasible"
        assert math.isinf(result.cost)

    def test_insertion_cannot_reorder(self):
        result = fed([0.0, 1.0], [1.0, 0.0], 0.1)
        assert result.status == "infeasible"

    def test_every_original_vertex_is_retained(self):
        result = fed([0.0, 5.0], [0.0], 0.1, return_witness=True)
        assert all(e.as_dict()["op"] == "insert" for e in result.edits)
        assert 0.0 in result.edited_curve.ravel().tolist()


class TestProperties:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_cost_is_non_increasing_in_delta(self, backend):
        ref = curve([0.0, 2.0, 4.0, 6.0, 8.0])
        obs = curve([0.0, 8.0])
        costs = [
            discrete_edit_distance(ref, obs, d, operations="insert", backend=backend).cost
            for d in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0)
        ]
        assert all(a >= b for a, b in itertools.pairwise(costs)), costs

    def test_insertion_count_never_exceeds_reference_length(self):
        """docs/oracle-protocol.md B.2: an optimal solution uses at most m insertions."""
        rng = np.random.default_rng(23)
        for _ in range(60):
            m = int(rng.integers(1, 7))
            ref = np.round(rng.normal(size=(m, 1)) * 3, 2)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 5)), 1)) * 3, 2)
            delta = float(rng.uniform(0.3, 4.0))
            result = discrete_edit_distance(ref, obs, delta, operations="insert")
            if result.status == "optimal":
                assert result.cost <= m, (result.cost, m)

    def test_witness_always_verifies(self):
        rng = np.random.default_rng(31)
        checked = 0
        for _ in range(80):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 6)), 2)) * 2, 2)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 5)), 2)) * 2, 2)
            delta = float(rng.uniform(0.5, 4.0))
            result = discrete_edit_distance(
                ref, obs, delta, operations="insert", return_witness=True
            )
            if result.status != "optimal" or result.witness_status != "certified":
                continue
            verify_witness(ref, obs, result).raise_for_status()
            checked += 1
        assert checked >= 15, f"only {checked} feasible cases exercised"
