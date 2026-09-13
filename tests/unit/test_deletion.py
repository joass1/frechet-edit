"""Deletion-only mode: the P1 correctness spine."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from frechet_edit import discrete_edit_distance, ordinary_discrete_frechet, verify_witness
from tests.conftest import ALL_BACKENDS, curve


def fed(ref, obs, delta, **kw):
    return discrete_edit_distance(curve(ref), curve(obs), delta, operations="delete", **kw)


class TestCanonicalFixtures:
    def test_identical_singletons_need_no_edit(self):
        result = fed([0.0], [0.0], 1.0, return_witness=True)
        assert (result.status, result.cost) == ("optimal", 0)
        assert result.edits == ()

    def test_isolated_outlier_costs_one_deletion(self):
        """Plan fixture 1: the spike at 100 is removed, nothing else."""
        result = fed([0.0, 1.0, 2.0], [0.0, 1.0, 100.0, 2.0], 0.1, return_witness=True)
        assert result.cost == 1
        assert [e.as_dict() for e in result.edits] == [{"op": "delete", "index": 2}]

    def test_prefix_outlier(self):
        result = fed([0.0, 1.0], [-50.0, 0.0, 1.0], 0.1, return_witness=True)
        assert result.cost == 1
        assert result.edits[0].index == 0

    def test_suffix_outlier(self):
        result = fed([0.0, 1.0], [0.0, 1.0, 50.0], 0.1, return_witness=True)
        assert result.cost == 1
        assert result.edits[0].index == 2

    def test_both_ends_corrupted(self):
        result = fed([0.0, 1.0], [-9.0, 0.0, 1.0, 9.0], 0.1, return_witness=True)
        assert result.cost == 2
        assert sorted(e.index for e in result.edits) == [0, 3]

    def test_repeated_vertices_cost_nothing(self):
        """Discrete Frechet tolerates unequal multiplicity; no string-style charge."""
        assert fed([0.0, 0.0, 0.0], [0.0, 0.0], 0.5).cost == 0
        assert fed([0.0], [0.0, 0.0, 0.0, 0.0], 0.5).cost == 0

    def test_one_vertex_may_serve_many_reference_vertices(self):
        """A single retained vertex covers a whole run: this is not LCS."""
        result = fed([0.0, 0.1, 0.2, 0.3], [0.15], 0.5, return_witness=True)
        assert result.cost == 0
        assert len(result.coupling) == 4

    def test_impossible_deletion_is_infeasible(self):
        result = fed([0.0, 10.0], [0.0], 1.0)
        assert result.status == "infeasible"
        assert math.isinf(result.cost)
        assert result.witness_status == "not_requested"

    def test_deleting_every_vertex_is_not_a_solution_in_delete_only(self):
        """The edited curve may never be empty against a non-empty reference."""
        result = fed([5.0], [0.0, 1.0], 0.1)
        assert result.status == "infeasible"

    def test_one_point_repaired_output(self):
        result = fed([3.0], [0.0, 3.0, 9.0], 0.1, return_witness=True)
        assert result.cost == 2
        assert result.edited_curve.ravel().tolist() == [3.0]


class TestProperties:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_cost_is_non_increasing_in_delta(self, backend):
        ref = curve([0.0, 1.0, 2.0, 3.0])
        obs = curve([0.0, 1.4, 1.0, 2.0, 8.0, 3.0])
        costs = [
            discrete_edit_distance(
                ref, obs, d, operations="delete", backend=backend
            ).cost
            for d in (0.05, 0.2, 0.5, 1.0, 5.0, 50.0)
        ]
        assert all(a >= b for a, b in itertools.pairwise(costs)), costs

    def test_zero_edits_exactly_when_ordinary_frechet_fits(self):
        rng = np.random.default_rng(5)
        for _ in range(60):
            ref = np.round(rng.normal(size=(rng.integers(1, 6), 1)), 3)
            obs = np.round(rng.normal(size=(rng.integers(1, 6), 1)), 3)
            delta = float(rng.uniform(0.1, 3.0))
            result = discrete_edit_distance(ref, obs, delta, operations="delete")
            fits = ordinary_discrete_frechet(ref, obs) <= delta
            assert (result.status == "optimal" and result.cost == 0) == fits

    def test_scaling_coordinates_and_delta_together_preserves_cost(self):
        ref = curve([0.0, 1.0, 2.0])
        obs = curve([0.0, 1.0, 100.0, 2.0])
        base = fed(ref, obs, 0.1).cost
        for factor in (1e-3, 3.0, 1e4):
            scaled = discrete_edit_distance(
                ref * factor, obs * factor, 0.1 * factor, operations="delete"
            )
            assert scaled.cost == base, factor

    def test_common_translation_preserves_cost(self):
        ref = curve([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        obs = curve([[0.0, 0.0], [40.0, 40.0], [2.0, 0.0]])
        base = discrete_edit_distance(ref, obs, 0.5, operations="delete").cost
        shift = np.array([123.5, -77.25])
        moved = discrete_edit_distance(ref + shift, obs + shift, 0.5, operations="delete")
        assert moved.cost == base

    def test_witness_always_verifies(self):
        rng = np.random.default_rng(17)
        checked = 0
        for _ in range(80):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 6)), 1)) * 2, 2)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 7)), 1)) * 2, 2)
            delta = float(rng.uniform(0.2, 2.5))
            result = discrete_edit_distance(
                ref, obs, delta, operations="delete", return_witness=True
            )
            if result.status != "optimal":
                continue
            assert result.witness_status == "certified"
            verify_witness(ref, obs, result).raise_for_status()
            checked += 1
        assert checked >= 20, f"only {checked} feasible cases exercised"


class TestNotAnEditDistanceOnStrings:
    def test_stationary_runs_are_free(self):
        """Levenshtein would charge for the length difference; discrete Frechet does not."""
        ref = curve([0.0] * 8)
        obs = curve([0.0])
        assert fed(ref, obs, 0.1).cost == 0

    def test_deletion_cannot_reorder(self):
        """Deletions preserve order, so a reversed observation is not repairable cheaply."""
        ref = curve([0.0, 1.0, 2.0])
        obs = curve([2.0, 1.0, 0.0])
        result = fed(ref, obs, 0.1)
        assert result.status == "infeasible"
