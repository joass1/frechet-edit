"""The public contract of docs/definition.md: validation, direction, statuses."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from frechet_edit import (
    UnsupportedDimensionError,
    discrete_edit_distance,
    ordinary_discrete_frechet,
)
from tests.conftest import ALL_BACKENDS, ALL_MODES, curve


class TestInputValidation:
    @pytest.mark.parametrize("empty", [[], np.empty((0, 2))])
    def test_empty_reference_rejected(self, empty):
        with pytest.raises(ValueError, match="non-empty"):
            discrete_edit_distance(empty, curve([0.0]), delta=1.0)

    def test_empty_observation_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            discrete_edit_distance(curve([0.0]), np.empty((0, 1)), delta=1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_delta_rejected(self, bad):
        with pytest.raises(ValueError):
            discrete_edit_distance(curve([0.0]), curve([0.0]), delta=bad)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_coordinates_rejected(self, bad):
        with pytest.raises(ValueError, match="non-finite"):
            discrete_edit_distance(curve([0.0, bad]), curve([0.0]), delta=1.0)

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(ValueError, match="dimensions differ"):
            discrete_edit_distance(curve([[0.0, 0.0]]), curve([0.0]), delta=1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"operations": "substitute"},
            {"backend": "cuda"},
            {"numeric_policy": "yolo"},
        ],
    )
    def test_unknown_option_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            discrete_edit_distance(curve([0.0]), curve([0.0]), delta=1.0, **kwargs)

    def test_inputs_are_never_mutated(self):
        ref = curve([0.0, 5.0, 1.0])
        obs = curve([0.0, 99.0, 1.0])
        ref_copy, obs_copy = ref.copy(), obs.copy()
        discrete_edit_distance(ref, obs, delta=0.5, return_witness=True)
        assert np.array_equal(ref, ref_copy)
        assert np.array_equal(obs, obs_copy)

    def test_order_and_duplicates_are_preserved(self):
        """Nothing is sorted or deduplicated: multiplicity is part of the input."""
        ref = curve([0.0, 0.0, 0.0])
        obs = curve([0.0, 0.0])
        result = discrete_edit_distance(ref, obs, delta=0.5, return_witness=True)
        assert result.cost == 0
        assert result.edited_curve.shape == (2, 1)

    def test_1d_flat_input_is_read_as_scalars(self):
        flat = discrete_edit_distance([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], delta=0.1)
        shaped = discrete_edit_distance(curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 2.0]), delta=0.1)
        assert flat.cost == shaped.cost == 0
        assert flat.dimension == 1


class TestDimensionScope:
    def test_deletion_accepts_higher_dimensions(self):
        ref = np.zeros((3, 5))
        obs = np.zeros((3, 5))
        assert discrete_edit_distance(ref, obs, delta=1.0, operations="delete").cost == 0

    @pytest.mark.parametrize("mode", ["insert", "both"])
    def test_insertion_rejects_unsupported_dimension_explicitly(self, mode):
        ref = np.zeros((3, 3))
        obs = np.zeros((3, 3))
        with pytest.raises(UnsupportedDimensionError) as exc:
            discrete_edit_distance(ref, obs, delta=1.0, operations=mode)
        # Not silently degraded, and not reported as infeasible.
        assert "dimensions (1, 2)" in str(exc.value)


class TestDirectedness:
    def test_arguments_are_not_interchangeable(self):
        """FED is directed: only the observation is edited."""
        short = curve([0.0])
        long = curve([0.0, 2.0, 4.0])
        forward = discrete_edit_distance(long, short, delta=1.0, operations="insert")
        backward = discrete_edit_distance(short, long, delta=1.0, operations="insert")
        assert forward.cost == 1
        # Editing the 3-vertex observation by insertion alone cannot shrink it.
        assert backward.status == "infeasible"

    def test_delete_only_is_directed_too(self):
        ref = curve([0.0, 1.0, 2.0])
        obs = curve([0.0, 1.0, 100.0, 2.0])
        assert discrete_edit_distance(ref, obs, delta=0.1, operations="delete").cost == 1
        assert (
            discrete_edit_distance(obs, ref, delta=0.1, operations="delete").status
            == "infeasible"
        )


class TestStatusSemantics:
    def test_infeasible_is_a_status_not_an_exception(self):
        result = discrete_edit_distance(
            curve([0.0, 10.0]), curve([0.0]), delta=1.0, operations="delete"
        )
        assert result.status == "infeasible"
        assert math.isinf(result.cost)
        assert result.is_feasible is False

    def test_json_never_emits_bare_infinity(self):
        result = discrete_edit_distance(
            curve([0.0, 10.0]), curve([0.0]), delta=1.0, operations="delete"
        )
        blob = json.dumps(result.to_json_obj())
        assert "Infinity" not in blob
        payload = json.loads(blob)
        assert payload["cost"] is None
        assert payload["cost_is_infinite"] is True
        assert payload["status"] == "infeasible"

    def test_json_roundtrips_for_an_optimal_witness(self):
        result = discrete_edit_distance(
            curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 100.0, 2.0]), delta=0.1,
            return_witness=True,
        )
        payload = json.loads(json.dumps(result.to_json_obj()))
        assert payload["cost"] == 1
        assert payload["witness_status"] == "certified"
        assert payload["edits"] == [{"op": "delete", "index": 2}]

    def test_witness_status_defaults_to_not_requested(self):
        result = discrete_edit_distance(curve([0.0]), curve([0.0]), delta=1.0)
        assert result.witness_status == "not_requested"
        assert result.edits is None and result.coupling is None


class TestBackendParity:
    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize(
        "ref,obs,delta",
        [
            ([0.0, 1.0, 2.0], [0.0, 1.0, 100.0, 2.0], 0.1),
            ([0.0, 2.0, 4.0], [0.0], 1.0),
            ([0.0, 10.0], [100.0, 200.0], 1.0),
            ([0.0, 0.0, 0.0], [0.0, 0.0], 0.5),
            ([0.0, 1.0], [0.0, 0.5, 1.0], 0.4),
        ],
    )
    def test_backends_agree_on_cost(self, ref, obs, delta, mode):
        costs = {
            backend: discrete_edit_distance(
                curve(ref), curve(obs), delta, operations=mode, backend=backend
            ).cost
            for backend in ALL_BACKENDS
        }
        assert costs["python"] == costs["reference"], costs


class TestOrdinaryFrechetHelper:
    def test_known_value(self):
        """Every traversal must pair 100 with its closest reference vertex, 2."""
        value = ordinary_discrete_frechet(curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 100.0, 2.0]))
        assert value == 98.0

    def test_zero_edits_iff_ordinary_frechet_within_delta(self):
        ref = curve([0.0, 1.0, 2.0])
        obs = curve([0.1, 1.1, 2.1])
        ordinary = ordinary_discrete_frechet(ref, obs)
        for delta in (0.05, 0.11, 1.0):
            result = discrete_edit_distance(ref, obs, delta, operations="both")
            assert (result.cost == 0) == (ordinary <= delta), (delta, ordinary, result.cost)
