"""Mixed insertion/deletion mode."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from frechet_edit import discrete_edit_distance, verify_witness
from tests.conftest import ALL_BACKENDS, curve


def fed(ref, obs, delta, mode="both", **kw):
    return discrete_edit_distance(curve(ref), curve(obs), delta, operations=mode, **kw)


class TestCoreBehaviour:
    def test_mixed_never_exceeds_either_restricted_mode(self):
        rng = np.random.default_rng(41)
        for _ in range(80):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 5)), 1)) * 3, 2)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 5)), 1)) * 3, 2)
            delta = float(rng.uniform(0.3, 3.0))
            costs = {
                mode: discrete_edit_distance(ref, obs, delta, operations=mode).cost
                for mode in ("delete", "insert", "both")
            }
            assert costs["both"] <= costs["delete"], costs
            assert costs["both"] <= costs["insert"], costs

    def test_mixed_is_always_feasible_and_bounded_by_m_plus_n(self):
        rng = np.random.default_rng(43)
        for _ in range(80):
            m = int(rng.integers(1, 6))
            n = int(rng.integers(1, 6))
            ref = np.round(rng.normal(size=(m, 2)) * 50, 2)
            obs = np.round(rng.normal(size=(n, 2)) * 50, 2)
            delta = float(rng.uniform(0.05, 1.0))
            result = discrete_edit_distance(ref, obs, delta, operations="both")
            assert result.status == "optimal"
            assert result.cost <= m + n, (result.cost, m, n)

    def test_delete_all_and_rebuild(self):
        """Every original vertex removed, then the reference supplied by insertion."""
        result = fed([0.0, 10.0], [100.0, 200.0], 1.0, return_witness=True)
        assert result.cost == 4
        ops = [e.as_dict() for e in result.edits]
        assert sum(o["op"] == "delete" for o in ops) == 2
        assert sum(o["op"] == "insert" for o in ops) == 2
        assert result.edited_curve.ravel().tolist() == [0.0, 10.0]

    def test_gap_ordering_survives_deleting_every_original_vertex(self):
        """Insertion order must be well defined even with no surviving anchors."""
        result = fed([0.0, 4.0, 8.0], [50.0, 60.0], 1.0, return_witness=True)
        inserted = [e for e in result.edits if e.as_dict()["op"] == "insert"]
        keys = [(e.gap, e.order) for e in inserted]
        assert keys == sorted(keys)
        points = [e.point[0] for e in inserted]
        assert points == sorted(points)
        verify_witness(curve([0.0, 4.0, 8.0]), curve([50.0, 60.0]), result).raise_for_status()

    def test_one_deletion_and_one_insertion(self):
        ref = curve([0.0, 5.0, 10.0])
        obs = curve([0.0, 99.0, 10.0])
        result = discrete_edit_distance(ref, obs, 0.6, operations="both", return_witness=True)
        assert result.cost == 2
        ops = sorted(e.as_dict()["op"] for e in result.edits)
        assert ops == ["delete", "insert"]
        verify_witness(ref, obs, result).raise_for_status()

    def test_the_discriminating_ranking_fixture(self):
        """Plan fixture 1: ordinary Frechet and FED rank the two candidates oppositely."""
        from frechet_edit import ordinary_discrete_frechet

        ref = curve([0.0, 1.0, 2.0])
        noisy = curve([0.0, 1.0, 100.0, 2.0])  # correct route, one spike
        wrong = curve([0.0, 1.0, 3.0])  # different route, no spike

        assert ordinary_discrete_frechet(ref, noisy) == 98.0
        assert ordinary_discrete_frechet(ref, wrong) == 1.0
        # Ordinary Frechet prefers the wrong candidate.
        assert ordinary_discrete_frechet(ref, wrong) < ordinary_discrete_frechet(ref, noisy)

        fed_noisy = fed(ref, noisy, 0.1).cost
        fed_wrong = fed(ref, wrong, 0.1).cost
        assert (fed_noisy, fed_wrong) == (1, 2)
        # FED prefers the noisy-but-correct candidate. Mechanism, not generalisation.
        assert fed_noisy < fed_wrong


class TestFailureModeFixtures:
    def test_a_genuine_detour_can_be_erased_cheaply(self):
        """Explicit failure-mode fixture: FED does not distinguish noise from intent."""
        ref = curve([0.0, 1.0, 2.0, 3.0])
        detour = curve([0.0, 1.0, 1.5, 40.0, 1.5, 2.0, 3.0])
        result = fed(ref, detour, 0.6)
        # A real excursion of two vertices costs the same as two noise spikes would.
        assert result.cost <= 2

    def test_insertion_does_not_help_a_tolerated_dropout(self):
        """Discrete Frechet already tolerates missing samples; no insertion is needed."""
        ref = curve([0.0, 0.1, 0.2, 0.3])
        obs = curve([0.0, 0.3])
        assert fed(ref, obs, 0.4).cost == 0


class TestProperties:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_cost_is_non_increasing_in_delta(self, backend):
        ref = curve([0.0, 3.0, 6.0, 9.0])
        obs = curve([0.0, 40.0, 9.0])
        costs = [
            discrete_edit_distance(ref, obs, d, operations="both", backend=backend).cost
            for d in (0.1, 0.5, 1.0, 2.0, 5.0, 100.0)
        ]
        assert all(a >= b for a, b in itertools.pairwise(costs)), costs

    def test_joint_translation_preserves_cost(self):
        rng = np.random.default_rng(47)
        for _ in range(30):
            ref = np.round(rng.normal(size=(4, 2)) * 3, 2)
            obs = np.round(rng.normal(size=(4, 2)) * 3, 2)
            delta = float(rng.uniform(0.5, 3.0))
            base = discrete_edit_distance(ref, obs, delta, operations="both").cost
            shift = np.round(rng.normal(size=2) * 500, 2)
            moved = discrete_edit_distance(ref + shift, obs + shift, delta, operations="both").cost
            assert moved == base, (base, moved, shift)

    def test_joint_rotation_preserves_cost(self):
        rng = np.random.default_rng(53)
        for _ in range(20):
            ref = np.round(rng.normal(size=(4, 2)) * 3, 3)
            obs = np.round(rng.normal(size=(4, 2)) * 3, 3)
            delta = float(rng.uniform(0.8, 3.0))
            theta = float(rng.uniform(0, 2 * np.pi))
            rot = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            base = discrete_edit_distance(ref, obs, delta, operations="both").cost
            turned = discrete_edit_distance(ref @ rot.T, obs @ rot.T, delta, operations="both").cost
            # Away from numerical boundaries this is exact; delta is generous here.
            assert turned == base, (base, turned, theta)

    def test_score_only_and_witness_calls_agree(self):
        rng = np.random.default_rng(59)
        for _ in range(40):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 5)), 2)) * 2, 2)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 5)), 2)) * 2, 2)
            delta = float(rng.uniform(0.4, 3.0))
            plain = discrete_edit_distance(ref, obs, delta, operations="both")
            witnessed = discrete_edit_distance(
                ref, obs, delta, operations="both", return_witness=True
            )
            assert plain.cost == witnessed.cost
            assert plain.status == witnessed.status
