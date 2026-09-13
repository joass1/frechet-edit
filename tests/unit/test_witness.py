"""Witness replay, coupling validity, and - most importantly - tamper rejection.

A verifier that cannot reject a corrupted witness is decoration. Every test in
:class:`TestTamperRejection` mutates exactly one thing and requires a named
violation, per docs/witness-invariants.md section 5.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from frechet_edit import (
    Deletion,
    Insertion,
    discrete_edit_distance,
    replay,
    verify_witness,
)
from tests.conftest import curve


@pytest.fixture
def deletion_case():
    ref = curve([0.0, 1.0, 2.0])
    obs = curve([0.0, 1.0, 100.0, 2.0])
    result = discrete_edit_distance(ref, obs, 0.1, operations="delete", return_witness=True)
    return ref, obs, result


@pytest.fixture
def mixed_case():
    ref = curve([0.0, 5.0, 10.0])
    obs = curve([0.0, 99.0, 10.0])
    result = discrete_edit_distance(ref, obs, 0.6, operations="both", return_witness=True)
    return ref, obs, result


@pytest.fixture
def rebuild_case():
    ref = curve([0.0, 4.0, 8.0])
    obs = curve([50.0, 60.0])
    result = discrete_edit_distance(ref, obs, 1.0, operations="both", return_witness=True)
    return ref, obs, result


class TestReplaySemantics:
    def test_indices_do_not_shift_after_a_deletion(self):
        obs = curve([0.0, 1.0, 2.0, 3.0])
        out = replay(obs, (Deletion(1), Deletion(2)))
        assert out.ravel().tolist() == [0.0, 3.0]

    def test_gap_zero_inserts_before_everything(self):
        obs = curve([1.0, 2.0])
        out = replay(obs, (Insertion(gap=0, order=0, point=(9.0,)),))
        assert out.ravel().tolist() == [9.0, 1.0, 2.0]

    def test_gap_n_inserts_after_everything(self):
        obs = curve([1.0, 2.0])
        out = replay(obs, (Insertion(gap=2, order=0, point=(9.0,)),))
        assert out.ravel().tolist() == [1.0, 2.0, 9.0]

    def test_order_within_a_gap_is_respected(self):
        obs = curve([1.0, 5.0])
        edits = (
            Insertion(gap=1, order=1, point=(3.0,)),
            Insertion(gap=1, order=0, point=(2.0,)),
        )
        assert replay(obs, edits).ravel().tolist() == [1.0, 2.0, 3.0, 5.0]

    def test_replay_works_when_every_original_vertex_is_deleted(self):
        obs = curve([1.0, 2.0])
        edits = (
            Deletion(0),
            Deletion(1),
            Insertion(gap=0, order=0, point=(7.0,)),
            Insertion(gap=2, order=0, point=(8.0,)),
        )
        assert replay(obs, edits).ravel().tolist() == [7.0, 8.0]

    def test_duplicate_deletion_rejected(self):
        with pytest.raises(ValueError, match="more than once"):
            replay(curve([1.0, 2.0]), (Deletion(0), Deletion(0)))

    def test_out_of_range_deletion_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            replay(curve([1.0, 2.0]), (Deletion(5),))

    def test_out_of_range_gap_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            replay(curve([1.0, 2.0]), (Insertion(gap=3, order=0, point=(0.0,)),))

    def test_non_contiguous_order_rejected(self):
        with pytest.raises(ValueError, match="non-contiguous"):
            replay(
                curve([1.0, 2.0]),
                (
                    Insertion(gap=1, order=0, point=(0.0,)),
                    Insertion(gap=1, order=2, point=(0.0,)),
                ),
            )


class TestHonestWitnessesVerify:
    def test_deletion_witness(self, deletion_case):
        ref, obs, result = deletion_case
        report = verify_witness(ref, obs, result)
        assert report.ok, report.violations
        assert report.ordinary_frechet <= result.delta

    def test_mixed_witness(self, mixed_case):
        ref, obs, result = mixed_case
        verify_witness(ref, obs, result).raise_for_status()

    def test_rebuild_witness(self, rebuild_case):
        ref, obs, result = rebuild_case
        verify_witness(ref, obs, result).raise_for_status()

    def test_verifier_refuses_a_result_without_a_witness(self):
        ref, obs = curve([0.0]), curve([0.0])
        result = discrete_edit_distance(ref, obs, 1.0)
        report = verify_witness(ref, obs, result)
        assert not report.ok
        assert "not_requested" in report.violations[0]


def _tampered(result, **changes):
    return dataclasses.replace(result, **changes)


class TestTamperRejection:
    def test_wrong_deletion_index(self, deletion_case):
        ref, obs, result = deletion_case
        bad = _tampered(result, edits=(Deletion(1),))
        report = verify_witness(ref, obs, bad)
        assert not report.ok and report.violations

    def test_reported_cost_inflated(self, deletion_case):
        ref, obs, result = deletion_case
        bad = _tampered(result, cost=5)
        report = verify_witness(ref, obs, bad)
        assert any("cost" in v for v in report.violations), report.violations

    def test_reported_cost_deflated(self, deletion_case):
        ref, obs, result = deletion_case
        bad = _tampered(result, cost=0)
        report = verify_witness(ref, obs, bad)
        assert any("cost" in v for v in report.violations), report.violations

    def test_edited_curve_does_not_match_the_edits(self, deletion_case):
        ref, obs, result = deletion_case
        forged = result.edited_curve.copy()
        forged[0, 0] = 42.0
        bad = _tampered(result, edited_curve=forged)
        report = verify_witness(ref, obs, bad)
        assert any("does not match" in v for v in report.violations), report.violations

    def test_moved_inserted_coordinate(self, mixed_case):
        ref, obs, result = mixed_case
        edits = tuple(
            Insertion(gap=e.gap, order=e.order, point=(e.point[0] + 3.0,))
            if isinstance(e, Insertion)
            else e
            for e in result.edits
        )
        bad = _tampered(result, edits=edits)
        report = verify_witness(ref, obs, bad)
        assert not report.ok and report.violations

    def test_moved_insertion_to_a_different_gap(self, mixed_case):
        ref, obs, result = mixed_case
        edits = tuple(
            Insertion(gap=0, order=e.order, point=e.point) if isinstance(e, Insertion) else e
            for e in result.edits
        )
        bad = _tampered(result, edits=edits)
        report = verify_witness(ref, obs, bad)
        assert not report.ok and report.violations

    def test_swapped_insertion_order_within_a_gap(self, rebuild_case):
        ref, obs, result = rebuild_case
        inserts = [e for e in result.edits if isinstance(e, Insertion)]
        assert len(inserts) >= 2, "fixture must contain at least two insertions in one gap"
        gap = inserts[0].gap
        same_gap = [e for e in inserts if e.gap == gap]
        assert len(same_gap) >= 2
        flipped = {
            (same_gap[0].gap, same_gap[0].order): same_gap[1].point,
            (same_gap[1].gap, same_gap[1].order): same_gap[0].point,
        }
        edits = tuple(
            Insertion(gap=e.gap, order=e.order, point=flipped.get((e.gap, e.order), e.point))
            if isinstance(e, Insertion)
            else e
            for e in result.edits
        )
        bad = _tampered(result, edits=edits)
        report = verify_witness(ref, obs, bad)
        assert not report.ok and report.violations

    def test_coupling_missing_an_endpoint(self, deletion_case):
        ref, obs, result = deletion_case
        bad = _tampered(result, coupling=result.coupling[1:])
        report = verify_witness(ref, obs, bad)
        assert any("starts at" in v for v in report.violations), report.violations

    def test_coupling_truncated_at_the_end(self, deletion_case):
        ref, obs, result = deletion_case
        bad = _tampered(result, coupling=result.coupling[:-1])
        report = verify_witness(ref, obs, bad)
        assert any("ends at" in v for v in report.violations), report.violations

    def test_coupling_with_an_illegal_double_jump(self):
        ref = curve([0.0, 1.0, 2.0, 3.0])
        obs = curve([0.0, 1.0, 2.0, 3.0])
        result = discrete_edit_distance(ref, obs, 0.1, return_witness=True)
        bad = _tampered(result, coupling=((0, 0), (2, 2), (3, 3)))
        report = verify_witness(ref, obs, bad)
        assert any("only (1,0)" in v for v in report.violations), report.violations

    def test_coupling_that_goes_backwards(self):
        ref = curve([0.0, 1.0, 2.0])
        obs = curve([0.0, 1.0, 2.0])
        result = discrete_edit_distance(ref, obs, 0.1, return_witness=True)
        bad = _tampered(result, coupling=((0, 0), (1, 1), (0, 1), (2, 2)))
        report = verify_witness(ref, obs, bad)
        assert not report.ok and report.violations

    def test_coupling_pair_exceeding_delta(self):
        ref = curve([0.0, 1.0, 9.0])
        obs = curve([0.0, 1.0, 9.0])
        result = discrete_edit_distance(ref, obs, 0.1, return_witness=True)
        # Pair reference vertex 2 (at 9.0) with edited vertex 0 (at 0.0).
        bad = _tampered(result, coupling=((0, 0), (1, 1), (2, 1), (2, 2)))
        report = verify_witness(ref, obs, bad)
        assert any("> delta" in v for v in report.violations), report.violations

    def test_mode_violation_is_caught(self, mixed_case):
        ref, obs, result = mixed_case
        bad = _tampered(result, mode="delete")
        report = verify_witness(ref, obs, bad)
        assert any("insertion" in v for v in report.violations), report.violations

    def test_claiming_a_witness_that_is_not_there(self, deletion_case):
        ref, obs, result = deletion_case
        bad = _tampered(result, edits=None)
        report = verify_witness(ref, obs, bad)
        assert any("incomplete" in v for v in report.violations), report.violations

    def test_raise_for_status_raises_on_a_bad_witness(self, deletion_case):
        ref, obs, result = deletion_case
        bad = _tampered(result, cost=99)
        with pytest.raises(AssertionError, match="verification failed"):
            verify_witness(ref, obs, bad).raise_for_status()


class TestCouplingStructure:
    def test_coupling_covers_both_curves_completely(self):
        rng = np.random.default_rng(67)
        for _ in range(40):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 5)), 2)) * 2, 2)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 5)), 2)) * 2, 2)
            delta = float(rng.uniform(0.5, 3.0))
            result = discrete_edit_distance(
                ref, obs, delta, operations="both", return_witness=True
            )
            if result.witness_status != "certified":
                continue
            ref_seen = {p[0] for p in result.coupling}
            edit_seen = {p[1] for p in result.coupling}
            assert ref_seen == set(range(len(ref)))
            assert edit_seen == set(range(len(result.edited_curve)))

    def test_retained_vertices_appear_unchanged_and_in_order(self):
        ref = curve([0.0, 1.0, 2.0, 3.0])
        obs = curve([0.0, 77.0, 1.0, 2.0, 88.0, 3.0])
        result = discrete_edit_distance(ref, obs, 0.2, operations="both", return_witness=True)
        deleted = {e.index for e in result.edits if isinstance(e, Deletion)}
        kept = [obs[i, 0] for i in range(len(obs)) if i not in deleted]
        produced = result.edited_curve.ravel().tolist()
        # Every kept original value survives, in order, inside the edited curve.
        it = iter(produced)
        assert all(any(v == x for x in it) for v in kept)
