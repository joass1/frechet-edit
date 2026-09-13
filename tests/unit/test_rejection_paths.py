"""Input rejection and witness rejection paths that the main suites don't reach.

The rest of the suite exercises the happy paths and the tamper cases that
naturally fall out of ordinary fixtures (see ``test_witness.py``). This file
targets what is left: boundary-validation branches that need a specific kind
of malformed input to trip (a non-numeric coordinate, a rank-3 array, a
zero-width point, a delta that isn't a number at all), and verifier /
traceback branches that guard against internal states no correct computation
can produce.

Some of those guard branches truly cannot be reached by tampering with an
``EditResult`` alone, because the value they check is *recomputed from the
same tampered fields* elsewhere in the same function, so it is trivially
self-consistent. For those, a small number of tests inject a fault directly
(monkeypatching a named function, or building a deliberately corrupted
``DPTables``), the same style already used in ``test_numeric_boundary.py``.
Each is labelled FAULT INJECTED with a comment explaining why the organic
path does not exist.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from frechet_edit import (
    Deletion,
    Insertion,
    _reference_dp,
    _traceback,
    discrete_edit_distance,
    replay,
    verify_witness,
)
from frechet_edit import verify as verify_module
from frechet_edit._numerics import NumericallyAmbiguous
from frechet_edit._reference_dp import FROM_K, FROM_NONE, FROM_P, FROM_X, INF, DPTables
from frechet_edit._traceback import TracebackError, reconstruct
from frechet_edit.verify import discrete_frechet
from tests.conftest import curve


def _tampered(result, **changes):
    """Return a copy of an ``EditResult`` with named fields replaced.

    ``EditResult`` is frozen, so tampering has to go through ``replace``
    rather than assignment - which is itself the point: a caller cannot
    accidentally mutate a witness, only deliberately forge one.
    """
    return dataclasses.replace(result, **changes)


class TestAsCurveRejectsMalformedInput:
    """`_validation.as_curve`, reached through the public entry point."""

    def test_non_numeric_coordinate_is_rejected(self):
        with pytest.raises(ValueError, match="numeric array of points"):
            discrete_edit_distance([["a"]], curve([0.0]), delta=1.0)

    def test_rank_three_input_is_rejected(self):
        with pytest.raises(ValueError, match=r"\(k, d\) array"):
            discrete_edit_distance(np.zeros((2, 2, 2)), np.zeros((2, 2)), delta=1.0)

    def test_zero_width_points_are_rejected(self):
        with pytest.raises(ValueError, match="zero-width"):
            discrete_edit_distance(np.empty((3, 0)), np.empty((2, 0)), delta=1.0)


class TestCheckDeltaRejectsMalformedInput:
    """`_validation.check_delta`, reached through the public entry point."""

    @pytest.mark.parametrize("bad", [None, "not-a-number", [1.0, 2.0]])
    def test_non_numeric_delta_is_rejected(self, bad):
        with pytest.raises(ValueError, match="must be a real number"):
            discrete_edit_distance(curve([0.0]), curve([0.0]), delta=bad)


class TestVerificationReportTruthiness:
    """`VerificationReport.__bool__`, distinct from reading `.ok` directly."""

    def test_report_is_truthy_exactly_when_it_is_ok(self):
        ref, obs = curve([0.0, 1.0]), curve([0.0, 1.0])
        result = discrete_edit_distance(ref, obs, 0.1, return_witness=True)
        passing = verify_witness(ref, obs, result)
        failing = verify_witness(ref, obs, _tampered(result, cost=99))

        assert bool(passing) is True
        assert bool(failing) is False
        # The idiom the type exists for: `if report:` rather than `if report.ok:`.
        assert passing
        assert not failing


class TestReplayRejectsAMismatchedInsertedPoint:
    """`replay`'s own dimension check, not reachable through the solver

    (the solver only ever proposes centres in the observation's own dimension),
    so this needs a hand-forged `Insertion`.
    """

    def test_inserted_point_of_the_wrong_dimension_is_rejected(self):
        obs = curve([1.0, 2.0])  # 1-D curve
        bad_insertion = Insertion(gap=0, order=0, point=(1.0, 2.0))  # 2-D point
        with pytest.raises(ValueError, match="is not 1-dimensional"):
            replay(obs, (bad_insertion,))


class TestDiscreteFrechetEmptyCurveConvention:
    """`verify.discrete_frechet`'s empty-vs-non-empty convention.

    Not reachable through `ordinary_discrete_frechet`: that wrapper runs
    inputs through `as_curve`, which rejects empty curves before
    `discrete_frechet` ever sees them. `discrete_frechet` is exercised here
    directly, the way `verify_witness` itself uses it.
    """

    def test_empty_against_non_empty_is_infinite(self):
        assert discrete_frechet(np.empty((0, 1)), curve([0.0])) == float("inf")

    def test_both_empty_is_zero(self):
        assert discrete_frechet(np.empty((0, 1)), np.empty((0, 1))) == 0.0


class TestCheckCouplingViaTamperedWitness:
    """`_check_coupling`'s own guard rails, reached by tampering `coupling`."""

    def test_an_empty_coupling_is_rejected(self):
        ref, obs = curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 100.0, 2.0])
        result = discrete_edit_distance(ref, obs, 0.1, operations="delete", return_witness=True)
        bad = _tampered(result, coupling=())
        report = verify_witness(ref, obs, bad)
        assert not report.ok
        assert "coupling is empty" in report.violations

    def test_a_coupling_pair_outside_both_curves_is_rejected(self):
        ref, obs = curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 100.0, 2.0])
        result = discrete_edit_distance(ref, obs, 0.1, operations="delete", return_witness=True)
        first, _middle, last = result.coupling
        bad = _tampered(result, coupling=(first, (999, 999), last))
        report = verify_witness(ref, obs, bad)
        assert any("out of range" in v for v in report.violations), report.violations


class TestReplayFailureIsAReportedViolationNotAnException:
    """`verify_witness` catches `replay`'s own `ValueError` and reports it."""

    def test_an_out_of_range_deletion_in_the_witness_is_reported(self):
        ref, obs = curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 100.0, 2.0])
        result = discrete_edit_distance(ref, obs, 0.1, operations="delete", return_witness=True)
        bad = _tampered(result, edits=(Deletion(99),))
        report = verify_witness(ref, obs, bad)
        assert not report.ok
        assert any(v.startswith("replay failed:") for v in report.violations), report.violations


class TestModeDeletionConsistency:
    """The `mode == "insert"` half of the mode/edit-kind cross-check.

    The `mode == "delete"` half is already covered by
    `test_witness.py::test_mode_violation_is_caught`.
    """

    def test_insert_mode_witness_containing_a_deletion_is_rejected(self):
        ref, obs = curve([0.0, 5.0, 10.0]), curve([0.0, 99.0, 10.0])
        result = discrete_edit_distance(ref, obs, 0.6, operations="both", return_witness=True)
        assert any(isinstance(e, Deletion) for e in result.edits), (
            "fixture must contain a deletion for this check to mean anything"
        )
        bad = _tampered(result, mode="insert")
        report = verify_witness(ref, obs, bad)
        assert any("mode 'insert'" in v for v in report.violations), report.violations


class TestRetainedVertexGuardCatchesAHypotheticalReplayDivergence:
    """FAULT INJECTED: `replay` is stubbed to disagree with the deletion set.

    ``deleted`` and ``replayed`` are both derived from the same
    ``result.edits`` inside `verify_witness`, so they are self-consistent by
    construction: `replay` always places every non-deleted original vertex
    into its output in original order, which makes the retained-vertex check
    a greedy subsequence test that provably succeeds whenever a valid
    embedding exists (it always does here). No tampering of `EditResult`
    fields alone can desynchronise them. This test stubs `replay` itself to
    silently drop a kept vertex - the one way this check could ever fire in
    practice, namely a future bug in `replay` - and confirms `verify_witness`
    would still catch it rather than trusting `replay`'s output blindly.
    """

    def test_a_replay_that_drops_a_kept_vertex_is_caught(self, monkeypatch):
        ref, obs = curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 100.0, 2.0])
        result = discrete_edit_distance(ref, obs, 0.1, operations="delete", return_witness=True)

        def dropped_middle_vertex(observation, edits):
            del observation, edits
            return curve([0.0, 2.0])  # the real replay would also keep 1.0

        monkeypatch.setattr(verify_module, "replay", dropped_middle_vertex)
        report = verify_witness(ref, obs, result)
        assert not report.ok
        assert any(
            "do not appear unchanged and in order" in v for v in report.violations
        ), report.violations


class TestEmptyEditedCurveIsRejected:
    """No solver ever proposes this: an edited curve with zero vertices can

    never be delta-close to a non-empty reference, so it would never be
    reported as an optimal, certified witness. It is reachable only by
    tampering the edit list to delete every original vertex with nothing
    inserted in their place.
    """

    def test_deleting_every_vertex_without_inserting_any_is_rejected(self):
        ref, obs = curve([0.0, 1.0, 2.0]), curve([0.0, 1.0, 100.0, 2.0])
        result = discrete_edit_distance(ref, obs, 0.1, operations="delete", return_witness=True)
        bad = _tampered(result, edits=tuple(Deletion(i) for i in range(len(obs))))
        report = verify_witness(ref, obs, bad)
        assert not report.ok
        assert any(
            "edited curve is empty" in v for v in report.violations
        ), report.violations


def _blank_tables(m: int, n: int) -> DPTables:
    """An all-infeasible, all-unset table of the right shape.

    Every real `DPTables` comes out of `_reference_dp.solve`, whose
    parent-record invariants are exactly what the traceback guards below
    exist to check. To test that those guards actually fire on the states
    they were written for, the tests below poke a single field of an
    otherwise-blank table rather than trying to coax `solve` into producing
    an internally inconsistent result (it cannot; that is the property under
    test elsewhere in the suite).
    """
    return DPTables(
        f=[[INF] * (n + 1) for _ in range(m + 1)],
        k=[[INF] * (n + 1) for _ in range(m + 1)],
        f_from=[[FROM_NONE] * (n + 1) for _ in range(m + 1)],
        k_from=[[FROM_NONE] * (n + 1) for _ in range(m + 1)],
        p_block=[[0] * (n + 1) for _ in range(m + 1)],
        mu=None,
        states=0,
    )


class TestReconstructRefusesAnInfeasibleInstance:
    """`reconstruct` must not be asked to explain an infeasible cost.

    `api.py` never calls it in that case (it checks `math.isinf(cost)` and
    returns an `infeasible` status first), so this calls the private
    `_reference_dp.solve` / `_traceback.reconstruct` pair directly, the same
    way `test_equivalence.py` already imports `_reference_dp.solve`.
    """

    def test_raises_rather_than_reconstructing_a_nonexistent_witness(self):
        reference = curve([0.0, 9.0])
        observation = curve([0.0])
        tables = _reference_dp.solve(reference, observation, 0.1, "delete")
        assert tables.cost == float("inf")
        with pytest.raises(TracebackError, match="infeasible instance"):
            reconstruct(tables, reference, observation, 0.1)


class TestReconstructGuardsAgainstCorruptedTables:
    """Each test corrupts exactly one parent-record entry of a blank table.

    These are internal-consistency assertions: a table `solve` actually
    produces can never take these branches, because the parent records it
    writes are always mutually consistent with each other by construction.
    They earn a test anyway, on the same theory as the safety nets in
    `test_numeric_boundary.py`: an unreachable guard that has never been
    exercised is not verified to guard anything.
    """

    def test_an_unreachable_layer_code_is_rejected(self):
        tables = _blank_tables(1, 1)
        tables.f[1][1] = 0.0  # finite, so the infeasibility check passes first
        tables.f_from[1][1] = FROM_NONE
        with pytest.raises(TracebackError, match="unreachable state"):
            reconstruct(tables, curve([0.0]), curve([0.0]), 1.0)

    def test_a_deletion_step_with_no_column_left_is_rejected(self):
        tables = _blank_tables(1, 0)
        tables.f[1][0] = 0.0
        tables.f_from[1][0] = FROM_X
        with pytest.raises(TracebackError, match="deletion branch"):
            reconstruct(tables, curve([0.0]), np.empty((0, 1)), 1.0)

    def test_a_keep_step_with_no_row_or_column_left_is_rejected(self):
        tables = _blank_tables(0, 1)
        tables.f[0][1] = 0.0
        tables.f_from[0][1] = FROM_K
        with pytest.raises(TracebackError, match="keep branch"):
            reconstruct(tables, np.empty((0, 1)), curve([0.0]), 1.0)

    def test_a_keep_step_with_no_recorded_source_is_rejected(self):
        tables = _blank_tables(1, 1)
        tables.f[1][1] = 0.0
        tables.f_from[1][1] = FROM_K
        tables.k_from[1][1] = FROM_NONE  # neither vertical, diagonal nor horizontal
        with pytest.raises(TracebackError, match="no recorded step"):
            reconstruct(tables, curve([0.0]), curve([0.0]), 1.0)

    def test_an_insertion_block_outside_its_valid_range_is_rejected(self):
        tables = _blank_tables(1, 1)
        tables.f[1][1] = 0.0
        tables.f_from[1][1] = FROM_P
        tables.p_block[1][1] = 0  # must satisfy 1 <= block_lo <= i, here i == 1
        with pytest.raises(TracebackError, match="has block 0"):
            reconstruct(tables, curve([0.0]), curve([0.0]), 1.0)


class TestReconstructReturnsNoneWhenNoCentreIsCertifiable:
    """FAULT INJECTED: `certified_block_centre` is stubbed to always abstain.

    An organic fixture for "the ball predicate says a block is insertable,
    but no float64 point near its exact centre can be certified" is exactly
    the search `test_numeric_boundary.py` had to do to find `ULP_BLOCK`, and
    reusing that machinery here would test the same thing twice. Instead
    this drives a real, otherwise-ordinary insertion witness through the
    branch by making `certified_block_centre` itself refuse, and checks the
    contract documented on `reconstruct`: `None` means "unavailable", not an
    exception and not a wrong witness.
    """

    def test_reconstruct_returns_none_instead_of_guessing(self, monkeypatch):
        reference = curve([0.0, 5.0, 10.0])
        observation = curve([0.0, 10.0])
        delta = 1.0
        tables = _reference_dp.solve(reference, observation, delta, "insert")
        assert tables.cost == 1.0  # one insertion is needed near 5.0

        monkeypatch.setattr(_traceback, "certified_block_centre", lambda block, delta: None)
        witness = reconstruct(tables, reference, observation, delta)
        assert witness is None


class TestApiHandlesReconstructOutcomes:
    """`api.discrete_edit_distance`'s handling of what `reconstruct` returns.

    Both scenarios need `reconstruct` to fail in a way that only the exact
    per-point centre certification triggers, distinct from the coarser
    ball-radius predicate `solve` itself uses - the same kind of gap
    `test_numeric_boundary.py` documents for `certified_block_centre`. Both
    tests reuse the fault-injection approach above rather than searching for
    a second organic fixture.
    """

    def test_ambiguity_found_while_reconstructing_keeps_the_cost_and_flags_the_witness(
        self, monkeypatch
    ):
        reference = curve([0.0, 5.0, 10.0])
        observation = curve([0.0, 10.0])

        def explode(block, delta):
            del block, delta
            raise NumericallyAmbiguous("forced for test")

        monkeypatch.setattr(_traceback, "certified_block_centre", explode)
        result = discrete_edit_distance(
            reference, observation, 1.0, operations="insert", return_witness=True
        )
        assert result.status == "optimal"
        assert result.cost == 1
        assert result.witness_status == "unavailable"
        assert result.detail is not None and "witness abandoned" in result.detail

    def test_an_uncertifiable_centre_marks_the_witness_unavailable_not_infeasible(
        self, monkeypatch
    ):
        reference = curve([0.0, 5.0, 10.0])
        observation = curve([0.0, 10.0])

        monkeypatch.setattr(_traceback, "certified_block_centre", lambda block, delta: None)
        result = discrete_edit_distance(
            reference, observation, 1.0, operations="insert", return_witness=True
        )
        assert result.status == "optimal"
        assert result.cost == 1
        assert result.witness_status == "unavailable"
        assert result.detail is not None and "delta was not adjusted" in result.detail
