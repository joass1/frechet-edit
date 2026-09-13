"""Optimality evidence: the production API against genuinely independent oracles.

This is the gate that makes the package's central claim - MINIMUM edit count -
mean something. Witness feasibility never proves minimality; only agreement
with a search that shares no recurrence does.

The oracles live in ``tests/oracles`` and were written from
``docs/definition.md`` and ``docs/oracle-protocol.md`` alone, without sight of
``src/frechet_edit``. See ``docs/oracle-protocol.md`` for their completeness
arguments and their budget rules.

A budget exhaustion is BLOCKED evidence, not agreement: ``OracleBudgetExceeded``
is allowed to propagate and fail the test rather than being turned into a skip
that reads like a pass.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from frechet_edit import discrete_edit_distance, verify_witness
from tests.oracles.block_edit_search import edit_oracle
from tests.oracles.subsequence import deletion_oracle

ALPHABET_1D = (0.0, 1.0, 2.5)
BACKENDS = ("python", "reference")


def _curves_1d(max_len, alphabet=ALPHABET_1D):
    for length in range(1, max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield np.array(combo, dtype=np.float64).reshape(-1, 1)


def _production(ref, obs, delta, mode, backend="python"):
    return discrete_edit_distance(ref, obs, delta, operations=mode, backend=backend)


def _as_list(curve):
    return [tuple(float(x) for x in point) for point in curve]


class TestDeletionOracle:
    """Oracle A: exhaustive subsequence enumeration."""

    @pytest.mark.parametrize("delta", [0.4, 1.0, 2.0])
    def test_exhaustive_1d_families(self, delta):
        compared = 0
        for ref in _curves_1d(3):
            for obs in _curves_1d(3):
                expected = deletion_oracle(_as_list(ref), _as_list(obs), delta)
                result = _production(ref, obs, delta, "delete")
                got = math.inf if result.status == "infeasible" else float(result.cost)
                assert result.status in ("optimal", "infeasible")
                assert got == expected, (
                    f"delta={delta} ref={_as_list(ref)} obs={_as_list(obs)}: "
                    f"production={got} oracle={expected}"
                )
                compared += 1
        assert compared == 39 * 39, compared

    def test_random_2d_cases(self):
        rng = np.random.default_rng(101)
        compared = 0
        for _ in range(150):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 5)), 2)) * 2, 1)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 7)), 2)) * 2, 1)
            delta = float(rng.choice([0.5, 1.0, 2.0, 3.0]))
            expected = deletion_oracle(_as_list(ref), _as_list(obs), delta)
            result = _production(ref, obs, delta, "delete")
            got = math.inf if result.status == "infeasible" else float(result.cost)
            assert got == expected, (_as_list(ref), _as_list(obs), delta, got, expected)
            compared += 1
        assert compared == 150

    def test_both_backends_agree_with_the_oracle(self):
        rng = np.random.default_rng(202)
        for _ in range(60):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 4)), 1)) * 2, 1)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 6)), 1)) * 2, 1)
            delta = float(rng.uniform(0.3, 2.5))
            expected = deletion_oracle(_as_list(ref), _as_list(obs), delta)
            for backend in BACKENDS:
                result = _production(ref, obs, delta, "delete", backend)
                got = math.inf if result.status == "infeasible" else float(result.cost)
                assert got == expected, (backend, got, expected)


class TestEditOracle:
    """Oracle B: finite block-centre candidates plus bounded enumeration."""

    @pytest.mark.parametrize("mode", ["insert", "both"])
    @pytest.mark.parametrize("delta", [1.0, 2.0])
    def test_exhaustive_tiny_families(self, mode, delta):
        compared = 0
        for ref in _curves_1d(3):
            for obs in _curves_1d(2):
                expected = edit_oracle(_as_list(ref), _as_list(obs), delta, mode)
                result = _production(ref, obs, delta, mode)
                got = math.inf if result.status == "infeasible" else float(result.cost)
                assert got == expected, (
                    f"mode={mode} delta={delta} ref={_as_list(ref)} "
                    f"obs={_as_list(obs)}: production={got} oracle={expected}"
                )
                compared += 1
        assert compared == 39 * 12, compared

    @pytest.mark.parametrize("mode", ["delete", "insert", "both"])
    def test_random_1d_cases(self, mode):
        rng = np.random.default_rng(303)
        for _ in range(40):
            ref = np.round(rng.integers(0, 6, size=(int(rng.integers(1, 4)), 1)), 1) * 1.0
            obs = np.round(rng.integers(0, 6, size=(int(rng.integers(1, 3)), 1)), 1) * 1.0
            delta = float(rng.choice([0.5, 1.0, 1.5, 2.0]))
            expected = edit_oracle(_as_list(ref), _as_list(obs), delta, mode)
            result = _production(ref, obs, delta, mode)
            got = math.inf if result.status == "infeasible" else float(result.cost)
            assert got == expected, (mode, _as_list(ref), _as_list(obs), delta, got, expected)

    def test_random_2d_cases(self):
        rng = np.random.default_rng(404)
        for _ in range(40):
            ref = np.round(rng.integers(0, 4, size=(int(rng.integers(1, 4)), 2)), 1) * 1.0
            obs = np.round(rng.integers(0, 4, size=(int(rng.integers(1, 3)), 2)), 1) * 1.0
            delta = float(rng.choice([1.0, 1.5, 2.5]))
            expected = edit_oracle(_as_list(ref), _as_list(obs), delta, "both")
            result = _production(ref, obs, delta, "both")
            got = math.inf if result.status == "infeasible" else float(result.cost)
            assert got == expected, (_as_list(ref), _as_list(obs), delta, got, expected)

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["insert", "both"])
    def test_larger_exhaustive_family(self, mode):
        """The m, n <= 4 tier from docs/oracle-protocol.md."""
        compared = 0
        delta = 1.5
        for ref in _curves_1d(4, alphabet=(0.0, 2.0)):
            for obs in _curves_1d(3, alphabet=(0.0, 2.0, 5.0)):
                expected = edit_oracle(_as_list(ref), _as_list(obs), delta, mode)
                result = _production(ref, obs, delta, mode)
                got = math.inf if result.status == "infeasible" else float(result.cost)
                assert got == expected, (mode, _as_list(ref), _as_list(obs), got, expected)
                compared += 1
        assert compared == 30 * 39, compared


class TestWitnessesAgreeWithOptimalCosts:
    """Every certified witness must both verify AND match the oracle's optimum."""

    @pytest.mark.parametrize("mode", ["delete", "insert", "both"])
    def test_witness_cost_equals_oracle_optimum(self, mode):
        rng = np.random.default_rng(505)
        certified = 0
        for _ in range(50):
            ref = np.round(rng.integers(0, 5, size=(int(rng.integers(1, 4)), 1)), 1) * 1.0
            obs = np.round(rng.integers(0, 5, size=(int(rng.integers(1, 3)), 1)), 1) * 1.0
            delta = float(rng.choice([0.5, 1.0, 2.0]))
            result = discrete_edit_distance(
                ref, obs, delta, operations=mode, return_witness=True
            )
            expected = edit_oracle(_as_list(ref), _as_list(obs), delta, mode)
            if result.status == "infeasible":
                assert math.isinf(expected)
                continue
            assert float(result.cost) == expected
            if result.witness_status == "certified":
                verify_witness(ref, obs, result).raise_for_status()
                assert len(result.edits) == result.cost == expected
                certified += 1
        assert certified >= 20, f"only {certified} certified witnesses exercised"


class TestOracleBudgetIsNotSilentlyAbsorbed:
    def test_budget_exhaustion_propagates(self):
        """An incomplete search must fail loudly, never pass as agreement."""
        from tests.oracles.block_edit_search import OracleBudgetExceeded

        with pytest.raises(OracleBudgetExceeded):
            edit_oracle(
                [(0.0,), (5.0,), (0.0,)],
                [(9.0,), (9.0,)],
                1.0,
                "both",
                max_states=3,
            )
