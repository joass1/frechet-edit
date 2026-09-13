"""Layered K/P/X form vs the collapsed single-table form.

docs/recurrences.md section 3 states the corrected relationship:

* deletion-only: the two forms agree, and that is checked exhaustively here;
* insertion / mixed: the collapsed form is UNSOUND, and the counterexample is
  pinned here so the finding cannot be lost.

The collapsed recurrence is transcribed independently in this file so that a
bug in ``_reference_dp`` cannot hide by being compared against itself.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from frechet_edit._geometry import mu_indices
from frechet_edit._numerics import within
from frechet_edit._reference_dp import solve
from tests.oracles.block_edit_search import edit_oracle
from tests.oracles.subsequence import deletion_oracle

INF = math.inf


def collapsed(reference, observation, delta, mode):
    """The collapsed single-table recurrence of docs/recurrences.md section 3.

    The keep branch uses the UNRESTRICTED ``F(i-1, j)`` as its vertical
    predecessor. That is the substitution section 3.2 shows to be unsound once
    insertions are allowed.
    """
    m, n = len(reference), len(observation)
    allow_delete = mode in ("delete", "both")
    allow_insert = mode in ("insert", "both")
    mu = mu_indices(reference, delta) if allow_insert else None

    f = [[INF] * (n + 1) for _ in range(m + 1)]
    f[0][0] = 0.0
    for j in range(n + 1):
        for i in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            if allow_delete and j >= 1 and f[i][j - 1] != INF:
                best = min(best, f[i][j - 1] + 1.0)
            if i >= 1 and j >= 1 and within(observation[j - 1], reference[i - 1], delta):
                best = min(best, f[i - 1][j - 1], f[i - 1][j], f[i][j - 1])
            if allow_insert and i >= 1:
                lo = int(mu[i - 1]) + 1
                window = min((f[k - 1][j] for k in range(lo, i + 1)), default=INF)
                if window != INF:
                    best = min(best, window + 1.0)
            f[i][j] = best
    return f[m][n]


def layered(reference, observation, delta, mode):
    return solve(reference, observation, delta, mode).cost


ALPHABET = (0.0, 1.0, 2.5)


def _curves(max_len):
    for length in range(1, max_len + 1):
        for combo in itertools.product(ALPHABET, repeat=length):
            yield np.array(combo, dtype=np.float64).reshape(-1, 1)


def _pts(curve):
    return [tuple(float(x) for x in p) for p in curve]


class TestDeletionModeAgreement:
    """Section 3.1: with no insertions the two forms are equivalent."""

    @pytest.mark.parametrize("delta", [0.4, 1.0, 2.0])
    def test_deletion_mode_forms_agree(self, delta):
        compared = 0
        for ref in _curves(3):
            for obs in _curves(2):
                a = layered(ref, obs, delta, "delete")
                b = collapsed(ref, obs, delta, "delete")
                assert a == b, (delta, ref.ravel().tolist(), obs.ravel().tolist(), a, b)
                compared += 1
        assert compared == 39 * 12

    def test_deletion_mode_random_2d(self):
        rng = np.random.default_rng(83)
        for _ in range(120):
            ref = np.round(rng.normal(size=(int(rng.integers(1, 6)), 2)) * 2, 2)
            obs = np.round(rng.normal(size=(int(rng.integers(1, 6)), 2)) * 2, 2)
            delta = float(rng.uniform(0.3, 3.0))
            assert layered(ref, obs, delta, "delete") == collapsed(ref, obs, delta, "delete")


class TestCollapsedFormIsUnsoundWithInsertions:
    """Section 3.2. The layered form is required, not merely tidier."""

    def test_minimal_counterexample(self):
        ref = np.array([[0.0], [1.0], [0.0]])
        obs = np.array([[0.0]])
        delta = 0.4

        assert layered(ref, obs, delta, "insert") == 2.0
        assert collapsed(ref, obs, delta, "insert") == 1.0
        # The independent oracle sides with the layered form.
        assert edit_oracle(_pts(ref), _pts(obs), delta, "insert") == 2.0

    def test_why_one_insertion_cannot_work(self):
        """One inserted point cannot cover both pi_2 = 1 and pi_3 = 0 at delta 0.4."""
        from frechet_edit._geometry import meb_radius_le

        assert not meb_radius_le(np.array([[1.0], [0.0]]), 0.4)
        assert meb_radius_le(np.array([[1.0], [0.0]]), 0.5)

    @pytest.mark.parametrize("mode", ["insert", "both"])
    def test_collapsed_never_exceeds_layered_and_sometimes_undercuts_it(self, mode):
        """Characterise the failure: the collapsed form is too cheap, never too dear."""
        undercuts = 0
        for delta in (0.4, 1.0, 2.0):
            for ref in _curves(3):
                for obs in _curves(2):
                    a = layered(ref, obs, delta, mode)
                    b = collapsed(ref, obs, delta, mode)
                    assert b <= a, (mode, delta, a, b)
                    undercuts += b < a
        assert undercuts > 0, "the counterexample family stopped separating the two forms"

    def test_oracle_sides_with_layered_on_every_disagreement(self):
        checked = 0
        for mode in ("insert", "both"):
            for delta in (0.4, 1.0):
                for ref in _curves(3):
                    for obs in _curves(2):
                        a = layered(ref, obs, delta, mode)
                        b = collapsed(ref, obs, delta, mode)
                        if a == b:
                            continue
                        expected = edit_oracle(_pts(ref), _pts(obs), delta, mode)
                        assert a == expected, (mode, delta, a, b, expected)
                        assert b != expected
                        checked += 1
        assert checked >= 10, f"only {checked} disagreements exercised"


class TestModelsAreNotVacuous:
    def test_the_collapsed_model_is_not_vacuous(self):
        """A deliberately broken variant must disagree with both forms.

        Without this guard the agreement tests could pass for the wrong reason.
        """

        def broken(reference, observation, delta, mode):
            """Insertion restricted to single reference vertices (mu(i) == i)."""
            m, n = len(reference), len(observation)
            f = [[INF] * (n + 1) for _ in range(m + 1)]
            f[0][0] = 0.0
            for j in range(n + 1):
                for i in range(m + 1):
                    if i == 0 and j == 0:
                        continue
                    best = INF
                    if mode in ("delete", "both") and j >= 1 and f[i][j - 1] != INF:
                        best = min(best, f[i][j - 1] + 1.0)
                    if i >= 1 and j >= 1 and within(observation[j - 1], reference[i - 1], delta):
                        best = min(best, f[i - 1][j - 1], f[i - 1][j], f[i][j - 1])
                    if mode in ("insert", "both") and i >= 1 and f[i - 1][j] != INF:
                        best = min(best, f[i - 1][j] + 1.0)
                    f[i][j] = best
            return f[m][n]

        ref = np.array([[0.0], [2.0], [4.0]])
        obs = np.array([[0.0]])
        assert layered(ref, obs, 1.0, "insert") == 1
        assert broken(ref, obs, 1.0, "insert") == 2
        assert edit_oracle(_pts(ref), _pts(obs), 1.0, "insert") == 1

    def test_deletion_oracle_confirms_the_agreeing_deletion_forms(self):
        """Both forms agreeing is only reassuring if they agree with the oracle."""
        for delta in (0.4, 1.0):
            for ref in _curves(2):
                for obs in _curves(2):
                    expected = deletion_oracle(_pts(ref), _pts(obs), delta)
                    assert layered(ref, obs, delta, "delete") == expected
                    assert collapsed(ref, obs, delta, "delete") == expected
