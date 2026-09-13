"""Differential test of the PUBLISHED recurrence against the definition.

This is the executable evidence behind `docs/errata-insertion-recurrence.md`.
It pits three implementations that share no recurrence against each other:

1. `tests/oracles/published_recurrence.py` - the paper's displayed recurrences,
   transcribed verbatim from the PDF text.
2. `tests/oracles/coupling_brute.py` - a definitional brute force with no
   dynamic program at all, enumerating edited curves and monotone couplings.
3. `frechet_edit.discrete_edit_distance` - this package.

The expected outcome is asymmetric and is asserted as such: the published
DELETION recurrence agrees with the definition everywhere, the published
INSERTION and MIXED recurrences do not, and this package agrees everywhere.
A test that merely said "they differ" would pass if the package were also
broken, so each claim is pinned separately.
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from frechet_edit import discrete_edit_distance
from tests.oracles.coupling_brute import brute_edit_distance
from tests.oracles.published_recurrence import (
    paper_deletion_dp,
    paper_insertion_dp,
    paper_mixed_dp,
)

ALPHABET = (0.0, 1.0, 2.5)

# The minimal witness from docs/errata-insertion-recurrence.md section 5.
WITNESS_PI = [(0.0,), (1.0,), (0.0,)]
WITNESS_SIGMA = [(0.0,)]
WITNESS_DELTA = 0.4


def _curves(max_len, alphabet=ALPHABET):
    for length in range(1, max_len + 1):
        for combo in itertools.product(alphabet, repeat=length):
            yield [(x,) for x in combo]


def _package(pi, sigma, delta, mode):
    result = discrete_edit_distance(
        np.array(pi, dtype=float),
        np.array(sigma, dtype=float),
        delta,
        operations=mode,
    )
    return math.inf if result.status == "infeasible" else float(result.cost)


class TestTheMinimalWitness:
    """The single instance the erratum is built on, pinned exactly."""

    def test_published_recurrence_under_reports(self):
        assert paper_insertion_dp(WITNESS_PI, WITNESS_SIGMA, WITNESS_DELTA) == 1

    def test_definition_says_two(self):
        truth = brute_edit_distance(
            WITNESS_PI, WITNESS_SIGMA, WITNESS_DELTA, "insert"
        )
        assert truth == 2

    def test_package_says_two(self):
        assert _package(WITNESS_PI, WITNESS_SIGMA, WITNESS_DELTA, "insert") == 2

    def test_mixed_mode_inherits_the_same_defect(self):
        assert paper_mixed_dp(WITNESS_PI, WITNESS_SIGMA, WITNESS_DELTA) == 1
        assert brute_edit_distance(
            WITNESS_PI, WITNESS_SIGMA, WITNESS_DELTA, "both"
        ) == 2


class TestPublishedDeletionRecurrenceIsSound:
    """Section 5.1 has no insertion branch, so it has nothing to get wrong."""

    @pytest.mark.parametrize("delta", [0.4, 1.0, 2.0])
    def test_agrees_with_the_definition_exhaustively(self, delta):
        compared = 0
        for pi in _curves(3):
            for sigma in _curves(3):
                truth = brute_edit_distance(pi, sigma, delta, "delete")
                assert paper_deletion_dp(pi, sigma, delta) == truth, (pi, sigma)
                compared += 1
        assert compared == 39 * 39


class TestPublishedInsertionRecurrenceIsUnsound:
    """Section 5.2 and 5.3. Failures are counted, not merely detected."""

    @pytest.mark.parametrize(
        "mode,paper_fn",
        [("insert", paper_insertion_dp), ("both", paper_mixed_dp)],
    )
    def test_disagrees_and_always_downward(self, mode, paper_fn):
        delta = 0.4
        failures = compared = 0
        for pi in _curves(3):
            for sigma in _curves(2):
                truth = brute_edit_distance(pi, sigma, delta, mode)
                published = paper_fn(pi, sigma, delta)
                compared += 1
                if published != truth:
                    failures += 1
                    # Every deviation must be an under-estimate: the min ranges
                    # over too permissive a predecessor set, so it can only go
                    # down. An over-estimate would mean a different bug.
                    assert published < truth, (pi, sigma, published, truth)
        assert compared == 39 * 12
        assert failures > 0, "the erratum claims this recurrence fails here"


class TestThisPackageAgreesWithTheDefinition:
    """The package must be right everywhere the published form is wrong."""

    @pytest.mark.parametrize("mode", ["delete", "insert", "both"])
    @pytest.mark.parametrize("delta", [0.4, 1.0, 2.0])
    def test_exhaustive_1d(self, mode, delta):
        for pi in _curves(3):
            for sigma in _curves(2):
                truth = brute_edit_distance(pi, sigma, delta, mode)
                assert _package(pi, sigma, delta, mode) == truth, (pi, sigma)

    @pytest.mark.parametrize("mode", ["insert", "both"])
    def test_random_2d(self, mode):
        rng = random.Random(7)
        for _ in range(60):
            pi = [
                (float(rng.randint(0, 3)), float(rng.randint(0, 3)))
                for _ in range(rng.randint(1, 3))
            ]
            sigma = [
                (float(rng.randint(0, 3)), float(rng.randint(0, 3)))
                for _ in range(rng.randint(1, 2))
            ]
            delta = rng.choice([0.5, 1.0, 1.5, 2.0])
            truth = brute_edit_distance(pi, sigma, delta, mode)
            assert _package(pi, sigma, delta, mode) == truth, (pi, sigma, delta)


@pytest.mark.slow
class TestWiderSweep:
    """The m <= 4 tier quoted in the erratum's verification table."""

    @pytest.mark.parametrize(
        "mode,paper_fn",
        [("insert", paper_insertion_dp), ("both", paper_mixed_dp)],
    )
    @pytest.mark.parametrize("delta", [0.4, 1.0, 2.0])
    def test_counts_match_the_documented_table(self, mode, paper_fn, delta):
        failures = compared = 0
        for pi in _curves(4):
            for sigma in _curves(2):
                truth = brute_edit_distance(pi, sigma, delta, mode)
                assert _package(pi, sigma, delta, mode) == truth, (pi, sigma)
                if paper_fn(pi, sigma, delta) != truth:
                    failures += 1
                compared += 1
        assert compared == 1440
        # delta = 2.0 is the documented no-failure case: mu(i) collapses and
        # the insertion branch stops competing with the keep branches.
        if delta == 2.0:
            assert failures == 0
        else:
            assert failures > 0


class TestTheCauseIsExactlyOneTerm:
    """Changing only the keep branch's vertical predecessor must fix it.

    This is what separates a diagnosis from a symptom. If some other part of the
    published recurrence were also wrong, restricting this single term would not
    be enough and these tests would fail.
    """

    def test_the_minimal_witness_is_repaired(self):
        from tests.oracles.published_recurrence import repaired_insertion_dp

        assert repaired_insertion_dp(WITNESS_PI, WITNESS_SIGMA, WITNESS_DELTA) == 2

    @pytest.mark.parametrize("delta", [0.4, 1.0, 2.0])
    def test_one_term_removes_every_failure(self, delta):
        from tests.oracles.published_recurrence import repaired_insertion_dp

        compared = 0
        for pi in _curves(3):
            for sigma in _curves(2):
                truth = brute_edit_distance(pi, sigma, delta, "insert")
                assert repaired_insertion_dp(pi, sigma, delta) == truth, (pi, sigma)
                compared += 1
        assert compared == 39 * 12
