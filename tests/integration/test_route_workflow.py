"""End-to-end workflow: gallery, corruption, retrieval, repair, verification.

This exercises the path an analyst actually takes, through the public API only,
and asserts the properties the application layer depends on. It uses synthetic
routes: no real trajectory data exists in this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments import corruptions, evaluate, synthetic
from experiments.report import load, render

from frechet_edit import (
    Deletion,
    discrete_edit_distance,
    ordinary_discrete_frechet,
    verify_witness,
)

DELTA = 25.0


@pytest.fixture
def gallery():
    return synthetic.make_gallery(np.random.default_rng(101), n_families=2, n_points=25)


class TestRepairWorkflow:
    def test_spiked_query_is_repaired_and_verified(self, gallery):
        truth = gallery[0]
        rng = np.random.default_rng(7)
        query, record = corruptions.spike(truth.points, rng, 2, 200.0)

        result = discrete_edit_distance(
            truth.points, query, DELTA, operations="both", return_witness=True
        )
        assert result.status == "optimal"
        assert result.witness_status == "certified"
        verify_witness(truth.points, query, result).raise_for_status()

        # The repair must actually bring the curve inside delta.
        residual = ordinary_discrete_frechet(truth.points, result.edited_curve)
        assert residual <= DELTA

        # And the raw trace must NOT have been inside delta to begin with,
        # otherwise the fixture proves nothing.
        assert ordinary_discrete_frechet(truth.points, query) > DELTA
        assert record.n_injected == 2

    def test_repair_cost_is_small_relative_to_curve_length(self, gallery):
        truth = gallery[0]
        rng = np.random.default_rng(11)
        query, _ = corruptions.spike(truth.points, rng, 2, 200.0)
        result = discrete_edit_distance(truth.points, query, DELTA, operations="both")
        assert result.cost <= 6, "two isolated spikes should not need a rebuild"

    def test_edits_localise_isolated_spikes(self, gallery):
        """With well-separated large spikes the deletions land on them.

        This is asserted only for a deliberately clean fixture. In general the
        optimal edit count is NOT the injected count and the located indices
        need not coincide; see docs/limitations.md.
        """
        truth = gallery[0]
        rng = np.random.default_rng(13)
        query, record = corruptions.spike(truth.points, rng, 2, 400.0)
        result = discrete_edit_distance(
            truth.points, query, DELTA, operations="both", return_witness=True
        )
        deleted = {e.index for e in result.edits if isinstance(e, Deletion)}
        assert set(record.affected_original_indices) <= deleted


class TestRetrievalWorkflow:
    def test_correct_template_outranks_hard_negatives(self, gallery):
        truth = gallery[0]
        rng = np.random.default_rng(17)
        query, _ = corruptions.spike(truth.points, rng, 2, 200.0)

        scores = {
            route.route_id: discrete_edit_distance(
                route.points, query, DELTA, operations="both"
            ).cost
            for route in gallery
        }
        best = min(scores.values())
        winners = [rid for rid, cost in scores.items() if cost == best]
        assert truth.route_id in winners, scores

    def test_raw_frechet_is_the_measure_that_breaks(self, gallery):
        """The mechanism claim: the spike ruins ordinary Frechet, not FED."""
        truth = gallery[0]
        rng = np.random.default_rng(19)
        query, _ = corruptions.spike(truth.points, rng, 1, 500.0)

        assert ordinary_discrete_frechet(truth.points, query) > 400.0
        assert discrete_edit_distance(truth.points, query, DELTA, operations="both").cost <= 3


class TestFullRunProducesAuditableEvidence:
    @pytest.mark.slow
    def test_run_writes_per_query_records_and_a_report(self, tmp_path: Path):
        cfg = {
            "seed": 3,
            "n_families": 2,
            "n_points": 16,
            "queries_per_route": 1,
            "n_spikes": 1,
            "spike_magnitude": 180.0,
            "jitter_sigma": 1.0,
            "n_open_set_queries": 1,
            "delta": DELTA,
            "eps": DELTA,
            "ks": [1, 3],
            "n_validation_families": 1,
        }
        payload = evaluate.run(cfg, tmp_path)

        assert payload["evidence_level"].startswith("A")
        per_query = json.loads((tmp_path / "per_query.json").read_text(encoding="utf-8"))
        assert per_query, "per-query evidence must be persisted, not only aggregates"
        for record in per_query:
            assert record.get("scores"), record["query_id"]

        text = render(*load(tmp_path))
        assert "EVIDENCE LEVEL A" in text
        assert "not evidence about GPS traces" in text
        # Losses must be visible, not filtered out.
        assert "`frechet`" in text
