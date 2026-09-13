"""Tests for the experiment harness itself: reproducibility and fail-closed reporting.

The metric arithmetic is tested separately and FED-free in ``test_metrics.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import corruptions, evaluate, synthetic
from experiments.report import IncompleteRun, load, render


class TestCorruptionReproducibility:
    def test_same_seed_gives_identical_output(self):
        base = synthetic.make_route(np.random.default_rng(1), 20)
        a, rec_a = corruptions.spike(base, np.random.default_rng(7), 2, 50.0)
        b, rec_b = corruptions.spike(base, np.random.default_rng(7), 2, 50.0)
        assert np.array_equal(a, b)
        assert rec_a.affected_original_indices == rec_b.affected_original_indices

    def test_different_seed_gives_different_output(self):
        base = synthetic.make_route(np.random.default_rng(1), 20)
        a, _ = corruptions.spike(base, np.random.default_rng(7), 2, 50.0)
        b, _ = corruptions.spike(base, np.random.default_rng(8), 2, 50.0)
        assert not np.array_equal(a, b)

    def test_ground_truth_names_real_indices(self):
        base = synthetic.make_route(np.random.default_rng(2), 25)
        corrupted, record = corruptions.spike(base, np.random.default_rng(3), 3, 80.0)
        assert len(record.affected_original_indices) == 3
        for idx in record.affected_original_indices:
            assert 0 <= idx < len(base)
            # The named vertex really did move.
            assert not np.allclose(base[idx], corrupted[idx])
        untouched = set(range(len(base))) - set(record.affected_original_indices)
        for idx in untouched:
            assert np.allclose(base[idx], corrupted[idx])

    def test_injected_count_is_not_claimed_to_be_the_edit_cost(self):
        """The record must document that n_injected is not an optimal edit count."""
        base = synthetic.make_route(np.random.default_rng(4), 15)
        _, record = corruptions.spike(base, np.random.default_rng(5), 2, 60.0)
        assert record.n_injected == 2
        assert "NOT the optimal edit count" in record.as_dict()["note"]


class TestGallery:
    def test_gallery_contains_hard_negatives_per_family(self):
        routes = synthetic.make_gallery(np.random.default_rng(11), n_families=3, n_points=20)
        kinds = {r.kind for r in routes}
        assert {"base", "corridor", "shared_endpoint", "partial_overlap", "detour"} <= kinds
        families = {r.family for r in routes}
        assert len(families) == 3
        for family in families:
            assert sum(1 for r in routes if r.family == family) == 5

    def test_corridor_twin_is_close_but_not_identical(self):
        base = synthetic.make_route(np.random.default_rng(13), 30)
        twin = synthetic.corridor_twin(base, 25.0)
        gaps = np.linalg.norm(twin - base, axis=1)
        assert np.allclose(gaps, 25.0, atol=1e-9)

    def test_shared_endpoint_route_shares_its_endpoints(self):
        base = synthetic.make_route(np.random.default_rng(17), 30)
        other = synthetic.shared_endpoint_route(np.random.default_rng(19), base, 100.0)
        assert np.allclose(other[0], base[0])
        assert np.allclose(other[-1], base[-1])

    def test_detour_route_is_longer_than_its_base(self):
        base = synthetic.make_route(np.random.default_rng(23), 30)
        detour = synthetic.detour_route(np.random.default_rng(29), base, 90.0, length=3)
        assert len(detour) == len(base) + 3


class TestRunnerEndToEnd:
    @pytest.mark.slow
    def test_small_run_produces_complete_evidence(self, tmp_path: Path):
        cfg = {
            "seed": 5,
            "n_families": 2,
            "n_points": 14,
            "queries_per_route": 1,
            "n_spikes": 1,
            "spike_magnitude": 150.0,
            "jitter_sigma": 1.0,
            "n_open_set_queries": 1,
            "delta": 25.0,
            "eps": 25.0,
            "ks": [1, 2],
            "n_validation_families": 1,
            "threshold_quantile": 0.95,
        }
        payload = evaluate.run(cfg, tmp_path)
        assert payload["evidence_level"].startswith("A")
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "per_query.json").exists()

        summary, per_query = load(tmp_path)
        text = render(summary, per_query)
        assert "EVIDENCE LEVEL A" in text
        # Every configured method must appear, wins and losses alike.
        for method in ("fed_delete", "fed_insert", "fed_both", "frechet", "edr", "dtw"):
            assert f"`{method}`" in text

    def test_abstaining_method_is_reported_not_hidden(self, tmp_path: Path):
        """Insert-only cannot remove spikes, so it must abstain and say so."""
        cfg = {
            "seed": 5, "n_families": 1, "n_points": 12, "queries_per_route": 1,
            "n_spikes": 1, "spike_magnitude": 200.0, "jitter_sigma": 0.5,
            "n_open_set_queries": 0, "delta": 20.0, "eps": 20.0, "ks": [1],
            "n_validation_families": 0,
        }
        payload = evaluate.run(cfg, tmp_path)
        insert = payload["aggregates"]["fed_insert"]
        assert insert["n_abstentions"] == insert["n_queries"]
        assert insert["coverage"] == 0.0
        text = render(*load(tmp_path))
        assert "TOTAL ABSTENTION" in text


class TestReportFailsClosed:
    def test_missing_summary_file_fails(self, tmp_path: Path):
        with pytest.raises(IncompleteRun, match="missing mandatory file"):
            load(tmp_path)

    def test_missing_mandatory_field_fails(self, tmp_path: Path):
        (tmp_path / "summary.json").write_text(
            json.dumps({"config": {}, "environment": {}, "aggregates": {"m": {}}}),
            encoding="utf-8",
        )
        (tmp_path / "per_query.json").write_text(json.dumps([{"query_id": "q"}]), encoding="utf-8")
        with pytest.raises(IncompleteRun, match="missing mandatory fields"):
            load(tmp_path)

    def test_missing_aggregate_field_fails(self, tmp_path: Path):
        (tmp_path / "summary.json").write_text(
            json.dumps(
                {
                    "evidence_level": "A",
                    "config": {},
                    "environment": {},
                    "gallery": [],
                    "aggregates": {"m": {"n_queries": 1}},
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "per_query.json").write_text(json.dumps([{"query_id": "q"}]), encoding="utf-8")
        with pytest.raises(IncompleteRun, match="is missing fields"):
            load(tmp_path)

    def test_empty_per_query_evidence_fails(self, tmp_path: Path):
        (tmp_path / "summary.json").write_text(
            json.dumps(
                {
                    "evidence_level": "A",
                    "config": {},
                    "environment": {},
                    "gallery": [],
                    "aggregates": {
                        "m": {
                            "n_queries": 1, "n_abstentions": 0, "coverage": 1.0,
                            "recall_at": {}, "mrr": {}, "n_open_set_queries": 0,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "per_query.json").write_text("[]", encoding="utf-8")
        with pytest.raises(IncompleteRun, match="per-query evidence is mandatory"):
            load(tmp_path)

    def test_no_aggregates_at_all_fails(self, tmp_path: Path):
        (tmp_path / "summary.json").write_text(
            json.dumps(
                {
                    "evidence_level": "A", "config": {}, "environment": {},
                    "gallery": [], "aggregates": {},
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "per_query.json").write_text(json.dumps([{"query_id": "q"}]), encoding="utf-8")
        with pytest.raises(IncompleteRun, match="no method aggregates"):
            load(tmp_path)
