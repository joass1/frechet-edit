"""Tests for the GeoLife loader.

Almost everything here runs without the dataset: the parser, the projection and
the resampler are exercised on synthetic .plt text, so CI covers them on a
machine that has no archive and is not entitled to one. The single test that
needs real data skips itself when the archive is absent, and never fabricates a
stand-in - a fake trajectory passing as real is exactly the failure this project
refuses to risk.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments import geolife

PLT_HEADER = "\n".join(
    [
        "Geolife trajectory",
        "WGS 84",
        "Altitude is in Feet",
        "Reserved 3",
        "0,2,255,My Track,0,0,2,8421376",
        "0",
    ]
)


def _plt(rows: list[tuple[float, float]]) -> str:
    body = "\n".join(
        f"{lat},{lon},0,492,39744.12,2008-10-23,02:53:04" for lat, lon in rows
    )
    return f"{PLT_HEADER}\n{body}\n"


class TestParsePlt:
    def test_skips_the_six_header_lines(self):
        text = _plt([(39.9, 116.3), (39.91, 116.31)])
        assert geolife._parse_plt(text).shape == (2, 2)

    def test_reads_latitude_and_longitude_in_order(self):
        parsed = geolife._parse_plt(_plt([(39.984702, 116.318417)]))
        assert parsed[0, 0] == pytest.approx(39.984702)
        assert parsed[0, 1] == pytest.approx(116.318417)

    def test_skips_malformed_rows_without_aborting_the_file(self):
        text = _plt([(39.9, 116.3)]) + "not,a,number\n" + "39.95,116.35,0,0,0,d,t\n"
        assert len(geolife._parse_plt(text)) == 2

    def test_rejects_out_of_range_coordinates(self):
        text = _plt([(39.9, 116.3), (999.0, 116.3)])
        assert len(geolife._parse_plt(text)) == 1

    def test_empty_body_gives_an_empty_array_not_an_error(self):
        assert geolife._parse_plt(PLT_HEADER + "\n").shape == (0, 2)


class TestProjection:
    def test_one_degree_of_latitude_is_about_111_km(self):
        latlon = np.array([[39.0, 116.0], [40.0, 116.0]])
        metres = geolife._to_metres(latlon)
        span = abs(metres[1, 1] - metres[0, 1])
        assert 110_000 < span < 112_000

    def test_longitude_is_compressed_by_latitude(self):
        latlon = np.array([[39.9, 116.0], [39.9, 117.0]])
        span = abs(geolife._to_metres(latlon)[1, 0] - geolife._to_metres(latlon)[0, 0])
        # cos(39.9 degrees) is about 0.767, so a degree of longitude here is
        # roughly 85 km rather than 111 km.
        assert 84_000 < span < 87_000


class TestResampling:
    def test_fixed_spacing_produces_that_spacing(self):
        line = np.column_stack([np.linspace(0, 1000, 101), np.zeros(101)])
        out = geolife._arc_length_resample(line, 40, spacing_m=10.0)
        steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert out.shape == (40, 2)
        assert steps == pytest.approx(np.full(39, 10.0), abs=1e-6)

    def test_fixed_spacing_refuses_a_trace_that_is_too_short(self):
        line = np.column_stack([np.linspace(0, 50, 10), np.zeros(10)])
        with pytest.raises(ValueError, match="needs"):
            geolife._arc_length_resample(line, 40, spacing_m=10.0)

    def test_without_spacing_it_spans_the_whole_trace(self):
        line = np.column_stack([np.linspace(0, 1000, 101), np.zeros(101)])
        out = geolife._arc_length_resample(line, 40)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[-1, 0] == pytest.approx(1000.0)

    def test_zero_length_trace_is_rejected(self):
        with pytest.raises(ValueError, match="degenerate"):
            geolife._arc_length_resample(np.zeros((5, 2)), 3, spacing_m=None)


class TestAcceptance:
    def test_a_stationary_trace_is_rejected(self):
        ok, _ = geolife._accept(np.zeros((100, 2)))
        assert not ok

    def test_a_trace_with_an_implausible_jump_is_rejected(self):
        points = np.array([[0.0, 0.0], [10.0, 0.0], [50_000.0, 0.0]])
        ok, _ = geolife._accept(points)
        assert not ok

    def test_an_ordinary_route_is_accepted(self):
        points = np.column_stack([np.linspace(0, 2000, 200), np.zeros(200)])
        ok, extent = geolife._accept(points)
        assert ok
        assert extent == pytest.approx(2000.0)


class TestSourceResolution:
    def test_missing_archive_raises_a_pointed_error(self, tmp_path):
        with pytest.raises(geolife.GeoLifeUnavailableError, match="not distributed"):
            geolife.resolve_source(tmp_path / "nope.zip")

    def test_an_explicit_path_is_honoured(self, tmp_path):
        target = tmp_path / "geolife"
        target.mkdir()
        assert geolife.resolve_source(target) == target


class TestAgainstTheRealArchive:
    """Skipped wherever the dataset is absent. Never substitutes fake data."""

    @pytest.fixture
    def archive(self):
        try:
            return geolife.resolve_source(None)
        except geolife.GeoLifeUnavailableError:
            pytest.skip("GeoLife archive not present; see docs/data-and-labels.md")

    def test_loads_well_formed_base_trajectories(self, archive):
        trajectories = geolife.load_base_trajectories(
            3, n_points=40, source=archive, spacing_m=10.0
        )
        assert len(trajectories) == 3
        for trajectory in trajectories:
            assert trajectory.points.shape == (40, 2)
            assert np.isfinite(trajectory.points).all()
            steps = np.linalg.norm(np.diff(trajectory.points, axis=0), axis=1)
            assert steps.mean() == pytest.approx(10.0, rel=0.2)

    def test_selection_is_deterministic(self, archive):
        first = geolife.load_base_trajectories(3, source=archive)
        second = geolife.load_base_trajectories(3, source=archive)
        assert [t.source_id for t in first] == [t.source_id for t in second]

    def test_chosen_traces_are_distinct(self, archive):
        trajectories = geolife.load_base_trajectories(5, source=archive)
        assert len({t.source_id for t in trajectories}) == 5
