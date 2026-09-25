"""Parsing, validation and the local projection. Untrusted input lives here."""

from __future__ import annotations

import math

import numpy as np
import pytest

from webapp.geo import (
    EARTH_RADIUS_M,
    Fix,
    GeoInputError,
    LocalFrame,
    decimate,
    douglas_peucker,
    frame_for,
    parse_csv,
    parse_geojson,
    parse_gpx,
    parse_track_file,
    point_to_polyline,
    simplify_to,
    to_gpx,
    validate_fixes,
)

GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><name>morning</name><trkseg>
    <trkpt lat="1.2868" lon="103.8545"><ele>5</ele><time>2026-03-07T06:30:00Z</time></trkpt>
    <trkpt lat="1.2879" lon="103.8542"><time>2026-03-07T06:30:02Z</time></trkpt>
    <trkpt lat="1.2890" lon="103.8544"></trkpt>
  </trkseg></trk>
</gpx>"""


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class TestGpx:
    def test_track_points_with_optional_times(self):
        fixes = parse_gpx(GPX)
        assert [f.lat for f in fixes] == [1.2868, 1.2879, 1.2890]
        assert fixes[0].time == "2026-03-07T06:30:00Z"
        assert fixes[2].time is None

    def test_route_points_when_there_is_no_track(self):
        gpx = (
            '<gpx xmlns="http://www.topografix.com/GPX/1/1"><rte>'
            '<rtept lat="1" lon="2"/><rtept lat="3" lon="4"/></rte></gpx>'
        )
        assert [(f.lat, f.lon) for f in parse_gpx(gpx)] == [(1.0, 2.0), (3.0, 4.0)]

    def test_doctype_and_entities_are_refused(self):
        bomb = '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]><gpx>&lol;</gpx>'
        with pytest.raises(GeoInputError, match="DOCTYPE"):
            parse_gpx(bomb)

    def test_malformed_xml_is_a_user_error(self):
        with pytest.raises(GeoInputError, match="well-formed"):
            parse_gpx("<gpx><trkpt lat='1'")

    def test_missing_coordinates_are_reported(self):
        with pytest.raises(GeoInputError, match="lat/lon"):
            parse_gpx('<gpx><trk><trkseg><trkpt lat="1"/></trkseg></trk></gpx>')

    def test_no_points(self):
        with pytest.raises(GeoInputError, match="no <trkpt>"):
            parse_gpx("<gpx></gpx>")

    def test_round_trip_through_the_writer(self):
        fixes = [Fix(1.25, 103.5, "2026-01-01T00:00:00Z"), Fix(1.26, 103.51, None)]
        again = parse_gpx(to_gpx(fixes, "a <b> & c"))
        assert [(f.lat, f.lon, f.time) for f in again] == [
            (1.25, 103.5, "2026-01-01T00:00:00Z"),
            (1.26, 103.51, None),
        ]


class TestGeoJsonAndCsv:
    def test_geojson_is_longitude_first(self):
        doc = (
            '{"type":"Feature","geometry":'
            '{"type":"LineString","coordinates":[[103.8,1.2],[103.9,1.3]]}}'
        )
        fixes = parse_geojson(doc)
        assert (fixes[0].lat, fixes[0].lon) == (1.2, 103.8)

    def test_feature_collection_and_multilinestring(self):
        doc = (
            '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":'
            '{"type":"MultiLineString","coordinates":[[[0,1],[0,2]],[[0,3]]]}}]}'
        )
        assert [f.lat for f in parse_geojson(doc)] == [1.0, 2.0, 3.0]

    def test_geojson_without_a_line(self):
        with pytest.raises(GeoInputError, match="LineString"):
            parse_geojson('{"type":"Point","coordinates":[0,0]}')

    def test_csv_with_named_columns_in_any_order(self):
        text = "time,longitude,latitude\n2026-01-01T00:00:00Z,103.8,1.2\n,103.9,1.3\n"
        fixes = parse_csv(text)
        assert [(f.lat, f.lon, f.time) for f in fixes] == [
            (1.2, 103.8, "2026-01-01T00:00:00Z"),
            (1.3, 103.9, None),
        ]

    def test_csv_without_a_header_is_lat_lon(self):
        assert [(f.lat, f.lon) for f in parse_csv("1.2,103.8\n1.3,103.9\n")] == [
            (1.2, 103.8),
            (1.3, 103.9),
        ]

    def test_csv_bad_row(self):
        with pytest.raises(GeoInputError, match="row 2"):
            parse_csv("lat,lon\n1,2\nx,y\n")

    def test_dispatch_by_extension_and_content(self):
        assert len(parse_track_file(GPX.encode(), "run.gpx")) == 3
        assert len(parse_track_file(b"1,2\n3,4\n", "run.csv")) == 2
        assert len(parse_track_file(b'{"type":"LineString","coordinates":[[1,2],[3,4]]}', "x")) == 2
        with pytest.raises(GeoInputError, match="format"):
            parse_track_file(b"1 2 3", "run.bin")
        with pytest.raises(GeoInputError, match="UTF-8"):
            parse_track_file(b"\xff\xfe\xfa", "run.csv")


class TestValidation:
    @pytest.mark.parametrize(
        ("fix", "message"),
        [
            (Fix(91.0, 0.0), "latitude"),
            (Fix(0.0, 181.0), "longitude"),
            (Fix(float("nan"), 0.0), "non-finite"),
            (Fix(85.0, 0.0), "projection"),
        ],
    )
    def test_bad_positions(self, fix, message):
        with pytest.raises(GeoInputError, match=message):
            validate_fixes([Fix(0.0, 0.0), fix], "track")

    def test_too_few_points(self):
        with pytest.raises(GeoInputError, match="at least 2"):
            validate_fixes([Fix(0.0, 0.0)], "course")

    def test_extent_limit(self):
        with pytest.raises(GeoInputError, match="span"):
            frame_for([Fix(1.0, 103.0), Fix(1.0, 104.5)], [Fix(1.0, 103.0), Fix(1.0, 103.1)])


class TestProjection:
    def test_distances_match_the_great_circle_to_a_centimetre_per_kilometre(self):
        frame = LocalFrame(1.2855, 103.8575)
        rng = np.random.default_rng(0)
        for _ in range(200):
            lat = 1.2855 + rng.uniform(-0.02, 0.02, 2)
            lon = 103.8575 + rng.uniform(-0.02, 0.02, 2)
            xy = frame.to_xy(lat, lon)
            planar = float(np.linalg.norm(xy[0] - xy[1]))
            sphere = _haversine(lat[0], lon[0], lat[1], lon[1])
            assert abs(planar - sphere) <= 1e-5 * sphere + 1e-6

    def test_round_trip(self):
        frame = LocalFrame(51.5, -0.12)
        lat, lon = np.array([51.49, 51.52]), np.array([-0.15, -0.09])
        back = frame.to_latlon(frame.to_xy(lat, lon))
        assert np.allclose(back[:, 0], lat) and np.allclose(back[:, 1], lon)

    def test_the_antimeridian_is_not_a_cliff(self):
        course = [Fix(-17.0, 179.999), Fix(-17.0, -179.999)]
        frame = frame_for(course, course)
        xy = frame.to_xy(np.array([-17.0, -17.0]), np.array([179.999, -179.999]))
        assert float(np.linalg.norm(xy[0] - xy[1])) < 250.0


class TestGeometry:
    def test_point_to_polyline(self):
        poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        pts = np.array([[5.0, 3.0], [12.0, 5.0], [-4.0, 3.0], [10.0, 10.0]])
        assert np.allclose(point_to_polyline(pts, poly), [3.0, 2.0, 5.0, 0.0])
        assert np.allclose(point_to_polyline(pts[:1], poly[:1]), [math.hypot(5, 3)])

    def test_douglas_peucker_keeps_ends_and_respects_tolerance(self):
        t = np.linspace(0, 1, 400)
        pts = np.column_stack([1000 * t, 30 * np.sin(12 * t)])
        kept = douglas_peucker(pts, 2.0)
        assert kept[0] == 0 and kept[-1] == len(pts) - 1
        assert point_to_polyline(pts, pts[kept]).max() <= 2.0 + 1e-9

    def test_simplify_to_a_point_budget(self):
        t = np.linspace(0, 1, 2000)
        pts = np.column_stack([3000 * t, 40 * np.sin(20 * t)])
        kept, tol = simplify_to(pts, 150, max_tolerance=10.0)
        assert len(kept) <= 150 and 0 < tol <= 10.0
        with pytest.raises(GeoInputError, match="cannot be simplified"):
            simplify_to(np.column_stack([t, np.sin(1000 * t)]) * 1000, 5, max_tolerance=0.01)

    def test_decimate_keeps_both_ends(self):
        idx = decimate(1001, 100)
        assert idx[0] == 0 and idx[-1] == 1000 and len(idx) <= 101
        assert list(decimate(10, 100)) == list(range(10))


def test_committed_sample_files_match_their_generator():
    """webapp/samples is generated; regenerate with `python -m webapp.make_samples`."""
    from webapp.make_samples import SAMPLES_DIR, build

    for name, content in build().items():
        on_disk = (SAMPLES_DIR / name).read_text(encoding="utf-8")
        assert on_disk == content, f"{name} is stale; run python -m webapp.make_samples"


class TestSimplificationCannotBeWeaponised:
    """Security review, CRITICAL: Douglas-Peucker is O(n^2) on a square wave.

    A 50 000-point course alternating between two lines 40 m apart, inside every
    request limit, took 315 s of CPU before the solver ran. The work is now
    budgeted, and exceeding the budget is a prompt, explained refusal.
    """

    @staticmethod
    def _square_wave(n: int) -> np.ndarray:
        x = np.arange(n) * 1.8
        y = np.where(np.arange(n) % 2 == 0, 0.0, 40.0)
        return np.column_stack([x, y])

    def test_the_reported_attack_now_finishes_in_seconds(self):
        """The reviewer's input: 40 m square wave, 50 000 points, delta 500
        (so up to 125 m of simplification is allowed). It is simplifiable, and
        now the answer comes fast instead of after 315 s."""
        import time

        pts = self._square_wave(50_000)
        started = time.perf_counter()
        kept, tol = simplify_to(pts, 150, max_tolerance=125.0)
        assert time.perf_counter() - started < 10.0
        assert len(kept) <= 150 and 0 < tol <= 125.0
        assert point_to_polyline(pts, pts[kept]).max() <= tol + 1e-9

    def test_an_unsimplifiable_square_wave_is_refused_in_seconds(self):
        import time

        pts = self._square_wave(50_000) * np.array([1.0, 10.0])  # 400 m swings
        started = time.perf_counter()
        with pytest.raises(GeoInputError, match="cannot be simplified"):
            simplify_to(pts, 150, max_tolerance=125.0)
        assert time.perf_counter() - started < 10.0

    def test_the_work_budget_is_a_hard_backstop(self, monkeypatch):
        import webapp.geo as geo

        monkeypatch.setattr(geo, "SIMPLIFY_WORK_BUDGET", 1_000)
        with pytest.raises(GeoInputError, match="too intricate"):
            geo.simplify_to(self._square_wave(5_000), 150, max_tolerance=125.0)

    def test_an_ordinary_dense_course_still_simplifies(self):
        t = np.linspace(0, 1, 50_000)
        pts = np.column_stack([20_000 * t, 300 * np.sin(9 * t) + 80 * np.sin(31 * t)])
        kept, tol = simplify_to(pts, 150, max_tolerance=6.25)
        assert len(kept) <= 150 and tol > 0
