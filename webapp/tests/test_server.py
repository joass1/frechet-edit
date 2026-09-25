"""HTTP contract: envelopes, validation, limits and security headers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp import server
from webapp.scenarios import get as get_scenario


@pytest.fixture(scope="module")
def client():
    with TestClient(server.create_app()) as test_client:
        yield test_client


def _fixes(fixes):
    return [{"lat": f.lat, "lon": f.lon, "time": f.time} for f in fixes]


class TestReadEndpoints:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body == {"success": True, "data": body["data"], "error": None}
        assert body["data"]["status"] == "ok"

    def test_scenarios_catalogue_and_detail(self, client):
        items = client.get("/api/scenarios").json()["data"]
        assert {i["id"] for i in items} == {
            "clean", "urban-canyon", "sparse-logging", "shortcut", "one-lap-of-two",
        }
        detail = client.get("/api/scenarios/urban-canyon").json()["data"]
        assert len(detail["injected"]) == 6 and len(detail["track"]) > 100

    def test_unknown_scenario_is_404_in_the_envelope(self, client):
        response = client.get("/api/scenarios/nope")
        assert response.status_code == 404
        assert response.json()["success"] is False

    def test_index_and_static_assets(self, client):
        page = client.get("/")
        assert page.status_code == 200 and "Course Check" in page.text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/vendor/leaflet/leaflet.js").status_code == 200

    def test_security_headers_on_every_response(self, client):
        for path in ("/", "/api/health", "/api/scenarios/nope"):
            headers = client.get(path).headers
            assert "script-src 'self'" in headers["content-security-policy"]
            assert "unsafe-inline" not in headers["content-security-policy"]
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["x-frame-options"] == "DENY"

    def test_api_docs_are_not_exposed(self, client):
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


class TestAudit:
    def test_round_trip_on_a_scenario(self, client):
        sc = get_scenario("urban-canyon")
        response = client.post(
            "/api/audit",
            json={"course": _fixes(sc.course), "track": _fixes(sc.track), "delta_m": sc.delta_m},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["verdict"]["code"] == "on_course_after_cleaning"
        assert sorted(g["index"] for g in data["glitches"]) == sorted(sc.injected)

    @pytest.mark.parametrize(
        ("patch", "fragment"),
        [
            ({"delta_m": 0}, "delta_m"),
            ({"delta_m": 9999}, "delta_m"),
            ({"course": [{"lat": 1, "lon": 2}]}, "course"),
            ({"track": "nope"}, "track"),
        ],
    )
    def test_invalid_requests_are_422_with_a_readable_message(self, client, patch, fragment):
        sc = get_scenario("clean")
        payload = {"course": _fixes(sc.course), "track": _fixes(sc.track), "delta_m": 25} | patch
        response = client.post("/api/audit", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False and fragment in body["error"]

    def test_out_of_range_coordinates_are_400(self, client):
        payload = {
            "course": [{"lat": 0, "lon": 0}, {"lat": 95, "lon": 0}],
            "track": [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 0.001}],
            "delta_m": 25,
        }
        response = client.post("/api/audit", json=payload)
        assert response.status_code == 400 and "latitude" in response.json()["error"]

    def test_busy_server_answers_503_instead_of_queueing(self, client, monkeypatch):
        class Full:
            def acquire(self, blocking=True):
                return False

            def release(self):
                raise AssertionError("never acquired")

        monkeypatch.setattr(server, "_audit_slots", Full())
        sc = get_scenario("clean")
        response = client.post(
            "/api/audit",
            json={"course": _fixes(sc.course), "track": _fixes(sc.track), "delta_m": 25},
        )
        assert response.status_code == 503


class TestUploadsAndExport:
    GPX = (
        b'<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        b'<trkpt lat="1.2868" lon="103.8545"/><trkpt lat="1.2879" lon="103.8542"/>'
        b"</trkseg></trk></gpx>"
    )

    def test_parse_a_gpx_upload(self, client):
        upload = {"file": ("run.gpx", self.GPX, "application/gpx+xml")}
        response = client.post("/api/parse", files=upload)
        data = response.json()["data"]
        assert response.status_code == 200 and len(data["points"]) == 2

    def test_hostile_xml_is_refused(self, client):
        bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><gpx>&a;</gpx>'
        response = client.post("/api/parse", files={"file": ("bomb.gpx", bomb, "text/xml")})
        assert response.status_code == 400 and "DOCTYPE" in response.json()["error"]

    def test_oversized_upload_is_413(self, client):
        big = b"1,2\n" * (server.MAX_UPLOAD_BYTES // 4 + 10)
        response = client.post("/api/parse", files={"file": ("big.csv", big, "text/csv")})
        assert response.status_code == 413

    def test_gpx_export_is_a_download_with_a_safe_name(self, client):
        response = client.post(
            "/api/gpx",
            json={"name": '../../evil"name', "points": [{"lat": 1.0, "lon": 2.0, "time": None}]},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/gpx+xml")
        assert response.headers["content-disposition"] == 'attachment; filename="evilname.gpx"'
        assert '<trkpt lat="1.0000000" lon="2.0000000">' in response.text


class TestSecurityReviewFindings:
    def test_a_chunked_body_cannot_bypass_the_size_cap(self, client):
        """HIGH: the cap once read only Content-Length; a chunked 150 MB body
        was buffered whole. It must now be cut off by counting bytes."""
        chunk = b" " * (1024 * 1024)

        def body():
            yield b'{"course": ['
            for _ in range(server.MAX_BODY_BYTES // len(chunk) + 4):
                yield chunk

        response = client.post(
            "/api/audit", content=body(), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413
        assert response.json()["success"] is False

    def test_a_declared_oversize_body_is_refused_before_reading(self, client):
        response = client.post(
            "/api/gpx",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(server.MAX_BODY_BYTES + 1),
            },
        )
        assert response.status_code == 413

    @pytest.mark.parametrize("site", ["cross-site", "same-site"])
    def test_cross_site_browser_posts_are_refused(self, client, site):
        """MEDIUM: multipart is a CORS 'simple' request, so another page could
        make a visitor's browser post files here without a preflight."""
        upload = {"file": ("run.gpx", TestUploadsAndExport.GPX, "application/gpx+xml")}
        response = client.post("/api/parse", files=upload, headers={"Sec-Fetch-Site": site})
        assert response.status_code == 403

    @pytest.mark.parametrize("site", ["same-origin", "none", None])
    def test_same_origin_and_non_browser_clients_are_allowed(self, client, site):
        upload = {"file": ("run.gpx", TestUploadsAndExport.GPX, "application/gpx+xml")}
        headers = {} if site is None else {"Sec-Fetch-Site": site}
        assert client.post("/api/parse", files=upload, headers=headers).status_code == 200
