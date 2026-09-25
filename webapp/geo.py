"""Geographic input: parsing GPX / GeoJSON / CSV, and a local metric projection.

The edit-distance core works in Euclidean coordinates, and latitude/longitude
are not Euclidean. Every audit therefore projects both curves onto a local
tangent plane at the course's centre (equirectangular about that point):

    x = R cos(phi0) (lambda - lambda0),    y = R (phi - phi0)

Over a route of a few kilometres this distorts distances by far less than GPS
error; ``projection_scale_error`` reports the worst-case relative scale error
for the actual extent, and extents where it would matter are rejected.

Everything here is pure and side-effect free. Untrusted input is validated at
this boundary: sizes, coordinate ranges, finiteness, and XML without DTDs.
"""

from __future__ import annotations

import csv
import io
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_008.8

#: Largest accepted extent of an audit, in kilometres (course and track together).
MAX_EXTENT_KM = 100.0

#: Largest accepted |latitude|; the local projection degrades towards the poles.
MAX_ABS_LATITUDE = 80.0

#: Hard cap on points read from any one file, before any simplification.
MAX_POINTS_PER_FILE = 50_000


class GeoInputError(ValueError):
    """User-facing input problem. The message is safe to show verbatim."""


@dataclass(frozen=True, slots=True)
class Fix:
    """One position: WGS84 degrees plus an optional ISO-8601 timestamp string."""

    lat: float
    lon: float
    time: str | None = None


@dataclass(frozen=True, slots=True)
class LocalFrame:
    """The tangent plane an audit is computed in."""

    lat0: float
    lon0: float

    def to_xy(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        dlon = (np.asarray(lon, dtype=float) - self.lon0 + 180.0) % 360.0 - 180.0
        x = EARTH_RADIUS_M * math.cos(math.radians(self.lat0)) * np.radians(dlon)
        y = EARTH_RADIUS_M * np.radians(np.asarray(lat, dtype=float) - self.lat0)
        return np.column_stack([x, y])

    def to_latlon(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        lat = self.lat0 + np.degrees(xy[:, 1] / EARTH_RADIUS_M)
        lon = self.lon0 + np.degrees(
            xy[:, 0] / (EARTH_RADIUS_M * math.cos(math.radians(self.lat0)))
        )
        lon = (lon + 180.0) % 360.0 - 180.0
        return np.column_stack([lat, lon])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_fixes(fixes: list[Fix], what: str, *, min_points: int = 2) -> list[Fix]:
    """Reject anything that is not a usable sequence of finite WGS84 positions."""
    if len(fixes) < min_points:
        raise GeoInputError(f"{what} needs at least {min_points} points, got {len(fixes)}")
    if len(fixes) > MAX_POINTS_PER_FILE:
        raise GeoInputError(
            f"{what} has {len(fixes)} points; the limit is {MAX_POINTS_PER_FILE}"
        )
    for idx, fix in enumerate(fixes):
        if not (math.isfinite(fix.lat) and math.isfinite(fix.lon)):
            raise GeoInputError(f"{what} point {idx} has a non-finite coordinate")
        if not -90.0 <= fix.lat <= 90.0:
            raise GeoInputError(f"{what} point {idx} has latitude {fix.lat} outside [-90, 90]")
        if not -180.0 <= fix.lon <= 180.0:
            raise GeoInputError(
                f"{what} point {idx} has longitude {fix.lon} outside [-180, 180]"
            )
        if abs(fix.lat) > MAX_ABS_LATITUDE:
            raise GeoInputError(
                f"{what} point {idx} is at latitude {fix.lat}; the local projection is "
                f"only used within +/-{MAX_ABS_LATITUDE} degrees"
            )
    return fixes


def frame_for(course: list[Fix], track: list[Fix]) -> LocalFrame:
    """A tangent plane at the course's bounding-box centre, after an extent check."""
    lats = np.array([f.lat for f in course + track])
    lons = np.array([f.lon for f in course + track])
    lat0 = float((lats.min() + lats.max()) / 2.0)
    # Centre longitudes on the course's first point so the antimeridian is safe.
    ref_lon = course[0].lon
    rel = (lons - ref_lon + 180.0) % 360.0 - 180.0
    lon0 = float(ref_lon + (rel.min() + rel.max()) / 2.0)
    lon0 = (lon0 + 180.0) % 360.0 - 180.0
    frame = LocalFrame(lat0, lon0)
    xy = frame.to_xy(lats, lons)
    extent_km = float(np.max(np.ptp(xy, axis=0))) / 1000.0
    if extent_km > MAX_EXTENT_KM:
        raise GeoInputError(
            f"course and track span {extent_km:.0f} km; audits are limited to "
            f"{MAX_EXTENT_KM:.0f} km so the local projection stays accurate"
        )
    return frame


def projection_scale_error(frame: LocalFrame, lats: np.ndarray) -> float:
    """Worst relative east-west scale error of the frame over these latitudes."""
    lats = np.asarray(lats, dtype=float)
    c0 = math.cos(math.radians(frame.lat0))
    return float(np.max(np.abs(np.cos(np.radians(lats)) / c0 - 1.0)))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_track_file(data: bytes, filename: str) -> list[Fix]:
    """Dispatch on extension, falling back to sniffing the content."""
    name = filename.lower()
    text = _decode(data)
    stripped = text.lstrip()
    if name.endswith(".gpx") or stripped.startswith("<"):
        return parse_gpx(text)
    if name.endswith((".geojson", ".json")) or stripped.startswith("{"):
        return parse_geojson(text)
    if name.endswith((".csv", ".txt")):
        return parse_csv(text)
    raise GeoInputError(
        f"cannot tell the format of {filename!r}; use .gpx, .geojson or .csv"
    )


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GeoInputError("file is not valid UTF-8 text") from exc


def parse_gpx(text: str) -> list[Fix]:
    """Track points (``trkpt``), else route points (``rtept``), in document order.

    GPX never needs a DTD, so any ``<!DOCTYPE`` or ``<!ENTITY`` is refused
    outright: that closes entity-expansion attacks without an extra dependency.
    """
    head = text[:4096].upper()
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in head:
        raise GeoInputError("GPX files with a DOCTYPE or entity declarations are not accepted")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise GeoInputError(f"not well-formed GPX: {exc}") from exc

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    for wanted in ("trkpt", "rtept"):
        fixes: list[Fix] = []
        for el in root.iter():
            if local(el.tag) != wanted:
                continue
            try:
                lat = float(el.attrib["lat"])
                lon = float(el.attrib["lon"])
            except (KeyError, ValueError) as exc:
                raise GeoInputError(f"a <{wanted}> is missing a numeric lat/lon") from exc
            time = None
            for child in el:
                if local(child.tag) == "time" and child.text:
                    time = child.text.strip()[:40]
            fixes.append(Fix(lat, lon, time))
            if len(fixes) > MAX_POINTS_PER_FILE:
                raise GeoInputError(f"GPX has more than {MAX_POINTS_PER_FILE} points")
        if fixes:
            return fixes
    raise GeoInputError("GPX contains no <trkpt> or <rtept> points")


def parse_geojson(text: str) -> list[Fix]:
    """The first LineString (or MultiLineString, concatenated) in the document.

    GeoJSON orders coordinates ``[longitude, latitude]``.
    """
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeoInputError(f"not valid JSON: {exc.msg}") from exc
    coords = _first_line(doc)
    if coords is None:
        raise GeoInputError("GeoJSON contains no LineString")
    fixes: list[Fix] = []
    for pos in coords:
        if not isinstance(pos, list) or len(pos) < 2:
            raise GeoInputError("GeoJSON positions must be [longitude, latitude] arrays")
        try:
            fixes.append(Fix(float(pos[1]), float(pos[0])))
        except (TypeError, ValueError) as exc:
            raise GeoInputError("GeoJSON positions must be numeric") from exc
    return fixes


def _first_line(doc: object) -> list[object] | None:
    if not isinstance(doc, dict):
        return None
    kind = doc.get("type")
    if kind == "LineString" and isinstance(doc.get("coordinates"), list):
        return list(doc["coordinates"])
    if kind == "MultiLineString" and isinstance(doc.get("coordinates"), list):
        out: list[object] = []
        for part in doc["coordinates"]:
            if isinstance(part, list):
                out.extend(part)
        return out
    if kind == "Feature":
        return _first_line(doc.get("geometry"))
    if kind == "FeatureCollection" and isinstance(doc.get("features"), list):
        for feature in doc["features"]:
            found = _first_line(feature)
            if found:
                return found
    return None


_LAT_NAMES = ("lat", "latitude", "y")
_LON_NAMES = ("lon", "lng", "long", "longitude", "x")
_TIME_NAMES = ("time", "timestamp", "datetime", "date_time")


def parse_csv(text: str) -> list[Fix]:
    """Columns found by header name (``lat``/``lon``/``time`` and common synonyms),
    or, without a header, the first two columns read as latitude, longitude."""
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        raise GeoInputError("CSV is empty")
    header = [cell.strip().lower() for cell in rows[0]]
    lat_i = next((header.index(n) for n in _LAT_NAMES if n in header), None)
    lon_i = next((header.index(n) for n in _LON_NAMES if n in header), None)
    time_i = next((header.index(n) for n in _TIME_NAMES if n in header), None)
    body = rows[1:]
    if lat_i is None or lon_i is None:
        lat_i, lon_i, time_i, body = 0, 1, None, rows
    fixes: list[Fix] = []
    for line_no, row in enumerate(body, start=1):
        try:
            lat = float(row[lat_i])
            lon = float(row[lon_i])
        except (IndexError, ValueError) as exc:
            raise GeoInputError(f"CSV row {line_no} has no numeric latitude/longitude") from exc
        time = row[time_i].strip()[:40] if time_i is not None and time_i < len(row) else None
        fixes.append(Fix(lat, lon, time or None))
        if len(fixes) > MAX_POINTS_PER_FILE:
            raise GeoInputError(f"CSV has more than {MAX_POINTS_PER_FILE} rows")
    return fixes


# ---------------------------------------------------------------------------
# Geometry helpers (projected metres)
# ---------------------------------------------------------------------------


def point_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to the nearest point of a polyline."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(polyline, dtype=float).reshape(-1, 2)
    if len(poly) == 1:
        return np.asarray(np.linalg.norm(points - poly[0], axis=1))
    a = poly[:-1][None, :, :]
    d = (poly[1:] - poly[:-1])[None, :, :]
    w = points[:, None, :] - a
    len2 = np.einsum("ijk,ijk->ij", d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(len2 > 0, np.einsum("ijk,ijk->ij", w, d) / len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    nearest = a + t[:, :, None] * d
    dist = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    return np.asarray(dist.min(axis=1))


#: Point-to-chord evaluations one course simplification may spend, across all
#: the tolerances it tries. Douglas-Peucker is O(n log n) on real routes but
#: O(n^2) on adversarial ones: a 50 000-point square wave once took 315 s here.
SIMPLIFY_WORK_BUDGET = 40_000_000


def douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices kept by Douglas-Peucker simplification (first and last always kept)."""
    kept, _, _ = _douglas_peucker(np.asarray(points, dtype=float), tolerance, None, None)
    assert kept is not None
    return kept


def _douglas_peucker(
    pts: np.ndarray, tolerance: float, max_keep: int | None, budget: int | None
) -> tuple[np.ndarray | None, int, bool]:
    """``(kept, work, budget_exhausted)``; ``kept`` is None if a limit was hit.

    Stopping as soon as more than ``max_keep`` points are kept bounds an attempt
    to about ``2 * max_keep + 1`` passes over the data, whatever its shape: every
    pass either keeps a point or ends a branch.
    """
    n = len(pts)
    if n <= 2:
        return np.arange(n), 0, False
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    kept_count = 2
    work = 0
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        work += hi - lo - 1
        if budget is not None and work > budget:
            return None, work, True
        dists = point_to_polyline(pts[lo + 1 : hi], pts[[lo, hi]])
        k = int(np.argmax(dists))
        if dists[k] > tolerance:
            mid = lo + 1 + k
            keep[mid] = True
            kept_count += 1
            if max_keep is not None and kept_count > max_keep:
                return None, work, False
            stack.append((lo, mid))
            stack.append((mid, hi))
    return np.flatnonzero(keep), work, False


def simplify_to(
    points: np.ndarray, max_points: int, max_tolerance: float
) -> tuple[np.ndarray, float]:
    """Douglas-Peucker to at most ``max_points``, moving the course at most ``max_tolerance``.

    Tries ``max_tolerance`` first, then halves it (to ``max_tolerance / 64``)
    while the result still fits, and returns the smallest tolerance that
    worked. Returns ``(kept_indices, tolerance_used)``; tolerance 0 means
    untouched. All attempts share :data:`SIMPLIFY_WORK_BUDGET`; if it runs out
    after any success, that valid result is returned. Raises
    :class:`GeoInputError` if even ``max_tolerance`` cannot fit, or if the
    budget runs out first.
    """
    if len(points) <= max_points:
        return np.arange(len(points)), 0.0
    pts = np.asarray(points, dtype=float)
    budget = SIMPLIFY_WORK_BUDGET
    best: tuple[np.ndarray, float] | None = None
    tolerance = max_tolerance
    for _ in range(7):
        kept, work, exhausted = _douglas_peucker(pts, tolerance, max_points, budget)
        budget -= work
        if kept is None:
            if exhausted and best is None:
                raise GeoInputError(
                    f"the course has {len(points)} points in a shape too intricate to "
                    "simplify within this server's work budget; supply a simpler course"
                )
            break
        best = (kept, tolerance)
        tolerance /= 2.0
    if best is None:
        raise GeoInputError(
            f"the course has {len(points)} points and cannot be simplified to {max_points} "
            f"without moving it more than {max_tolerance:.1f} m; supply a simpler course"
        )
    return best


def decimate(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices (always including the last) when ``n > max_points``."""
    if n <= max_points:
        return np.arange(n)
    step = math.ceil((n - 1) / (max_points - 1))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def to_gpx(fixes: list[Fix], name: str) -> str:
    """A minimal GPX 1.1 track. Text content is XML-escaped."""
    from xml.sax.saxutils import escape

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="frechet-edit course audit" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <trk><name>{escape(name)}</name><trkseg>",
    ]
    for fix in fixes:
        time = f"<time>{escape(fix.time)}</time>" if fix.time else ""
        lines.append(f'    <trkpt lat="{fix.lat:.7f}" lon="{fix.lon:.7f}">{time}</trkpt>')
    lines.append("  </trkseg></trk>")
    lines.append("</gpx>")
    return "\n".join(lines) + "\n"
