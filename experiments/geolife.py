"""Loader for GeoLife GPS trajectories, for evidence level B.

**No data ships with this repository and none may be committed.** GeoLife is
distributed under the Microsoft Research License Agreement (non-commercial use
only), which permits academic research and forbids distributing the data *or any
derivative work*. See `docs/data-and-labels.md`. This module therefore reads
from a local path the user supplies and never writes trajectory coordinates
anywhere the repository tracks.

What this module does and does not give you:

* It gives real trajectory GEOMETRY, which is what evidence level B needs.
* It does NOT give route-identity labels. GeoLife's `labels.txt` records
  transportation mode (`bus`, `train`, ...) for 69 of 182 users, which is not a
  route identity. Level C still requires independent annotation and remains
  blocked; nothing here should be read as unblocking it.

Ground truth at level B comes from the corruption applied to a base trajectory,
so it is known by construction and never derived from any Fréchet or FED score.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# GeoLife .plt files carry six header lines before the CSV body.
_PLT_HEADER_LINES = 6
_EARTH_RADIUS_M = 6_371_000.0

# Rejection thresholds for degenerate traces. A stationary logger produces
# hundreds of points inside a few metres; such a trace is not a route and would
# make the retrieval task meaningless rather than hard.
_MIN_RAW_POINTS = 50
_MIN_EXTENT_M = 300.0
_MAX_GAP_M = 2_000.0


@dataclass(frozen=True)
class Trajectory:
    """One resampled GeoLife trace in local metric coordinates."""

    source_id: str
    points: np.ndarray
    raw_points: int
    extent_m: float


class GeoLifeUnavailableError(RuntimeError):
    """Raised when the archive is absent, so callers can skip rather than fake."""


def _parse_plt(text: str) -> np.ndarray:
    """Parse one .plt body into an ``(n, 2)`` array of (latitude, longitude)."""
    rows: list[tuple[float, float]] = []
    for line in text.splitlines()[_PLT_HEADER_LINES:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            # A malformed row is skipped rather than silently zeroed, and
            # rather than aborting a 18670-file scan over one bad line.
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        rows.append((lat, lon))
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _to_metres(latlon: np.ndarray) -> np.ndarray:
    """Equirectangular projection about the trace centroid, in metres.

    GeoLife traces are city-scale, so the distortion of an equirectangular
    projection about the local centroid is far below the delta values used here.
    A full UTM projection would add a dependency for no measurable gain; if that
    ever stops being true the assumption is isolated to this function.
    """
    lat0 = float(np.mean(latlon[:, 0]))
    lat_rad = np.radians(latlon[:, 0])
    lon_rad = np.radians(latlon[:, 1])
    lat0_rad = np.radians(lat0)
    x = _EARTH_RADIUS_M * (lon_rad - np.mean(lon_rad)) * np.cos(lat0_rad)
    y = _EARTH_RADIUS_M * (lat_rad - lat0_rad)
    return np.column_stack([x, y])


def _arc_length_resample(
    points: np.ndarray, n_points: int, spacing_m: float | None = None
) -> np.ndarray:
    """Resample a polyline by arc length.

    With ``spacing_m`` the output has ``n_points`` vertices a fixed distance
    apart, cropping the trace to the first ``(n_points - 1) * spacing_m`` metres.
    That matters because the experiment's spatial parameters - delta, the
    corridor offset, the spike magnitude - are absolute distances in metres. A
    raw GeoLife trace runs for kilometres, so spreading 40 vertices over the
    whole of it would put consecutive vertices hundreds of metres apart and make
    a 25 m threshold describe a completely different problem. Fixing the spacing
    keeps the geometry real and the spatial scale comparable.

    Without ``spacing_m`` the vertices are spread over the whole trace.
    """
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    if total <= 0.0:
        raise ValueError("degenerate trace of zero length")
    if spacing_m is None:
        targets = np.linspace(0.0, total, n_points)
    else:
        needed = (n_points - 1) * spacing_m
        if total < needed:
            raise ValueError(
                f"trace is {total:.0f} m, needs {needed:.0f} m at this spacing"
            )
        targets = np.arange(n_points, dtype=np.float64) * spacing_m
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.column_stack([x, y])


def _accept(points_m: np.ndarray) -> tuple[bool, float]:
    """Reject stationary traces and traces with implausible jumps."""
    extent = float(
        np.linalg.norm(points_m.max(axis=0) - points_m.min(axis=0))
    )
    if extent < _MIN_EXTENT_M:
        return False, extent
    gaps = np.linalg.norm(np.diff(points_m, axis=0), axis=1)
    if gaps.size and float(gaps.max()) > _MAX_GAP_M:
        return False, extent
    return True, extent


def _iter_plt_from_zip(archive: Path) -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".plt"):
                continue
            with zf.open(name) as handle:
                yield name, io.TextIOWrapper(handle, encoding="utf-8", errors="replace").read()


def _iter_plt_from_dir(root: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(root.rglob("*.plt")):
        yield str(path.relative_to(root)), path.read_text(encoding="utf-8", errors="replace")


def resolve_source(source: str | Path | None) -> Path:
    """Locate the local GeoLife archive or extracted directory.

    Looks at ``source`` if given, else the conventional gitignored locations.
    Raises `GeoLifeUnavailableError` rather than inventing data.
    """
    candidates: list[Path] = []
    if source is not None:
        candidates.append(Path(source))
    else:
        base = Path("data/raw")
        candidates.extend(
            [
                base / "geolife-1.3-archive.zip",
                base / "Geolife Trajectories 1.3",
                base / "geolife",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise GeoLifeUnavailableError(
        "No GeoLife archive found. Place the archive at "
        "data/raw/geolife-1.3-archive.zip (gitignored) or pass an explicit "
        "path. The data is not distributed with this repository; see "
        "docs/data-and-labels.md."
    )


def load_base_trajectories(
    n_routes: int,
    n_points: int = 40,
    source: str | Path | None = None,
    max_files_scanned: int = 4000,
    min_separation_m: float = 5_000.0,
    spacing_m: float | None = 10.0,
) -> list[Trajectory]:
    """Load ``n_routes`` well-separated real trajectories, resampled to ``n_points``.

    Selection is deterministic given the archive: files are visited in sorted
    order and the first acceptable, sufficiently separated traces are taken.
    Separation matters because two traces along the same Beijing arterial would
    be near-duplicates, which would corrupt the retrieval task by making a
    "negative" effectively a second correct answer.
    """
    if n_routes < 1:
        raise ValueError("n_routes must be at least 1")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    path = resolve_source(source)
    iterator = (
        _iter_plt_from_zip(path)
        if path.is_file() and path.suffix.lower() == ".zip"
        else _iter_plt_from_dir(path)
    )

    chosen: list[Trajectory] = []
    centroids: list[np.ndarray] = []
    scanned = 0
    for name, text in iterator:
        if len(chosen) >= n_routes or scanned >= max_files_scanned:
            break
        scanned += 1
        latlon = _parse_plt(text)
        if len(latlon) < _MIN_RAW_POINTS:
            continue
        points_m = _to_metres(latlon)
        ok, extent = _accept(points_m)
        if not ok:
            continue
        # Separation is measured in absolute lat/lon space, since each trace is
        # projected about its own centroid and local frames are not comparable.
        geo_centroid = latlon.mean(axis=0)
        if any(
            _haversine_m(geo_centroid, other) < min_separation_m
            for other in centroids
        ):
            continue
        try:
            resampled = _arc_length_resample(points_m, n_points, spacing_m)
        except ValueError:
            continue
        centroids.append(geo_centroid)
        chosen.append(
            Trajectory(
                source_id=name.rsplit("/", 1)[-1].replace(".plt", ""),
                points=resampled,
                raw_points=len(latlon),
                extent_m=extent,
            )
        )

    if len(chosen) < n_routes:
        raise GeoLifeUnavailableError(
            f"found only {len(chosen)} acceptable trajectories after scanning "
            f"{scanned} files; asked for {n_routes}. Raise max_files_scanned or "
            f"lower min_separation_m."
        )
    return chosen


def _haversine_m(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(h)))
