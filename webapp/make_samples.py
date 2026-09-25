"""Write the sample upload files in ``webapp/samples`` from the synthetic scenarios.

    python -m webapp.make_samples

They exist so the upload path can be tried by hand (drag them onto the page)
and so the browser tests exercise real files. ``webapp/tests`` checks that the
committed copies still match this generator.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from . import scenarios
from .geo import Fix, to_gpx

SAMPLES_DIR = Path(__file__).parent / "samples"


def _csv(fixes: list[Fix]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["time", "lat", "lon"])
    for fix in fixes:
        writer.writerow([fix.time or "", f"{fix.lat:.7f}", f"{fix.lon:.7f}"])
    return buffer.getvalue()


def _geojson(fixes: list[Fix], name: str) -> str:
    doc = {
        "type": "Feature",
        "properties": {"name": name, "note": "synthetic course, not a surveyed route"},
        "geometry": {"type": "LineString", "coordinates": [[f.lon, f.lat] for f in fixes]},
    }
    return json.dumps(doc, indent=1) + "\n"


def build() -> dict[str, str]:
    """File name -> content, deterministic."""
    clean = scenarios.get("clean")
    canyon = scenarios.get("urban-canyon")
    shortcut = scenarios.get("shortcut")
    sparse = scenarios.get("sparse-logging")
    return {
        "marina-bay-course.gpx": to_gpx(clean.course, "Marina Bay loop (synthetic course)"),
        "marina-bay-course.geojson": _geojson(clean.course, "Marina Bay loop (synthetic course)"),
        "urban-canyon-track.gpx": to_gpx(canyon.track, "Urban-canyon run (synthetic)"),
        "urban-canyon-track.csv": _csv(canyon.track),
        "shortcut-track.gpx": to_gpx(shortcut.track, "Shortcut run (synthetic)"),
        "sparse-track.csv": _csv(sparse.track),
    }


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    for name, content in build().items():
        (SAMPLES_DIR / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {SAMPLES_DIR / name}")


if __name__ == "__main__":
    main()
