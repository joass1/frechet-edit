"""Built-in demonstration scenarios.

All five are SYNTHETIC and deterministic. The course is a loop drawn roughly
around Marina Bay, Singapore, from approximate landmark coordinates; it is not
a surveyed route and does not follow real footpaths exactly. Tracks are
simulated runners with spatially correlated GPS noise, so every result is known
by construction. Nothing here is a real person's data.

Each scenario isolates one reason simple checks fail:

* ``clean``            - control: a good run, nothing to remove.
* ``urban-canyon``     - multipath spikes from tall buildings. Ordinary Frechet
                         is wrecked by one spike; the edit distance removes them.
* ``sparse-logging``   - one fix every 30 s. Discrete Frechet cannot align a
                         sparse course with sparse fixes; continuous can.
* ``shortcut``         - the runner cuts across instead of rounding the bay.
* ``one-lap-of-two``   - a two-lap race, one lap run. Every fix is on the
                         course and every part of the course is visited, so
                         both naive checks PASS; only an order-aware measure
                         notices that half the race is missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .geo import Fix, LocalFrame

# A loop around Marina Bay, clockwise from the Merlion. (lat, lon), degrees.
MARINA_BAY_LOOP: tuple[tuple[float, float], ...] = (
    (1.28680, 103.85446),  # Merlion
    (1.28790, 103.85420),  # Esplanade Bridge, south end
    (1.28900, 103.85440),  # Esplanade Bridge, north end
    (1.28960, 103.85560),  # Esplanade waterfront
    (1.28930, 103.85730),  # Makansutra / Esplanade east
    (1.28880, 103.85900),  # Bayfront approach
    (1.28840, 103.86030),  # Helix Bridge, north end
    (1.28700, 103.86070),  # Helix Bridge, south end
    (1.28520, 103.86000),  # Marina Bay Sands promenade, north
    (1.28390, 103.85890),  # Event Plaza
    (1.28240, 103.85760),  # Promontory approach
    (1.28130, 103.85600),  # The Promontory
    (1.28210, 103.85430),  # Marina Bay Financial Centre promenade
    (1.28310, 103.85290),  # Clifford Pier
    (1.28430, 103.85300),  # Fullerton Bay
    (1.28560, 103.85370),  # One Fullerton
    (1.28680, 103.85446),  # Merlion (finish)
)

START_TIME = datetime(2026, 3, 7, 6, 30, 0, tzinfo=timezone.utc)
FRAME = LocalFrame(1.2855, 103.8575)
RUN_SPEED_MPS = 3.2
DEFAULT_DELTA_M = 25.0


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    blurb: str
    expectation: str
    course: list[Fix]
    track: list[Fix]
    delta_m: float = DEFAULT_DELTA_M
    #: Track indices where a glitch was injected (ground truth by construction).
    injected: tuple[int, ...] = ()


def _course_xy(laps: int = 1) -> np.ndarray:
    lat = np.array([p[0] for p in MARINA_BAY_LOOP])
    lon = np.array([p[1] for p in MARINA_BAY_LOOP])
    loop = FRAME.to_xy(lat, lon)
    parts = [loop] + [loop[1:] for _ in range(laps - 1)]
    return np.vstack(parts)


def _resample(path: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1], spacing)
    s = np.append(s, cum[-1])
    return np.column_stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])


def _gps_noise(rng: np.random.Generator, n: int, sigma: float, rho: float = 0.93) -> np.ndarray:
    """AR(1) noise: consecutive GPS errors are strongly correlated in practice."""
    out = np.zeros((n, 2))
    innovation = sigma * math.sqrt(1.0 - rho * rho)
    out[0] = rng.normal(0.0, sigma, 2)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + rng.normal(0.0, innovation, 2)
    return out


def _fixes(xy: np.ndarray, interval_s: float) -> list[Fix]:
    latlon = FRAME.to_latlon(xy)
    return [
        Fix(
            round(float(lat), 7),
            round(float(lon), 7),
            (START_TIME + timedelta(seconds=interval_s * i)).isoformat().replace("+00:00", "Z"),
        )
        for i, (lat, lon) in enumerate(latlon)
    ]


def _course_fixes(laps: int = 1) -> list[Fix]:
    latlon = FRAME.to_latlon(_course_xy(laps))
    return [Fix(round(float(a), 7), round(float(b), 7)) for a, b in latlon]


def _run(path: np.ndarray, interval_s: float, seed: int, sigma: float = 3.0) -> np.ndarray:
    pts = _resample(path, RUN_SPEED_MPS * interval_s)
    rng = np.random.default_rng(seed)
    noisy = pts + _gps_noise(rng, len(pts), sigma)
    noisy[0], noisy[-1] = pts[0], pts[-1]
    return np.asarray(noisy)


def clean() -> Scenario:
    track = _run(_course_xy(), 2.0, seed=1)
    return Scenario(
        id="clean",
        title="Clean run",
        blurb="A good 3.4 km loop logged every 2 s with ordinary GPS noise.",
        expectation="On course, nothing removed.",
        course=_course_fixes(),
        track=_fixes(track, 2.0),
    )


def urban_canyon() -> Scenario:
    track = _run(_course_xy(), 2.0, seed=2)
    rng = np.random.default_rng(22)
    n = len(track)
    # Multipath near the CBD towers (the south-west third of the loop).
    spots = [int(n * f) for f in (0.66, 0.71, 0.72, 0.79, 0.86, 0.91)]
    for i in spots:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = rng.uniform(70.0, 240.0)
        track[i] = track[i] + radius * np.array([math.cos(angle), math.sin(angle)])
    return Scenario(
        id="urban-canyon",
        title="Urban-canyon GPS spikes",
        blurb="Same run, but six fixes near the CBD towers jump 70-240 m (multipath).",
        expectation="On course after removing exactly the six spikes.",
        course=_course_fixes(),
        track=_fixes(track, 2.0),
        injected=tuple(spots),
    )


def sparse_logging() -> Scenario:
    track = _run(_course_xy(), 30.0, seed=3, sigma=4.0)
    return Scenario(
        id="sparse-logging",
        title="Sparse logging (1 fix / 30 s)",
        blurb="A watch in power-saving mode: one fix every 30 s, about 96 m apart.",
        expectation="On course. Discrete Frechet cannot tell; continuous can.",
        course=_course_fixes(),
        track=_fixes(track, 30.0),
        delta_m=30.0,
    )


def shortcut() -> Scenario:
    course = _course_xy()
    # Skip the Helix Bridge / Marina Bay Sands side: cut straight from the
    # Esplanade waterfront (vertex 4) to The Promontory (vertex 11).
    cut = np.vstack([course[:5], course[11:]])
    track = _run(cut, 2.0, seed=4)
    return Scenario(
        id="shortcut",
        title="Shortcut across the bay",
        blurb="The runner skips the Helix Bridge side and cuts straight across.",
        expectation="Off course: part of the course is never visited.",
        course=_course_fixes(),
        track=_fixes(track, 2.0),
    )


def one_lap_of_two() -> Scenario:
    track = _run(_course_xy(), 2.0, seed=5)
    return Scenario(
        id="one-lap-of-two",
        title="One lap of a two-lap race",
        blurb="The course is two laps. The runner does one. Every fix is on the course.",
        expectation="Off course - although both naive checks pass.",
        course=_course_fixes(laps=2),
        track=_fixes(track, 2.0),
    )


BUILDERS = {
    "clean": clean,
    "urban-canyon": urban_canyon,
    "sparse-logging": sparse_logging,
    "shortcut": shortcut,
    "one-lap-of-two": one_lap_of_two,
}


def get(scenario_id: str) -> Scenario:
    try:
        return BUILDERS[scenario_id]()
    except KeyError:
        raise KeyError(scenario_id) from None


def catalogue() -> list[dict[str, object]]:
    out = []
    for builder in BUILDERS.values():
        sc = builder()
        out.append(
            {
                "id": sc.id,
                "title": sc.title,
                "blurb": sc.blurb,
                "expectation": sc.expectation,
                "delta_m": sc.delta_m,
            }
        )
    return out
