"""Synthetic route galleries with HARD negatives.

Evidence level A. These are analytically controlled fixtures for studying the
mechanism of the edit objective. They are not trajectories, they are not
sampled from any real distribution, and no conclusion about real GPS data
follows from them. See ``docs/experiment-protocol.md``.

A gallery of random distant routes would make retrieval trivial and the
resulting numbers meaningless, so every gallery here contains, for each base
route, negatives that are genuinely hard:

* a **corridor** twin running parallel a short distance away,
* a **shared-endpoint** route that starts and ends in the same places but takes
  a different path between them,
* a **partial-overlap** route that follows the base for part of its length and
  then diverges,
* a **detour** route that follows the base but takes a real excursion, which is
  the acknowledged failure mode of an edit objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RouteKind = Literal["base", "corridor", "shared_endpoint", "partial_overlap", "detour", "distant"]


@dataclass(frozen=True)
class Route:
    """One gallery entry."""

    route_id: str
    kind: RouteKind
    family: str
    points: np.ndarray

    def as_dict(self) -> dict:
        return {
            "route_id": self.route_id,
            "kind": self.kind,
            "family": self.family,
            "n_points": len(self.points),
        }


def make_route(
    rng: np.random.Generator,
    n_points: int = 40,
    step: float = 10.0,
    turn_sigma: float = 0.22,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """A persistent random walk standing in for a sampled route, in metres."""
    heading = float(rng.uniform(0.0, 2.0 * np.pi))
    points = np.zeros((n_points, 2), dtype=np.float64)
    points[0] = origin
    for i in range(1, n_points):
        heading += float(rng.normal(0.0, turn_sigma))
        points[i] = points[i - 1] + step * np.array([np.cos(heading), np.sin(heading)])
    return points


def _normals(points: np.ndarray) -> np.ndarray:
    """Unit normals along a polyline, used to build a parallel corridor."""
    tangents = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    unit = tangents / lengths
    return np.stack([-unit[:, 1], unit[:, 0]], axis=1)


def corridor_twin(points: np.ndarray, offset: float) -> np.ndarray:
    """A route running parallel to ``points`` at a fixed lateral offset."""
    return points + offset * _normals(points)


def shared_endpoint_route(
    rng: np.random.Generator, points: np.ndarray, bulge: float
) -> np.ndarray:
    """Same start and end, a different path between them."""
    n = len(points)
    start, end = points[0], points[-1]
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    straight = start + t * (end - start)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        perpendicular = np.array([0.0, 1.0])
    else:
        perpendicular = np.array([-direction[1], direction[0]]) / norm
    arc = np.sin(np.pi * t) * bulge * float(rng.choice([-1.0, 1.0]))
    return straight + arc * perpendicular


def partial_overlap_route(
    rng: np.random.Generator, points: np.ndarray, keep_fraction: float = 0.5
) -> np.ndarray:
    """Follows the base route, then diverges."""
    n = len(points)
    split = max(2, int(n * keep_fraction))
    head = points[:split].copy()
    tail = make_route(rng, n - split + 1, origin=(float(head[-1, 0]), float(head[-1, 1])))
    return np.vstack([head, tail[1:]])


def detour_route(
    rng: np.random.Generator, points: np.ndarray, magnitude: float, length: int = 3
) -> np.ndarray:
    """The base route with a genuine excursion inserted in the middle.

    This is the acknowledged failure-mode fixture: an edit objective may erase
    a real detour cheaply, which is exactly what makes it a hard negative.
    """
    n = len(points)
    at = n // 2
    direction = np.array([float(rng.normal()), float(rng.normal())])
    norm = float(np.linalg.norm(direction)) or 1.0
    direction = direction / norm
    excursion = points[at] + magnitude * direction * np.linspace(0.4, 1.0, length).reshape(-1, 1)
    return np.vstack([points[:at], excursion, points[at:]])


def make_gallery(
    rng: np.random.Generator,
    n_families: int = 6,
    n_points: int = 40,
    corridor_offset: float = 30.0,
    bulge: float = 120.0,
    detour_magnitude: float = 90.0,
) -> list[Route]:
    """Build a gallery of ``n_families`` base routes plus hard negatives for each."""
    routes: list[Route] = []
    for f in range(n_families):
        family = f"fam{f:02d}"
        base = make_route(rng, n_points, origin=(float(rng.uniform(-500, 500)),
                                                 float(rng.uniform(-500, 500))))
        routes.append(Route(f"{family}-base", "base", family, base))
        routes.append(
            Route(f"{family}-corridor", "corridor", family, corridor_twin(base, corridor_offset))
        )
        routes.append(
            Route(
                f"{family}-endpoints",
                "shared_endpoint",
                family,
                shared_endpoint_route(rng, base, bulge),
            )
        )
        routes.append(
            Route(
                f"{family}-overlap",
                "partial_overlap",
                family,
                partial_overlap_route(rng, base),
            )
        )
        routes.append(
            Route(
                f"{family}-detour",
                "detour",
                family,
                detour_route(rng, base, detour_magnitude),
            )
        )
    return routes
