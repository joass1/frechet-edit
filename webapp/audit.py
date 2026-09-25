"""Course audit: did this GPS track follow that course, and which fixes are glitches?

This is the application layer over :func:`frechet_edit.continuous_edit_distance`
(deletion-only continuous Frechet edit distance, Fox et al., SoCG 2024,
Theorem 3). The question it answers:

    What is the fewest GPS fixes that must be discarded so that the rest of
    the track stays within ``delta`` metres of the course - continuously, and
    IN ORDER, from start to finish?

* 0 means the track follows the course as recorded.
* A small number means it follows the course once those fixes (glitches) are
  discarded, and the witness names them.
* No solution within the glitch budget means the track genuinely departs from
  the course: a shortcut, a skipped section, a missing lap.

The naive checks shown alongside - "every fix is near the course" and "every
part of the course is near the track" - are what a Hausdorff-style comparison
gives. They are order-blind, and the one-lap-of-two scenario passes both.

Honest limits, surfaced in every response: inputs may be simplified or
decimated to bound the running time, and the verdict inherits the accuracy of
the local projection and of ``delta`` itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from frechet_edit import (
    Deletion,
    continuous_edit_distance,
    discrete_edit_distance,
    verify_witness,
)

from .geo import (
    Fix,
    decimate,
    frame_for,
    point_to_polyline,
    projection_scale_error,
    simplify_to,
    validate_fixes,
)

MIN_DELTA_M = 1.0
MAX_DELTA_M = 500.0


@dataclass(frozen=True)
class AuditLimits:
    """Bounds that keep one audit to a few seconds on a laptop.

    The solver is ``O(k^2 m n)``; these cap ``m`` (course vertices after
    simplification), ``n`` (fixes analysed) and ``k`` (glitch budget).
    """

    max_course_points: int = 150
    max_track_points: int = 600
    max_glitches: int = 25
    #: Independent exact verification is run when m * n' is at most this.
    verify_cells: int = 40_000


class AuditInputError(ValueError):
    """Bad audit parameters. The message is safe to show verbatim."""


def _latlon_list(fixes: list[Fix]) -> list[list[float]]:
    return [[f.lat, f.lon] for f in fixes]


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of True values; (0, 0) if none."""
    best_start, best_len, start = 0, 0, None
    for i, flag in enumerate([*mask.tolist(), False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    return best_start, best_len


def run_audit(
    course: list[Fix],
    track: list[Fix],
    delta_m: float,
    limits: AuditLimits | None = None,
) -> dict[str, Any]:
    """Audit ``track`` against ``course`` with corridor half-width ``delta_m`` metres."""
    limits = limits or AuditLimits()
    if not isinstance(delta_m, (int, float)) or not MIN_DELTA_M <= float(delta_m) <= MAX_DELTA_M:
        raise AuditInputError(
            f"tolerance must be between {MIN_DELTA_M:g} and {MAX_DELTA_M:g} metres"
        )
    delta = float(delta_m)
    validate_fixes(course, "course")
    validate_fixes(track, "track")
    started = time.perf_counter()

    frame = frame_for(course, track)
    course_all = frame.to_xy(np.array([f.lat for f in course]), np.array([f.lon for f in course]))
    track_all = frame.to_xy(np.array([f.lat for f in track]), np.array([f.lon for f in track]))

    kept_course, tolerance = simplify_to(
        course_all, limits.max_course_points, max_tolerance=delta / 4.0
    )
    course_xy = course_all[kept_course]
    track_idx = decimate(len(track), limits.max_track_points)
    track_xy = track_all[track_idx]
    budget = min(limits.max_glitches, max(1, len(track_idx) // 4))

    solve_start = time.perf_counter()
    result = continuous_edit_distance(
        course_xy, track_xy, delta, return_witness=True, max_deletions=budget
    )
    solve_ms = (time.perf_counter() - solve_start) * 1000.0

    discrete = discrete_edit_distance(course_xy, track_xy, delta, operations="delete")
    offsets = point_to_polyline(track_xy, course_xy)
    coverage = point_to_polyline(course_xy, track_xy)

    glitch_local = [e.index for e in result.edits or () if isinstance(e, Deletion)]
    glitches = [
        {
            "index": int(track_idx[i]),
            "lat": track[int(track_idx[i])].lat,
            "lon": track[int(track_idx[i])].lon,
            "time": track[int(track_idx[i])].time,
            "offset_m": round(float(offsets[i]), 1),
        }
        for i in glitch_local
    ]

    verification = _verify(course_xy, track_xy, result, limits)
    verdict = _verdict(result, delta, budget, offsets, coverage, course_xy, track_xy)
    discarded = set(glitch_local)
    cleaned = [
        [track[int(j)].lat, track[int(j)].lon, track[int(j)].time]
        for i, j in enumerate(track_idx)
        if i not in discarded
    ]

    uncovered = np.flatnonzero(coverage > delta)
    course_simplified_latlon = frame.to_latlon(course_xy)
    return {
        "verdict": verdict,
        "delta_m": delta,
        "glitches": glitches,
        "cleaned_track": cleaned if result.status == "optimal" else None,
        "course": _latlon_list(course),
        "track": [[f.lat, f.lon] for f in track],
        "analysed_indices": [int(i) for i in track_idx],
        "offsets_m": [round(float(x), 2) for x in offsets],
        "off_corridor": [int(track_idx[i]) for i in np.flatnonzero(offsets > delta)],
        "uncovered_course": [
            [float(a), float(b)] for a, b in course_simplified_latlon[uncovered]
        ],
        "checks": {
            "fixes_near_course": bool(offsets.max() <= delta),
            "max_offset_m": round(float(offsets.max()), 1),
            "course_covered": bool(coverage.max() <= delta),
            "max_uncovered_m": round(float(coverage.max()), 1),
            "ordinary_frechet_within": result.status == "optimal" and result.cost == 0,
            "continuous_edit_distance": result.cost if result.status == "optimal" else None,
            "continuous_status": result.status,
            "discrete_edit_distance": discrete.cost if discrete.status == "optimal" else None,
            "discrete_status": discrete.status,
            "glitch_budget": budget,
        },
        "verification": verification,
        "processing": {
            "course_points": len(course),
            "course_points_analysed": len(course_xy),
            "course_simplified_m": round(tolerance, 2),
            "track_points": len(track),
            "track_points_analysed": len(track_idx),
            "track_decimation": int(track_idx[1] - track_idx[0]) if len(track_idx) > 1 else 1,
            "projection_scale_error": projection_scale_error(
                frame, np.array([f.lat for f in course + track])
            ),
            "solver_ms": round(solve_ms, 1),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "budgets_tried": result.stats.get("budgets_tried"),
        },
    }


def _verify(
    course_xy: np.ndarray, track_xy: np.ndarray, result: Any, limits: AuditLimits
) -> dict[str, Any]:
    if result.status != "optimal" or result.witness_status != "certified":
        return {
            "status": "not_applicable",
            "detail": "the verdict is negative, so there is no cleaned track to re-check",
        }
    cells = len(course_xy) * (len(track_xy) - int(result.cost))
    if cells > limits.verify_cells:
        return {
            "status": "skipped",
            "detail": f"{cells} free-space cells exceed the in-request budget of "
            f"{limits.verify_cells}; verify offline with frechet_edit.verify_witness",
        }
    started = time.perf_counter()
    report = verify_witness(course_xy, track_xy, result)
    elapsed = (time.perf_counter() - started) * 1000.0
    if report.ok:
        return {
            "status": "verified",
            "detail": "an independent exact-arithmetic check confirmed the cleaned track "
            "is within the tolerance",
            "ms": round(elapsed, 1),
        }
    return {"status": "failed", "detail": "; ".join(report.violations), "ms": round(elapsed, 1)}


def _verdict(
    result: Any,
    delta: float,
    budget: int,
    offsets: np.ndarray,
    coverage: np.ndarray,
    course_xy: np.ndarray,
    track_xy: np.ndarray,
) -> dict[str, Any]:
    if result.status == "optimal" and result.cost == 0:
        return {
            "code": "on_course",
            "headline": "On course",
            "detail": f"The track follows the course within ±{delta:g} m from start to "
            "finish, in order. No fixes needed removing.",
        }
    if result.status == "optimal":
        noun = "glitch" if result.cost == 1 else "glitches"
        return {
            "code": "on_course_after_cleaning",
            "headline": f"On course · {result.cost} GPS {noun} removed",
            "detail": f"Discarding {result.cost} of {len(track_xy)} fixes makes the track "
            f"follow the course within ±{delta:g} m, in order. No smaller set works. "
            "The discarded fixes are listed below.",
        }

    reasons: list[str] = []
    far_course = int((coverage > delta).sum())
    if far_course:
        reasons.append(
            f"{far_course} course point(s) are never approached within {delta:g} m: a "
            "section was skipped or cut."
        )
    start, length = _longest_run(offsets > delta)
    if length > 1:
        seg = track_xy[start : start + length]
        span = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if length > 1 else 0.0
        reasons.append(
            f"The track leaves the ±{delta:g} m corridor for {length} consecutive fixes "
            f"(about {span:.0f} m), which is a route, not a glitch."
        )
    if not reasons:
        reasons.append(
            "Every fix is near the course and every part of the course is visited, but "
            "not in the right order or the right number of times - for example a missing "
            "lap or a section run backwards. Order-blind checks cannot see this."
        )
    beyond = (
        "No set of deletions at all makes the track fit."
        if result.status == "infeasible"
        else f"Fitting it would take more than {budget} discarded fixes."
    )
    return {
        "code": "off_course",
        "headline": "Off course",
        "detail": " ".join([*reasons, beyond]),
    }
