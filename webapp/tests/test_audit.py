"""The course audit on scenarios whose answers are known by construction."""

from __future__ import annotations

import pytest

from webapp import scenarios
from webapp.audit import AuditInputError, AuditLimits, run_audit
from webapp.geo import Fix, GeoInputError


@pytest.fixture(scope="module")
def audits():
    out = {}
    for sid in scenarios.BUILDERS:
        sc = scenarios.get(sid)
        out[sid] = (sc, run_audit(sc.course, sc.track, sc.delta_m))
    return out


class TestScenarioVerdicts:
    def test_clean_run_is_on_course_with_nothing_removed(self, audits):
        _, result = audits["clean"]
        assert result["verdict"]["code"] == "on_course"
        assert result["glitches"] == []
        assert result["verification"]["status"] == "verified"

    def test_urban_canyon_removes_exactly_the_injected_spikes(self, audits):
        sc, result = audits["urban-canyon"]
        assert result["verdict"]["code"] == "on_course_after_cleaning"
        assert sorted(g["index"] for g in result["glitches"]) == sorted(sc.injected)
        assert result["checks"]["continuous_edit_distance"] == len(sc.injected)
        # The cleaned track is the original minus exactly those fixes.
        assert len(result["cleaned_track"]) == len(sc.track) - len(sc.injected)
        assert result["verification"]["status"] == "verified"

    def test_sparse_logging_defeats_discrete_but_not_continuous(self, audits):
        _, result = audits["sparse-logging"]
        assert result["verdict"]["code"] == "on_course"
        assert result["checks"]["discrete_status"] == "infeasible"
        assert result["checks"]["continuous_edit_distance"] == 0

    def test_a_shortcut_is_off_course_and_localised(self, audits):
        _, result = audits["shortcut"]
        assert result["verdict"]["code"] == "off_course"
        assert result["checks"]["course_covered"] is False
        assert result["uncovered_course"], "the skipped course points should be reported"
        assert result["off_corridor"], "the cut-across fixes should be reported"
        assert result["cleaned_track"] is None

    def test_one_lap_of_two_passes_both_naive_checks_and_still_fails(self, audits):
        """The reason to use an order-aware measure at all."""
        _, result = audits["one-lap-of-two"]
        checks = result["checks"]
        assert checks["fixes_near_course"] is True
        assert checks["course_covered"] is True
        assert result["verdict"]["code"] == "off_course"
        assert "order" in result["verdict"]["detail"]

    def test_the_continuous_engine_never_needs_more_than_the_discrete_one(self, audits):
        for _, result in audits.values():
            cont = result["checks"]["continuous_edit_distance"]
            disc = result["checks"]["discrete_edit_distance"]
            if cont is not None and disc is not None:
                assert cont <= disc


class TestLimitsAndInputs:
    def test_tolerance_bounds(self):
        sc = scenarios.get("clean")
        for bad in (0.5, 501.0, "25"):
            with pytest.raises(AuditInputError):
                run_audit(sc.course, sc.track, bad)

    def test_decimation_is_disclosed_and_indices_stay_original(self):
        sc = scenarios.get("urban-canyon")
        result = run_audit(sc.course, sc.track, sc.delta_m, AuditLimits(max_track_points=200))
        proc = result["processing"]
        assert proc["track_points_analysed"] <= 200
        assert proc["track_decimation"] > 1
        assert all(0 <= g["index"] < len(sc.track) for g in result["glitches"])
        assert set(result["analysed_indices"]) >= {g["index"] for g in result["glitches"]}

    def test_a_densely_recorded_course_is_simplified_and_that_is_disclosed(self):
        """A recorded GPX used as the course has hundreds of points; it is
        reduced by Douglas-Peucker within delta/4 and the tolerance reported."""
        sc = scenarios.get("clean")
        dense = scenarios._fixes(scenarios._resample(scenarios._course_xy(), 4.0), 1.0)
        result = run_audit(dense, sc.track, sc.delta_m, AuditLimits(max_course_points=40))
        proc = result["processing"]
        assert proc["course_points"] == len(dense) > 500
        assert proc["course_points_analysed"] <= 40
        assert 0 < proc["course_simplified_m"] <= sc.delta_m / 4
        assert result["verdict"]["code"] == "on_course"

    def test_a_course_too_intricate_to_simplify_is_refused_not_distorted(self):
        sc = scenarios.get("clean")
        with pytest.raises(GeoInputError, match="cannot be simplified"):
            run_audit(sc.course, sc.track, sc.delta_m, AuditLimits(max_course_points=8))

    def test_large_inputs_skip_in_request_verification_honestly(self):
        sc = scenarios.get("clean")
        result = run_audit(sc.course, sc.track, sc.delta_m, AuditLimits(verify_cells=10))
        assert result["verification"]["status"] == "skipped"

    def test_bad_geometry_is_a_user_error(self):
        with pytest.raises(GeoInputError):
            run_audit([Fix(0.0, 0.0)], [Fix(0.0, 0.0), Fix(0.0, 0.001)], 25.0)

    def test_the_glitch_budget_bounds_the_verdict(self):
        sc = scenarios.get("urban-canyon")
        result = run_audit(sc.course, sc.track, sc.delta_m, AuditLimits(max_glitches=3))
        assert result["verdict"]["code"] == "off_course"
        assert result["checks"]["continuous_status"] == "budget_exceeded"
        assert "more than 3" in result["verdict"]["detail"]
