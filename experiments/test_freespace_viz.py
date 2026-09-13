"""Tests for free-space diagram rendering.

The geometry is tested directly rather than through the picture: a rendered
figure is hard to assert on, but the free-space grid, the reachability
propagation and the extracted path are ordinary data structures with properties
that must hold. The plotting call itself is smoke-tested to catch API drift.

The key correctness property is the one the diagram exists to illustrate:
the top-right corner is reachable exactly when the continuous Fréchet distance
is within eps. That is cross-checked against the independent Alt-Godau decision
procedure in `baselines`, at a resolution fine enough that grid effects do not
dominate.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.baselines import continuous_frechet_le
from experiments.freespace_viz import (
    free_space_grid,
    monotone_path,
    reachable_grid,
)

pytest.importorskip("matplotlib", reason="plotting requires the experiments extra")


def _sine(offset: float = 0.0, n: int = 60) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([t, np.sin(t) + offset])


class TestFreeSpaceGrid:
    def test_identical_curves_are_free_on_the_diagonal(self):
        curve = _sine()
        free = free_space_grid(curve, curve, eps=0.01, resolution=64)
        assert np.all(np.diag(free))

    def test_a_huge_eps_frees_the_whole_square(self):
        free = free_space_grid(_sine(), _sine(5.0), eps=1000.0, resolution=32)
        assert free.all()

    def test_a_zero_eps_on_disjoint_curves_frees_nothing(self):
        free = free_space_grid(_sine(), _sine(10.0), eps=0.0, resolution=32)
        assert not free.any()

    def test_grid_has_the_requested_shape(self):
        assert free_space_grid(_sine(), _sine(), 0.5, resolution=48).shape == (48, 48)

    def test_negative_eps_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            free_space_grid(_sine(), _sine(), -1.0)

    def test_degenerate_resolution_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            free_space_grid(_sine(), _sine(), 1.0, resolution=1)


class TestReachability:
    def test_reachable_is_always_a_subset_of_free(self):
        free = free_space_grid(_sine(), _sine(0.3), eps=0.5, resolution=64)
        assert not (reachable_grid(free) & ~free).any()

    def test_a_blocked_origin_makes_everything_unreachable(self):
        free = np.ones((8, 8), dtype=bool)
        free[0, 0] = False
        assert not reachable_grid(free).any()

    def test_a_severed_band_does_not_reach_the_corner(self):
        free = np.zeros((16, 16), dtype=bool)
        np.fill_diagonal(free, True)
        free[8, 8] = False  # cut the only route
        reachable = reachable_grid(free)
        assert not reachable[-1, -1]

    def test_path_is_none_when_the_corner_is_unreachable(self):
        free = np.zeros((8, 8), dtype=bool)
        free[0, 0] = True
        assert monotone_path(reachable_grid(free)) is None


class TestPathIsMonotone:
    def test_extracted_path_never_steps_backwards(self):
        free = free_space_grid(_sine(), _sine(0.3), eps=0.5, resolution=64)
        path = monotone_path(reachable_grid(free))
        assert path is not None
        steps = np.diff(path, axis=0)
        assert (steps >= 0).all(), "a monotone path may not decrease in either axis"

    def test_path_spans_corner_to_corner(self):
        free = free_space_grid(_sine(), _sine(0.3), eps=0.5, resolution=64)
        path = monotone_path(reachable_grid(free))
        assert tuple(path[0]) == (0.0, 0.0)
        assert tuple(path[-1]) == (63.0, 63.0)


class TestAgreesWithTheExactDecision:
    """The picture must tell the same story as the Alt-Godau procedure."""

    @pytest.mark.parametrize(
        "offset,eps",
        [(0.2, 0.5), (0.3, 0.5), (1.6, 0.5), (0.0, 0.01), (2.0, 3.0)],
    )
    def test_corner_reachability_matches_continuous_frechet(self, offset, eps):
        a, b = _sine(), _sine(offset)
        free = free_space_grid(a, b, eps, resolution=256)
        grid_says = bool(reachable_grid(free)[-1, -1])
        assert grid_says == continuous_frechet_le(a, b, eps)


class TestPlotting:
    def test_returns_a_figure_without_raising(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from experiments.freespace_viz import plot_free_space_diagram

        fig = plot_free_space_diagram(_sine(), _sine(0.3), 0.5, resolution=64)
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_writes_example_files(self, tmp_path):
        from experiments.freespace_viz import save_example_figures

        written = save_example_figures(tmp_path, resolution=64)
        assert len(written) == 3
        for path in written:
            assert path.exists() and path.stat().st_size > 0
