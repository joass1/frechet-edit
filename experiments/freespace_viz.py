"""Free-space diagram rendering for the continuous Fréchet distance.

The free-space diagram of two polygonal curves `A`, `B` and a threshold `eps` is
the subset of the parameter square where the curves are within `eps`:

    F = { (s, t) : || A(s) - B(t) || <= eps }

`d_F(A, B) <= eps` exactly when `F` contains a monotone path from the bottom
left corner to the top right. That picture is the whole Alt-Godau argument in
one image, which is why it is worth drawing.

Two honesty notes about what is plotted:

* The shaded region and the reachable overlay are computed on a GRID, so they
  are a rendering of the free space, not a decision procedure. The verdict
  printed on the figure comes from `baselines.continuous_frechet_le`, which is
  the exact Alt-Godau computation on cell boundaries. If a thin neck of free
  space falls between grid samples the picture may look disconnected while the
  exact verdict says feasible; the verdict is the one to trust.
* This draws the CONTINUOUS Fréchet distance. It is not a picture of the edit
  distance this package computes, which is discrete. Continuous edit variants
  are stretch item S2 and are not implemented.

Requires the `experiments` extra:  pip install -e ".[experiments]"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.baselines import continuous_frechet_le

# Plot styling. Kept here so the figures look like one deliberate set rather
# than nine different matplotlib defaults.
_FREE_COLOUR = "#f4f1ea"
_BLOCKED_COLOUR = "#2f3640"
_REACHABLE_COLOUR = "#4a9d7f"
_PATH_COLOUR = "#c9452e"
_CURVE_A_COLOUR = "#2f3640"
_CURVE_B_COLOUR = "#c9452e"


def _interpolate(curve: np.ndarray, samples: int) -> np.ndarray:
    """Sample a polygonal curve uniformly in its natural parameterisation."""
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2:
        raise ValueError("curve must be a (k, d) array")
    if len(curve) < 2:
        return np.repeat(curve, samples, axis=0)
    positions = np.linspace(0.0, len(curve) - 1.0, samples)
    lower = np.floor(positions).astype(int)
    lower = np.clip(lower, 0, len(curve) - 2)
    frac = (positions - lower)[:, None]
    return curve[lower] + frac * (curve[lower + 1] - curve[lower])


def free_space_grid(
    curve_a: np.ndarray, curve_b: np.ndarray, eps: float, resolution: int = 320
) -> np.ndarray:
    """Boolean grid, ``True`` where the curves are within ``eps``.

    Indexed ``[t, s]`` so that it can be handed to ``imshow`` with ``origin
    ="lower"`` and read with `A` along x and `B` along y.
    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    sampled_a = _interpolate(curve_a, resolution)
    sampled_b = _interpolate(curve_b, resolution)
    diff = sampled_b[:, None, :] - sampled_a[None, :, :]
    return np.linalg.norm(diff, axis=2) <= eps


def reachable_grid(free: np.ndarray) -> np.ndarray:
    """Cells of ``free`` reachable from the origin by a monotone path.

    A grid approximation of Alt-Godau reachability propagation: a cell is
    reachable when it is free and some predecessor to its left, below, or
    diagonally below-left is reachable.
    """
    reachable = np.zeros_like(free, dtype=bool)
    if not free[0, 0]:
        return reachable
    reachable[0, 0] = True
    rows, cols = free.shape
    for t in range(rows):
        for s in range(cols):
            if not free[t, s] or (t == 0 and s == 0):
                continue
            if (
                (s > 0 and reachable[t, s - 1])
                or (t > 0 and reachable[t - 1, s])
                or (t > 0 and s > 0 and reachable[t - 1, s - 1])
            ):
                reachable[t, s] = True
    return reachable


def monotone_path(reachable: np.ndarray) -> np.ndarray | None:
    """One monotone path through the reachable set, or ``None`` if the top
    right corner cannot be reached. Walks back greedily from the corner."""
    rows, cols = reachable.shape
    if not reachable[rows - 1, cols - 1]:
        return None
    t, s = rows - 1, cols - 1
    path = [(s, t)]
    while (t, s) != (0, 0):
        if t > 0 and s > 0 and reachable[t - 1, s - 1]:
            t, s = t - 1, s - 1
        elif s > 0 and reachable[t, s - 1]:
            s -= 1
        elif t > 0 and reachable[t - 1, s]:
            t -= 1
        else:  # pragma: no cover - unreachable while reachable[] is consistent
            break
        path.append((s, t))
    return np.array(path[::-1], dtype=float)


def plot_free_space_diagram(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    eps: float,
    resolution: int = 320,
    title: str | None = None,
) -> Any:
    """Render the diagram beside the two curves. Returns a matplotlib Figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            'matplotlib is required for plotting. Install with: pip install -e ".[experiments]"'
        ) from exc

    curve_a = np.asarray(curve_a, dtype=np.float64)
    curve_b = np.asarray(curve_b, dtype=np.float64)
    free = free_space_grid(curve_a, curve_b, eps, resolution)
    reachable = reachable_grid(free)
    path = monotone_path(reachable)
    feasible = continuous_frechet_le(curve_a, curve_b, eps)

    fig, (ax_curves, ax_diagram) = plt.subplots(
        1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # A is drawn as a wide underlay and B thin on top, so that curves which
    # coincide except at a few vertices still read as two curves.
    ax_curves.plot(curve_a[:, 0], curve_a[:, 1], "-", color=_CURVE_A_COLOUR,
                   linewidth=3.4, alpha=0.85, label="A", zorder=1)
    ax_curves.plot(curve_b[:, 0], curve_b[:, 1], "-o", color=_CURVE_B_COLOUR,
                   markersize=2.6, linewidth=1.4, label="B", zorder=2)
    ax_curves.set_aspect("equal", adjustable="datalim")
    ax_curves.legend(frameon=False, loc="best")
    ax_curves.set_title("The two curves", loc="left", fontsize=11)
    ax_curves.spines[["top", "right"]].set_visible(False)

    ax_diagram.imshow(
        free, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
        cmap=_binary_cmap(plt), interpolation="nearest",
    )
    masked = np.ma.masked_where(~reachable, reachable.astype(float))
    ax_diagram.imshow(
        masked, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
        cmap=_solid_cmap(plt, _REACHABLE_COLOUR), alpha=0.55,
        interpolation="nearest", vmin=0, vmax=1,
    )
    if path is not None:
        scale = 1.0 / (resolution - 1)
        ax_diagram.plot(path[:, 0] * scale, path[:, 1] * scale,
                        color=_PATH_COLOUR, linewidth=2.2,
                        label="monotone path")
        ax_diagram.legend(frameon=False, loc="lower right")

    verdict = "reachable" if feasible else "no monotone path"
    ax_diagram.set_xlabel("parameter along A")
    ax_diagram.set_ylabel("parameter along B")
    ax_diagram.set_title(
        f"Free space at eps = {eps:g}  -  exact verdict: {verdict}",
        loc="left", fontsize=11,
    )

    fig.suptitle(title or "Continuous Fréchet free-space diagram",
                 fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _binary_cmap(plt: Any) -> Any:
    from matplotlib.colors import ListedColormap

    return ListedColormap([_BLOCKED_COLOUR, _FREE_COLOUR])


def _solid_cmap(plt: Any, colour: str) -> Any:
    from matplotlib.colors import ListedColormap

    return ListedColormap([colour])


def save_example_figures(out_dir: Path, resolution: int = 320) -> list[Path]:
    """Write a small set of illustrative diagrams. Synthetic curves only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    t = np.linspace(0, 2 * np.pi, 60)
    base = np.column_stack([t, np.sin(t)])

    cases = {
        "feasible": (base, np.column_stack([t, np.sin(t) + 0.25]), 0.5,
                     "Nearby curves: the free space connects corner to corner"),
        "infeasible": (base, np.column_stack([t, np.sin(t) + 1.6]), 0.5,
                       "Far apart: free space exists but no monotone path"),
        "spike": (base, _with_spike(base), 0.5,
                  "One outlier vertex: a single spike blocks the diagonal"),
    }
    for name, (a, b, eps, title) in cases.items():
        fig = plot_free_space_diagram(a, b, eps, resolution=resolution, title=title)
        path = out_dir / f"freespace_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _with_spike(curve: np.ndarray) -> np.ndarray:
    spiked = curve.copy()
    spiked[len(spiked) // 2, 1] += 3.0
    return spiked


def main() -> int:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/freespace"))
    parser.add_argument("--resolution", type=int, default=320)
    args = parser.parse_args()
    written = save_example_figures(args.out_dir, args.resolution)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
