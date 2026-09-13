"""Memory benchmark: check the claimed storage shape, not just the speed.

The design claim from ``docs/recurrences.md`` section 6 is:

* score-only queries use ``O(m)`` working storage and never build a dense
  ``m x n`` distance matrix;
* witness queries additionally pay ``O(mn)`` for the parent tables.

This script measures peak allocation with ``tracemalloc`` for a fixed ``m`` and
a growing ``n``. If score-only storage were secretly quadratic, the measured
peak would grow with ``n``; if it is linear in ``m`` only, the peak stays flat.

    python benchmarks/bench_memory.py
"""

from __future__ import annotations

import argparse
import tracemalloc

import numpy as np

from frechet_edit import discrete_edit_distance


def make_route(rng: np.random.Generator, n: int) -> np.ndarray:
    heading = rng.uniform(0, 2 * np.pi)
    pts = np.zeros((n, 2))
    for i in range(1, n):
        heading += rng.normal(0, 0.25)
        pts[i] = pts[i - 1] + np.array([np.cos(heading), np.sin(heading)])
    return pts


def peak_kib(reference, observation, delta, mode, witness) -> float:
    tracemalloc.start()
    tracemalloc.reset_peak()
    discrete_edit_distance(
        reference, observation, delta, operations=mode, return_witness=witness
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=120)
    parser.add_argument("--ns", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20240612)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    reference = make_route(rng, args.m)

    print(f"reference length m = {args.m}, delta = {args.delta}, mode = both\n")
    print(f"{'n':>6} {'score-only KiB':>16} {'witness KiB':>14} {'ratio':>8}")
    print("-" * 48)
    score_peaks = []
    for n in args.ns:
        observation = make_route(rng, n)
        score = peak_kib(reference, observation, args.delta, "both", False)
        wit = peak_kib(reference, observation, args.delta, "both", True)
        score_peaks.append(score)
        print(f"{n:6d} {score:16.1f} {wit:14.1f} {wit / max(score, 1e-9):8.1f}x")

    # The first rows are dominated by one-time allocations (module import,
    # numpy warm-up), so the trend is read from the LARGEST sizes only.
    tail = score_peaks[-3:] if len(score_peaks) >= 3 else score_peaks
    tail_ns = args.ns[-len(tail) :]
    growth = max(tail) / max(min(tail), 1e-9)
    print(
        f"\nOver the largest sizes n={tail_ns}, the score-only peak moved "
        f"{growth:.2f}x while n grew {tail_ns[-1] / tail_ns[0]:.0f}x."
    )
    print(
        "A quadratic working set would track the n growth. A flat number is\n"
        "consistent with the O(m) rolling-column design, and the witness column\n"
        "shows the O(mn) parent tables it deliberately trades for. Small-n rows\n"
        "are warm-up dominated and are excluded from the trend. This is a\n"
        "measurement, not a proof."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
