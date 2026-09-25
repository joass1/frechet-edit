"""Continuous deletion vs discrete deletion on GPS-like inputs.

A sparse reference route (m vertices) against a dense noisy observation of the
same path (n fixes) with `spikes` injected outliers. Reports each measure's
cost and wall time. Single machine, single run per row: a measurement, not a
guarantee.

    python benchmarks/bench_continuous.py
"""

from __future__ import annotations

import time

import numpy as np

from frechet_edit import continuous_edit_distance, discrete_edit_distance

DELTA = 25.0
ROWS = [(20, 100, 2), (30, 200, 3), (40, 300, 5), (60, 400, 8), (60, 600, 10)]


def scenario(m: int, n: int, spikes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, m)
    ref = np.column_stack([5000 * t, 400 * np.sin(6 * t) + 150 * np.sin(17 * t)])
    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n)
    obs = np.column_stack([np.interp(s, cum, ref[:, 0]), np.interp(s, cum, ref[:, 1])])
    obs += rng.normal(0, 3.0, obs.shape)
    idx = rng.choice(np.arange(2, n - 2), spikes, replace=False)
    obs[idx] += rng.normal(0, 150, (spikes, 2))
    return ref, obs


def main() -> None:
    print(f"{'m':>4} {'n':>5} {'spikes':>6} | {'discrete cost':>13} {'time':>7} |"
          f" {'continuous cost':>15} {'time':>7}  budgets")
    for m, n, spikes in ROWS:
        ref, obs = scenario(m, n, spikes)
        t0 = time.perf_counter()
        disc = discrete_edit_distance(ref, obs, DELTA, operations="delete")
        t1 = time.perf_counter()
        cont = continuous_edit_distance(ref, obs, DELTA, return_witness=True)
        t2 = time.perf_counter()
        print(f"{m:>4} {n:>5} {spikes:>6} | {disc.cost!s:>13} {t1 - t0:>6.2f}s |"
              f" {cont.cost!s:>15} {t2 - t1:>6.2f}s  {cont.stats['budgets_tried']}")


if __name__ == "__main__":
    main()
