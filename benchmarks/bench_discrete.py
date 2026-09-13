"""Timing benchmark for the discrete backends.

Reports p50/p95 wall time per configuration together with the environment,
seeds and the instrumentation counters the complexity claims rest on
(enclosing-ball calls, minimum-queue operations, DP states).

A timing plot is not a proof of an asymptotic bound. What is reported here is
measurement; the structural argument lives in ``docs/recurrences.md`` section 6.

    python benchmarks/bench_discrete.py --sizes 50 100 200 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import frechet_edit
from frechet_edit import discrete_edit_distance


def make_route(rng: np.random.Generator, n: int, step: float = 1.0) -> np.ndarray:
    """A random walk with persistence, standing in for a sampled trajectory."""
    heading = rng.uniform(0, 2 * np.pi)
    points = np.zeros((n, 2))
    for i in range(1, n):
        heading += rng.normal(0, 0.25)
        points[i] = points[i - 1] + step * np.array([np.cos(heading), np.sin(heading)])
    return points


def corrupt(rng: np.random.Generator, curve: np.ndarray, rate: float) -> np.ndarray:
    out = curve.copy()
    k = max(1, int(len(curve) * rate))
    idx = rng.choice(len(curve), size=k, replace=False)
    out[idx] += rng.normal(0, 25.0, size=(k, 2))
    return out


@dataclass
class Row:
    mode: str
    backend: str
    m: int
    n: int
    delta: float
    p50_ms: float
    p95_ms: float
    cost: float
    states: int
    enclosing_ball_calls: int
    exact_fallbacks: int
    minqueue_pushes: int
    minqueue_pops: int


def bench(sizes: list[int], repeats: int, seed: int) -> list[Row]:
    rng = np.random.default_rng(seed)
    rows: list[Row] = []
    for size in sizes:
        reference = make_route(rng, size)
        observation = corrupt(rng, make_route(rng, size), 0.05)
        for delta in (2.0, 10.0):
            for mode in ("delete", "insert", "both"):
                for backend in ("python", "reference"):
                    times = []
                    result = None
                    for _ in range(repeats):
                        start = time.perf_counter()
                        result = discrete_edit_distance(
                            reference, observation, delta,
                            operations=mode, backend=backend,
                        )
                        times.append((time.perf_counter() - start) * 1000.0)
                    assert result is not None
                    stats = result.stats
                    rows.append(
                        Row(
                            mode=mode,
                            backend=backend,
                            m=size,
                            n=size,
                            delta=delta,
                            p50_ms=round(statistics.median(times), 3),
                            p95_ms=round(
                                sorted(times)[min(len(times) - 1, int(0.95 * len(times)))], 3
                            ),
                            cost=float(result.cost) if result.cost is not None else float("nan"),
                            states=int(stats.get("states", 0)),
                            enclosing_ball_calls=int(stats.get("enclosing_ball_calls", 0)),
                            exact_fallbacks=int(stats.get("exact_fallbacks", 0)),
                            minqueue_pushes=int(stats.get("minqueue_pushes", 0)),
                            minqueue_pops=int(stats.get("minqueue_pops", 0)),
                        )
                    )
    return rows


def environment(seed: int, repeats: int) -> dict:
    return {
        "frechet_edit": frechet_edit.__version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "seed": seed,
        "repeats": repeats,
        "state": "warm (same process, repeated calls)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20240612)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = bench(args.sizes, args.repeats, args.seed)
    payload = {
        "environment": environment(args.seed, args.repeats),
        "rows": [asdict(r) for r in rows],
    }

    header = (
        f"{'mode':7} {'backend':10} {'m=n':>5} {'delta':>6} {'p50 ms':>9} "
        f"{'p95 ms':>9} {'cost':>6} {'balls':>7} {'exact':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.mode:7} {r.backend:10} {r.m:5d} {r.delta:6.1f} {r.p50_ms:9.2f} "
            f"{r.p95_ms:9.2f} {r.cost:6.0f} {r.enclosing_ball_calls:7d} {r.exact_fallbacks:6d}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
