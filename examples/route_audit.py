"""Route-template audit: rank candidate routes and show what was repaired.

The analyst's question is not "how far apart are these curves" but "which
template is this trace, and which observations had to be thrown away to make
that fit". This example answers both.

It is deliberately SELF-CONTAINED: it imports only ``numpy`` and the installed
``frechet_edit`` package, so it runs from a wheel outside this checkout.

    python examples/route_audit.py

IMPORTANT: the routes here are GENERATED, not measured. Inserted points are
geometric hypotheses - they are not recovered GPS measurements, they are not
evidence of road legality, and no timestamps are inferred. See
docs/limitations.md.
"""

from __future__ import annotations

import argparse

import numpy as np

from frechet_edit import (
    Deletion,
    Insertion,
    discrete_edit_distance,
    ordinary_discrete_frechet,
    verify_witness,
)

DELTA = 25.0  # metres


def make_route(rng: np.random.Generator, n: int = 30, step: float = 10.0) -> np.ndarray:
    """A persistent random walk standing in for a sampled route, in metres."""
    heading = float(rng.uniform(0.0, 2.0 * np.pi))
    points = np.zeros((n, 2), dtype=np.float64)
    for i in range(1, n):
        heading += float(rng.normal(0.0, 0.22))
        points[i] = points[i - 1] + step * np.array([np.cos(heading), np.sin(heading)])
    return points


def corridor_twin(points: np.ndarray, offset: float) -> np.ndarray:
    """A hard negative: a parallel route a fixed distance away."""
    tangents = np.gradient(points, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    unit = tangents / lengths
    normals = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    return points + offset * normals


def shared_endpoints(rng: np.random.Generator, points: np.ndarray, bulge: float) -> np.ndarray:
    """A hard negative: same start and end, different path between."""
    n = len(points)
    start, end = points[0], points[-1]
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    direction = end - start
    norm = float(np.linalg.norm(direction)) or 1.0
    perp = np.array([-direction[1], direction[0]]) / norm
    return start + t * direction + np.sin(np.pi * t) * bulge * perp


def add_spikes(
    rng: np.random.Generator, curve: np.ndarray, count: int, magnitude: float
) -> tuple[np.ndarray, list[int]]:
    """Displace ``count`` interior vertices by exactly ``magnitude`` metres."""
    out = curve.copy()
    pool = np.arange(1, len(curve) - 1)
    chosen = sorted(int(i) for i in rng.choice(pool, size=count, replace=False))
    for idx in chosen:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        out[idx] = out[idx] + magnitude * np.array([np.cos(angle), np.sin(angle)])
    return out, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20240612)
    parser.add_argument("--delta", type=float, default=DELTA)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    truth = make_route(rng, 30)
    gallery = {
        "route-A (truth)": truth,
        "route-A-corridor": corridor_twin(truth, 30.0),
        "route-A-endpoints": shared_endpoints(rng, truth, 120.0),
        "route-B": make_route(rng, 30),
        "route-C": make_route(rng, 30),
    }

    query, injected = add_spikes(rng, truth, 2, 150.0)
    query = query + rng.normal(0.0, 2.0, size=query.shape)  # background jitter

    print("ROUTE AUDIT (synthetic data - not a real trajectory)")
    print("=" * 66)
    print(f"delta              : {args.delta} m")
    print(f"vertices corrupted : {injected}")
    print(f"gallery            : {len(gallery)} templates, including hard negatives\n")

    rows = []
    for name, template in gallery.items():
        result = discrete_edit_distance(
            template, query, args.delta, operations="both", return_witness=True
        )
        rows.append((name, template, result, ordinary_discrete_frechet(template, query)))

    def rank_key(row):
        cost = row[2].cost
        return float("inf") if cost is None else float(cost)

    print(f"{'template':20} {'FED':>6} {'ordinary Frechet (m)':>22}")
    print("-" * 50)
    for name, _, result, ordinary in sorted(rows, key=rank_key):
        cost = "amb" if result.cost is None else str(result.cost)
        print(f"{name:20} {cost:>6} {ordinary:22.1f}")

    best_name, best_template, best_result, _ = min(rows, key=rank_key)
    frechet_first = min(rows, key=lambda r: r[3])[0]

    print(f"\nFED ranks first              : {best_name}")
    print(f"ordinary Frechet ranks first : {frechet_first}")

    tied = [name for name, _, result, _ in rows if result.cost == best_result.cost]
    if len(tied) > 1:
        print(
            f"\nNOTE: {len(tied)} templates tie at cost {best_result.cost} "
            f"({', '.join(tied)}). An integer edit count ties easily, and a tie "
            "is not a retrieval success."
        )

    print(f"\nREPAIR for {best_name}")
    print("-" * 66)
    if best_result.witness_status != "certified":
        print(f"  witness unavailable: {best_result.detail}")
        return

    deletions = [e for e in best_result.edits if isinstance(e, Deletion)]
    insertions = [e for e in best_result.edits if isinstance(e, Insertion)]
    print(f"  total edits : {best_result.cost}")
    print(f"  deleted     : {[e.index for e in deletions]} (original query indices)")
    for ins in insertions:
        print(f"  inserted    : {[round(c, 2) for c in ins.point]} in gap {ins.gap}")
    if insertions:
        print("                ^ geometric hypotheses, NOT recovered measurements")

    report = verify_witness(best_template, query, best_result)
    print(f"  verified    : {report.ok}")
    print(
        f"  residual ordinary Frechet after repair: {report.ordinary_frechet:.2f} m "
        f"(delta = {args.delta})"
    )

    print(f"\n  injected corruptions at : {injected}")
    print(f"  edits located at        : {sorted(e.index for e in deletions)}")
    print(
        "  These need not coincide. Repeated samples and delta permit cheaper\n"
        "  repairs than the injection, so edit count is not corruption count."
    )


if __name__ == "__main__":
    main()
