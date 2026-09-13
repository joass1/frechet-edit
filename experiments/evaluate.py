"""Retrieval experiment runner (evidence level A, synthetic only).

For each corrupted query trace, every gallery template is scored by every
method and the templates are ranked. The relevant template is the one the query
was generated from; the hard negatives from the same family are explicitly NOT
relevant, which is what makes the task non-trivial.

Honesty rules enforced here:

* every method sees the IDENTICAL inputs;
* a query whose gallery is entirely infeasible or ambiguous is an abstention
  and stays in the denominator;
* per-query records are written out, not only aggregates, so the analysis can
  be redone without re-running;
* nothing in this module knows which template is correct while scoring.

    python -m experiments.evaluate --config experiments/configs/pilot.toml
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import frechet_edit
from experiments import baselines, corruptions, synthetic
from experiments.metrics import Scored, aggregate, evaluate_query
from frechet_edit import discrete_edit_distance

ScoreFn = Callable[[np.ndarray, np.ndarray], float | None]


def build_methods(cfg: dict[str, Any]) -> dict[str, ScoreFn]:
    """All competing scorers. Lower is better for every one of them."""
    delta = float(cfg["delta"])
    eps = float(cfg.get("eps", delta))
    window = int(cfg.get("filter_window", 3))

    def fed(mode: str) -> ScoreFn:
        def score(template: np.ndarray, query: np.ndarray) -> float | None:
            result = discrete_edit_distance(
                template, query, delta, operations=mode, backend="python"
            )
            if result.status == "numerically_ambiguous":
                return None
            return float(result.cost)

        return score

    def filtered_frechet(template: np.ndarray, query: np.ndarray) -> float:
        smoothed = baselines.moving_average_filter(query, window)
        return baselines.discrete_frechet(template, smoothed)

    return {
        "fed_delete": fed("delete"),
        "fed_insert": fed("insert"),
        "fed_both": fed("both"),
        "frechet": lambda t, q: baselines.discrete_frechet(t, q),
        "filter_frechet": filtered_frechet,
        "edr": lambda t, q: float(baselines.edr(t, q, eps)),
        "lcss": lambda t, q: float(baselines.lcss_distance(t, q, eps)),
        "dtw": lambda t, q: baselines.dtw(t, q),
        "erp": lambda t, q: baselines.erp(t, q),
    }


def make_queries(
    rng: np.random.Generator, routes: list[synthetic.Route], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Corrupt base routes into queries, plus open-set queries matching nothing."""
    bases = [r for r in routes if r.kind == "base"]
    per_route = int(cfg.get("queries_per_route", 2))
    magnitude = float(cfg.get("spike_magnitude", 150.0))
    n_spikes = int(cfg.get("n_spikes", 2))
    jitter_sigma = float(cfg.get("jitter_sigma", 2.0))

    queries: list[dict[str, Any]] = []
    for route in bases:
        for rep in range(per_route):
            curve, record = corruptions.spike(route.points, rng, n_spikes, magnitude)
            curve, _ = corruptions.jitter(curve, rng, jitter_sigma)
            queries.append(
                {
                    "query_id": f"{route.route_id}-q{rep}",
                    "relevant": [route.route_id],
                    "curve": curve,
                    "corruption": record.as_dict(),
                    "open_set": False,
                }
            )

    for i in range(int(cfg.get("n_open_set_queries", 4))):
        stranger = synthetic.make_route(
            rng,
            int(cfg.get("n_points", 40)),
            origin=(float(rng.uniform(50_000, 60_000)), float(rng.uniform(50_000, 60_000))),
        )
        queries.append(
            {
                "query_id": f"openset-q{i}",
                "relevant": [],
                "curve": stranger,
                "corruption": None,
                "open_set": True,
            }
        )
    return queries


def run(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    seed = int(cfg.get("seed", 20240612))
    rng = np.random.default_rng(seed)
    n_families = int(cfg.get("n_families", 6))
    n_points = int(cfg.get("n_points", 40))

    # Evidence level is decided here and nowhere else. Level A generates its own
    # base curves; level B takes them from real trajectories. Everything
    # downstream - hard negatives, corruptions, metrics - is identical, so the
    # only thing that changes is where the geometry came from.
    gallery_source = str(cfg.get("gallery_source", "synthetic"))
    bases = None
    source_ids: list[str] | None = None
    if gallery_source == "geolife":
        from experiments import geolife

        trajectories = geolife.load_base_trajectories(
            n_families,
            n_points=n_points,
            source=cfg.get("geolife_source"),
            min_separation_m=float(cfg.get("geolife_min_separation_m", 5000.0)),
            spacing_m=float(cfg.get("geolife_spacing_m", 10.0)),
        )
        bases = [t.points for t in trajectories]
        source_ids = [t.source_id for t in trajectories]
    elif gallery_source != "synthetic":
        raise ValueError(f"unknown gallery_source {gallery_source!r}")

    routes = synthetic.make_gallery(
        rng,
        n_families=n_families,
        n_points=n_points,
        bases=bases,
    )
    queries = make_queries(rng, routes, cfg)
    methods = build_methods(cfg)
    ks = tuple(int(k) for k in cfg.get("ks", [1, 5]))

    per_query: list[dict[str, Any]] = []
    outcomes: dict[str, list] = {name: [] for name in methods}
    timings: dict[str, float] = dict.fromkeys(methods, 0.0)

    for query in queries:
        record: dict[str, Any] = {
            "query_id": query["query_id"],
            "relevant": query["relevant"],
            "open_set": query["open_set"],
            "corruption": query["corruption"],
            "scores": {},
        }
        for name, fn in methods.items():
            scored: list[Scored] = []
            start = time.perf_counter()
            for route in routes:
                value = fn(route.points, query["curve"])
                scored.append(Scored(route.route_id, value))
            timings[name] += time.perf_counter() - start
            outcome = evaluate_query(query["query_id"], scored, query["relevant"], ks)
            outcomes[name].append(outcome)
            record["scores"][name] = {
                s.template_id: (None if s.score is None else
                                ("inf" if math.isinf(s.score) else round(float(s.score), 6)))
                for s in scored
            }
            record.setdefault("outcomes", {})[name] = outcome.as_dict()
        per_query.append(record)

    # Acceptance thresholds are chosen PER METHOD on its own scale, using a
    # frozen rule applied to a VALIDATION split of families that is disjoint
    # from the reported test split. An edit count and a distance in metres are
    # not commensurable, so a single shared threshold would be meaningless.
    n_val = int(cfg.get("n_validation_families", 2))
    val_families = set(sorted({r.family for r in routes})[:n_val])
    quantile = float(cfg.get("threshold_quantile", 0.95))

    def family_of(query_id: str) -> str:
        return query_id.split("-")[0]

    thresholds: dict[str, float | None] = {}
    test_outcomes: dict[str, list] = {}
    for name, outs in outcomes.items():
        val_scores = [
            o.top1_score
            for o in outs
            if o.relevant and family_of(o.query_id) in val_families
            and not o.abstained and o.top1_score is not None
        ]
        thresholds[name] = (
            float(np.quantile(val_scores, quantile)) if val_scores else None
        )
        test_outcomes[name] = [
            o for o in outs
            if not o.relevant or family_of(o.query_id) not in val_families
        ]

    aggregates = {
        name: aggregate(
            name, outs, ks=ks, acceptance_threshold=thresholds[name]
        ).as_dict()
        for name, outs in test_outcomes.items()
    }
    for name in aggregates:
        aggregates[name]["total_scoring_seconds"] = round(timings[name], 3)
        aggregates[name]["acceptance_threshold"] = thresholds[name]
        aggregates[name]["threshold_rule"] = (
            f"quantile {quantile} of top-1 scores on validation families "
            f"{sorted(val_families)}, which supply no test query"
        )

    if gallery_source == "geolife":
        evidence_level = "B (real trajectory geometry, corruption-derived labels)"
        evidence_warning = (
            "Base curves are real GeoLife traces; the hard negatives are derived "
            "from them and the queries are corrupted copies, so ground truth is "
            "known BY CONSTRUCTION and is not a natural route-identity label. "
            "This supports a claim about recovering a source trajectory under "
            "injected corruption, and NOT a claim about route matching in the "
            "wild. Level C of docs/experiment-protocol.md remains BLOCKED: it "
            "needs blinded annotation that no dataset supplies."
        )
    else:
        evidence_level = "A (synthetic mechanism tests only)"
        evidence_warning = (
            "These routes are generated, not measured. No real trajectory data was "
            "used. Nothing here supports a claim about real GPS traces, and levels "
            "B and C of docs/experiment-protocol.md remain BLOCKED."
        )

    payload = {
        "evidence_level": evidence_level,
        "evidence_warning": evidence_warning,
        "gallery_source": gallery_source,
        # Source file stems only: provenance for reproducibility, no coordinates.
        # GeoLife may not be redistributed, so no trajectory data is written here.
        "geolife_source_ids": source_ids,
        "config": cfg,
        "environment": {
            "frechet_edit": frechet_edit.__version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "seed": seed,
        },
        "gallery": [r.as_dict() for r in routes],
        "aggregates": aggregates,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "per_query.json").write_text(json.dumps(per_query, indent=2), encoding="utf-8")
    return payload


def _load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML config.

    Imported lazily and only here: `tomllib` is 3.11+, and the package supports
    3.10. Keeping this out of module scope means importing `experiments.evaluate`
    - which the test suite does on every supported version - never depends on a
    parser that 3.10 lacks.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - exercised only on 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "Reading a TOML config on Python 3.10 needs tomli: "
                'pip install -e ".[experiments]", or use Python 3.11+.'
            ) from exc
    return dict(tomllib.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = _load_toml(args.config)
    run_dir = args.run_dir or Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    payload = run(cfg, run_dir)

    print(f"evidence level: {payload['evidence_level']}")
    print(f"run directory : {run_dir}\n")
    header = f"{'method':16} {'R@1 exp':>8} {'R@5 exp':>8} {'MRR exp':>8} {'cover':>7} {'FMR':>7}"
    print(header)
    print("-" * len(header))
    ordered = sorted(
        payload["aggregates"].items(),
        key=lambda kv: -kv[1]["recall_at"]["1"]["expected"],
    )
    for name, agg in ordered:
        fmr = agg["false_match_rate"]
        print(
            f"{name:16} {agg['recall_at']['1']['expected']:8.3f} "
            f"{agg['recall_at']['5']['expected']:8.3f} {agg['mrr']['expected']:8.3f} "
            f"{agg['coverage']:7.2f} {('n/a' if fmr is None else f'{fmr:.2f}'):>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
