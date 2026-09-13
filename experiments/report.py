"""Render a run directory as a self-contained Markdown report.

The report is required to show every method, including the ones that beat FED
and the ones where FED abstains entirely. A report that omitted a loss would be
worse than no report.

    python -m experiments.report --run-dir runs/pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_SUMMARY_FIELDS = ("evidence_level", "config", "environment", "aggregates", "gallery")
REQUIRED_AGG_FIELDS = (
    "n_queries",
    "n_abstentions",
    "coverage",
    "recall_at",
    "mrr",
    "n_open_set_queries",
)


class IncompleteRun(RuntimeError):
    """Raised when a run directory is missing mandatory evidence.

    A missing field must fail the report. Rendering a partial run as if it were
    complete is how an evaluation silently becomes a lie.
    """


def load(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = run_dir / "summary.json"
    per_query_path = run_dir / "per_query.json"
    for path in (summary_path, per_query_path):
        if not path.exists():
            raise IncompleteRun(f"missing mandatory file: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_query = json.loads(per_query_path.read_text(encoding="utf-8"))

    missing = [f for f in REQUIRED_SUMMARY_FIELDS if f not in summary]
    if missing:
        raise IncompleteRun(f"summary.json is missing mandatory fields: {missing}")
    if not summary["aggregates"]:
        raise IncompleteRun("summary.json contains no method aggregates")
    for name, agg in summary["aggregates"].items():
        gaps = [f for f in REQUIRED_AGG_FIELDS if f not in agg]
        if gaps:
            raise IncompleteRun(f"aggregate for {name!r} is missing fields: {gaps}")
    if not per_query:
        raise IncompleteRun("per_query.json is empty; per-query evidence is mandatory")
    return summary, per_query


def _triple(d: dict[str, float]) -> str:
    return f"{d['expected']:.3f} [{d['strict']:.3f}, {d['optimistic']:.3f}]"


def render(summary: dict[str, Any], per_query: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Retrieval pilot report")
    add("")
    # The banner is derived from the run, never hardcoded: a level A banner on a
    # level B run would misreport the strength of the evidence, which is the one
    # mistake this report exists to prevent.
    level = str(summary.get("evidence_level", "A"))
    if level.startswith("B"):
        add("> ## EVIDENCE LEVEL B - REAL GEOMETRY, CORRUPTION-DERIVED LABELS")
        add("> ")
        add(f"> {summary['evidence_warning']}")
        add("> ")
        add("> The base curves are measured GPS traces, so these numbers DO describe")
        add("> real trajectory geometry. The ground truth is still known by")
        add("> construction, not observed, so they are NOT evidence of natural route")
        add("> recovery and NOT evidence of any deployment.")
        source_ids = summary.get("geolife_source_ids") or []
        if source_ids:
            add("> ")
            add(f"> Source traces ({len(source_ids)}): `{'`, `'.join(source_ids)}`.")
            add("> No trajectory coordinates are reproduced here; the dataset licence")
            add("> forbids redistributing the data or any derivative work.")
    else:
        add("> ## EVIDENCE LEVEL A - SYNTHETIC DATA ONLY")
        add("> ")
        add(f"> {summary['evidence_warning']}")
        add("> ")
        add("> No real trajectory was used, downloaded, or derived from. These numbers")
        add("> describe the MECHANISM of the measures on constructed fixtures. They are")
        add("> not evidence about GPS traces, route recovery, or any deployment.")
    add("")

    env = summary["environment"]
    add("## Environment and seeds")
    add("")
    add("| key | value |")
    add("|---|---|")
    for key in sorted(env):
        add(f"| {key} | `{env[key]}` |")
    add("")

    add("## Configuration (frozen before the run)")
    add("")
    add("```toml")
    for key, value in summary["config"].items():
        add(f"{key} = {json.dumps(value)}")
    add("```")
    add("")

    gallery = summary["gallery"]
    kinds: dict[str, int] = {}
    for route in gallery:
        kinds[route["kind"]] = kinds.get(route["kind"], 0) + 1
    add(f"Gallery: {len(gallery)} templates.")
    add("")
    add("| template kind | count |")
    add("|---|---|")
    for kind in sorted(kinds):
        add(f"| {kind} | {kinds[kind]} |")
    add("")
    add(
        "Only the `base` template is relevant for a query derived from it. The "
        "`corridor`, `shared_endpoint`, `partial_overlap` and `detour` entries in "
        "the same family are HARD NEGATIVES and must not be retrieved."
    )
    add("")

    add("## Results")
    add("")
    add("Every cell is `expected [strict, optimistic]` over tie groups. Ties are")
    add("never broken using the ground truth. `coverage` is the fraction of")
    add("closed-set queries that were not abstentions; abstentions remain in the")
    add("denominator of the failure-aware metrics.")
    add("")
    aggs = summary["aggregates"]
    # Render whatever cut-offs the run actually used, rather than assuming any.
    ks = sorted(next(iter(aggs.values()))["recall_at"], key=int)
    primary = ks[0]
    recall_cols = " ".join(f"Recall@{k} |" for k in ks)
    add(f"| method | {recall_cols} MRR | coverage | abstentions | false-match rate |")
    add("|---|" + "---|" * (len(ks) + 4))
    order = sorted(aggs.items(), key=lambda kv: -kv[1]["recall_at"][primary]["expected"])
    for name, agg in order:
        fmr = agg.get("false_match_rate")
        fmr_text = "n/a" if fmr is None else f"{fmr:.3f}"
        cells = " | ".join(_triple(agg["recall_at"][k]) for k in ks)
        add(
            f"| `{name}` | {cells} | {_triple(agg['mrr'])} | "
            f"{agg['coverage']:.2f} | {agg['n_abstentions']}/{agg['n_queries']} | {fmr_text} |"
        )
    add("")

    add("### Successful-query-only figures")
    add("")
    add("Reported alongside the failure-aware table above, never instead of it.")
    add("")
    add(f"| method | Recall@{primary} (successful only) | MRR (successful only) |")
    add("|---|---|---|")
    for name, agg in order:
        add(
            f"| `{name}` | {_triple(agg['recall_at_successful_only'][primary])} | "
            f"{_triple(agg['mrr_successful_only'])} |"
        )
    add("")

    add("### Acceptance thresholds")
    add("")
    add("| method | threshold | rule |")
    add("|---|---|---|")
    for name, agg in order:
        thr = agg.get("acceptance_threshold")
        thr_text = "none (no validation score available)" if thr is None else f"{thr:.4g}"
        add(f"| `{name}` | {thr_text} | {agg.get('threshold_rule', 'n/a')} |")
    add("")
    add(
        "Thresholds are per method and on each method's own scale. An edit count "
        "and a distance in metres are not commensurable and are never compared "
        "numerically."
    )
    add("")

    add("## Reading these results honestly")
    add("")
    best = order[0][1]["recall_at"][primary]["expected"]
    winners = [
        name for name, agg in order
        if agg["recall_at"][primary]["expected"] >= best - 1e-12
    ]
    add(
        f"- Top Recall@{primary} is {best:.3f}, attained by: "
        f"{', '.join(f'`{w}`' for w in winners)}."
    )
    if len(winners) > 1:
        add(
            "- Several methods tie at the top, so this regime does NOT separate them. "
            "A tie is not a win for FED; drawing a ranking between tied methods here "
            "would be unsupported."
        )
    for name, agg in order:
        if agg["coverage"] == 0.0:
            add(
                f"- `{name}` abstained on every query "
                f"({agg['n_abstentions']}/{agg['n_queries']}). Its score of 0 is a "
                "TOTAL ABSTENTION, not a measured failure to rank."
            )
    fed_both = aggs.get("fed_both", {}).get("recall_at", {}).get(primary, {}).get("expected")
    plain = aggs.get("frechet", {}).get("recall_at", {}).get(primary, {}).get("expected")
    if fed_both is not None and plain is not None:
        add(
            f"- `fed_both` {fed_both:.3f} vs raw `frechet` {plain:.3f} on identical "
            "inputs. This is the mechanism the edit objective is designed for."
        )
        beaten = [
            name for name, agg in aggs.items()
            if name != "fed_both"
            and agg["recall_at"][primary]["expected"] > (fed_both + 1e-12)
        ]
        if beaten:
            add(f"- Methods that BEAT `fed_both` here: {', '.join(f'`{b}`' for b in beaten)}.")
        else:
            add("- No method strictly beat `fed_both` in this regime.")
    add("")
    add(
        "- No confidence intervals are reported. The query count here is a pilot "
        "and the units are not independent enough to support them."
    )
    add(f"- Per-query records for all {len(per_query)} queries are in `per_query.json`.")
    add("")

    add("## What this run does NOT establish")
    add("")
    add("- Nothing about real GPS traces, GeoLife, T-Drive, or any measured data.")
    add("- Nothing about natural route recovery: that needs evidence level C,")
    add("  which requires independently annotated repeated trips and is BLOCKED.")
    add("- No latency or deployment claim; see `docs/performance.md` for timings.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary, per_query = load(args.run_dir)
    text = render(summary, per_query)
    out = args.out or (args.run_dir / "report.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
