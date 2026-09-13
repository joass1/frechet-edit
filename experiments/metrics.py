"""Tie-aware retrieval metrics with explicit abstention accounting.

This module is deliberately independent of any distance. It takes scores and
relevance labels and nothing else, so it can be tested against hand-computed
values without FED being involved at all. That matters: ranking code is where
an evaluation quietly becomes wrong.

Three rules are load-bearing.

1. **Ties are never broken using the truth.** A tied group is reported as an
   interval - strict (worst case), optimistic (best case) - plus the exact
   expectation under a uniformly random ordering within the group.
2. **Failures stay in the denominator.** A query whose whole gallery is
   infeasible or numerically ambiguous is an ABSTENTION. It is reported, it
   counts against coverage, and it is never silently dropped. Both
   failure-aware and successful-query-only figures are produced.
3. **Lower is better.** Every score here is a dissimilarity.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

# A template whose score could not be computed at all.
AMBIGUOUS = "ambiguous"
INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class Scored:
    """One (template, score) pair for a single query.

    ``score`` is a float for a computed dissimilarity, ``math.inf`` for a
    mathematically infeasible template, or ``None`` when the numerical policy
    abstained.
    """

    template_id: str
    score: float | None

    @property
    def state(self) -> str:
        if self.score is None:
            return AMBIGUOUS
        if math.isinf(self.score):
            return INFEASIBLE
        return "scored"


@dataclass
class QueryOutcome:
    """Per-query results. Aggregates are derived from these, never the reverse."""

    query_id: str
    relevant: frozenset[str]
    n_templates: int
    n_scored: int
    n_infeasible: int
    n_ambiguous: int
    abstained: bool
    hit_at: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    rr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    top1_id: str | None = None
    top1_score: float | None = None
    top1_tied: int = 1

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "relevant": sorted(self.relevant),
            "n_templates": self.n_templates,
            "n_scored": self.n_scored,
            "n_infeasible": self.n_infeasible,
            "n_ambiguous": self.n_ambiguous,
            "abstained": self.abstained,
            "hit_at": {
                str(k): {"strict": v[0], "expected": v[1], "optimistic": v[2]}
                for k, v in sorted(self.hit_at.items())
            },
            "rr": {"strict": self.rr[0], "expected": self.rr[1], "optimistic": self.rr[2]},
            "top1_id": self.top1_id,
            "top1_score": self.top1_score,
            "top1_tied": self.top1_tied,
        }


def _comb(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def group_scores(scored: Sequence[Scored]) -> list[tuple[float | None, list[str]]]:
    """Order templates into tie groups, best first.

    Finite scores ascend. Infeasible templates form one group after them.
    Ambiguous templates form a final group: they are not retrievable, and they
    are kept visible rather than discarded.
    """
    finite: dict[float, list[str]] = {}
    infeasible: list[str] = []
    ambiguous: list[str] = []
    for item in scored:
        if item.state == AMBIGUOUS:
            ambiguous.append(item.template_id)
        elif item.state == INFEASIBLE:
            infeasible.append(item.template_id)
        else:
            finite.setdefault(float(item.score), []).append(item.template_id)

    groups: list[tuple[float | None, list[str]]] = [
        (score, sorted(ids)) for score, ids in sorted(finite.items())
    ]
    if infeasible:
        groups.append((math.inf, sorted(infeasible)))
    if ambiguous:
        groups.append((None, sorted(ambiguous)))
    return groups


def _first_relevant_group(
    groups: Sequence[tuple[float | None, list[str]]], relevant: Iterable[str]
) -> tuple[int, int, int] | None:
    """Return ``(n_before, group_size, n_relevant_in_group)`` for the first group
    containing a relevant template, or ``None`` if none does."""
    rel = set(relevant)
    before = 0
    for score, ids in groups:
        hits = sum(1 for t in ids if t in rel)
        if hits:
            if score is None:
                # Only reachable in an ambiguous group: not retrievable at all.
                return None
            return before, len(ids), hits
        before += len(ids)
    return None


def hit_at_k(
    groups: Sequence[tuple[float | None, list[str]]],
    relevant: Iterable[str],
    k: int,
) -> tuple[float, float, float]:
    """``(strict, expected, optimistic)`` probability that a relevant template is
    ranked in the top ``k``.

    ``strict`` assumes every tie resolves against us, ``optimistic`` that every
    tie resolves in our favour, and ``expected`` is the exact probability under
    a uniformly random ordering inside each tie group.
    """
    found = _first_relevant_group(groups, relevant)
    if found is None:
        return 0.0, 0.0, 0.0
    before, size, rel = found

    optimistic = 1.0 if before + 1 <= k else 0.0
    strict = 1.0 if before + (size - rel) + 1 <= k else 0.0

    if before >= k:
        expected = 0.0
    elif before + size <= k:
        expected = 1.0
    else:
        t = k - before
        # P(at least one of `rel` relevant items in the first t of a uniformly
        # random permutation of `size` items).
        expected = 1.0 - (_comb(size - rel, t) / _comb(size, t))
    return strict, expected, optimistic


def reciprocal_rank(
    groups: Sequence[tuple[float | None, list[str]]], relevant: Iterable[str]
) -> tuple[float, float, float]:
    """``(strict, expected, optimistic)`` reciprocal rank of the FIRST relevant template."""
    found = _first_relevant_group(groups, relevant)
    if found is None:
        return 0.0, 0.0, 0.0
    before, size, rel = found

    optimistic = 1.0 / (before + 1)
    strict = 1.0 / (before + (size - rel) + 1)

    # P(first relevant at position p) = C(size - p, rel - 1) / C(size, rel).
    total = Fraction(0)
    denom = _comb(size, rel)
    for p in range(1, size - rel + 2):
        weight = Fraction(_comb(size - p, rel - 1), denom)
        total += weight * Fraction(1, before + p)
    return strict, float(total), optimistic


def evaluate_query(
    query_id: str,
    scored: Sequence[Scored],
    relevant: Iterable[str],
    ks: Sequence[int] = (1, 5),
) -> QueryOutcome:
    """Score one query. An all-unusable gallery is an abstention, not a miss."""
    relevant = frozenset(relevant)
    groups = group_scores(scored)
    n_scored = sum(1 for s in scored if s.state == "scored")
    n_inf = sum(1 for s in scored if s.state == INFEASIBLE)
    n_amb = sum(1 for s in scored if s.state == AMBIGUOUS)
    abstained = n_scored == 0

    outcome = QueryOutcome(
        query_id=query_id,
        relevant=relevant,
        n_templates=len(scored),
        n_scored=n_scored,
        n_infeasible=n_inf,
        n_ambiguous=n_amb,
        abstained=abstained,
    )
    if not abstained:
        top_score, top_ids = groups[0]
        if top_score is not None and not math.isinf(top_score):
            outcome.top1_id = top_ids[0]
            outcome.top1_score = top_score
            outcome.top1_tied = len(top_ids)
    if relevant and not abstained:
        outcome.hit_at = {k: hit_at_k(groups, relevant, k) for k in ks}
        outcome.rr = reciprocal_rank(groups, relevant)
    else:
        outcome.hit_at = dict.fromkeys(ks, (0.0, 0.0, 0.0))
    return outcome


@dataclass
class Aggregate:
    """Aggregated metrics with both denominators reported."""

    method: str
    n_queries: int
    n_abstentions: int
    recall_at: dict[int, tuple[float, float, float]]
    mrr: tuple[float, float, float]
    recall_at_successful: dict[int, tuple[float, float, float]]
    mrr_successful: tuple[float, float, float]
    n_open_set: int
    n_false_matches: int

    @property
    def coverage(self) -> float:
        return 1.0 - (self.n_abstentions / self.n_queries) if self.n_queries else 0.0

    @property
    def false_match_rate(self) -> float | None:
        """Fraction of no-match queries falsely accepted. ``None`` if none were posed."""
        if self.n_open_set == 0:
            return None
        return self.n_false_matches / self.n_open_set

    def as_dict(self) -> dict:
        def fmt(triple):
            return {"strict": triple[0], "expected": triple[1], "optimistic": triple[2]}

        return {
            "method": self.method,
            "n_queries": self.n_queries,
            "n_abstentions": self.n_abstentions,
            "coverage": self.coverage,
            "recall_at": {str(k): fmt(v) for k, v in sorted(self.recall_at.items())},
            "mrr": fmt(self.mrr),
            "recall_at_successful_only": {
                str(k): fmt(v) for k, v in sorted(self.recall_at_successful.items())
            },
            "mrr_successful_only": fmt(self.mrr_successful),
            "n_open_set_queries": self.n_open_set,
            "n_false_matches": self.n_false_matches,
            "false_match_rate": self.false_match_rate,
        }


def aggregate(
    method: str,
    outcomes: Sequence[QueryOutcome],
    ks: Sequence[int] = (1, 5),
    acceptance_threshold: float | None = None,
) -> Aggregate:
    """Combine per-query outcomes.

    Closed-set queries (those with a non-empty relevant set) drive recall and
    MRR. Open-set queries (empty relevant set) drive the false-match rate, with
    ``acceptance_threshold`` deciding what counts as accepting a match.
    """
    closed = [o for o in outcomes if o.relevant]
    openset = [o for o in outcomes if not o.relevant]
    n = len(closed)
    n_abs = sum(1 for o in closed if o.abstained)
    successful = [o for o in closed if not o.abstained]

    def mean_triple(items, pick):
        if not items:
            return (0.0, 0.0, 0.0)
        return tuple(
            sum(pick(o)[i] for o in items) / len(items) for i in range(3)
        )  # type: ignore[return-value]

    recall_at = {k: mean_triple(closed, lambda o, k=k: o.hit_at[k]) for k in ks}
    recall_ok = {k: mean_triple(successful, lambda o, k=k: o.hit_at[k]) for k in ks}

    # A false-match rate is only meaningful against an explicit acceptance
    # threshold on THIS method's own scale. An edit count and a distance in
    # metres are not commensurable, so without a threshold the rate is not
    # computed at all rather than being silently defined as "always accept".
    false_matches = 0
    if acceptance_threshold is not None:
        for o in openset:
            if o.abstained or o.top1_score is None:
                continue
            if o.top1_score <= acceptance_threshold:
                false_matches += 1
    n_open = len(openset) if acceptance_threshold is not None else 0

    return Aggregate(
        method=method,
        n_queries=n,
        n_abstentions=n_abs,
        recall_at=recall_at,
        mrr=mean_triple(closed, lambda o: o.rr),
        recall_at_successful=recall_ok,
        mrr_successful=mean_triple(successful, lambda o: o.rr),
        n_open_set=n_open,
        n_false_matches=false_matches,
    )
