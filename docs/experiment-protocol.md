# Experiment protocol and preregistration

Status: the only evidence this repository actually contains is LEVEL A.
Levels B and C are BLOCKED, for the reasons recorded below.

## 1. Evidence levels

Taken from the project plan, and used unchanged so that claims cannot drift.

| level | what it is | status here |
|---|---|---|
| A | synthetic mechanism tests on analytically controlled fixtures | **DONE** |
| B | real-trace-derived controlled corruption (real curves, injected noise) | **BLOCKED** |
| C | natural repeated-route evaluation with independent blinded annotation | **BLOCKED** |

**Level B is blocked** because no licensed real trajectory data is present in
this workspace and none was downloaded. GeoLife and T-Drive are distributed
under non-commercial terms that also forbid redistribution of the data or
derivative works, so no raw or transformed fixture may be committed here. See
`docs/data-and-labels.md`.

**Level C is blocked** additionally because it needs independent blinded
annotators and a route-equivalence rubric that does not exist. Pseudo-labels
derived from Frechet or FED scores would make the evaluation circular and are
explicitly forbidden.

A level A result can never be reported as evidence for a level B or C claim.

## 2. Hypothesis under test (level A)

> On curves corrupted by isolated large spikes plus small background jitter,
> ranking templates by minimum edit count (FED) recovers the source template
> more often than ranking by ordinary discrete Frechet distance on identical
> inputs.

This is a hypothesis about the MECHANISM of the objective. It is not a
hypothesis about GPS traces, and confirming it says nothing about real data.

## 3. What was frozen before running

* **Gallery construction.** Six families. Each family contributes one `base`
  route and four HARD negatives: a parallel `corridor` twin at 30 m, a
  `shared_endpoint` route with the same start and end, a `partial_overlap`
  route that follows the base then diverges, and a `detour` route containing a
  genuine excursion. Random distant negatives alone would make the task
  trivial. Only the `base` template is relevant for a query derived from it.
* **Query construction.** Two queries per base route: two spikes of 150 m plus
  Gaussian jitter of 2 m on every vertex. Four additional open-set queries that
  match nothing in the gallery.
* **Primary metric.** Recall@1, reported as `expected [strict, optimistic]`
  over tie groups. Secondary: Recall@5 and MRR in the same form.
* **Tie policy.** Ties are NEVER broken using the ground truth. A tied group is
  reported as an interval plus the exact expectation under uniformly random
  ordering within the group.
* **Abstention policy.** A query whose whole gallery is infeasible or
  numerically ambiguous is an ABSTENTION. It stays in the denominator of the
  failure-aware metrics, and successful-query-only figures are reported
  separately and alongside, never instead.
* **Delta selection.** `delta = 25` m, fixed a priori for all FED modes, and
  reused as the `eps` of EDR and LCSS so those baselines see the same spatial
  tolerance. It was not tuned on outcomes.
* **Acceptance threshold.** Chosen PER METHOD on its own scale, as the 0.95
  quantile of top-1 scores over two VALIDATION families that contribute no test
  query. An edit count and a distance in metres are not commensurable, so a
  single shared threshold would be meaningless.
* **Split.** Validation and test families are disjoint. No family contributes to
  both.

## 4. Baselines

The full competitor set is implemented in
`experiments/baselines.py` and independently tested in
`experiments/test_baselines.py` (71 tests): ordinary discrete Frechet,
continuous Frechet, a preregistered moving-average filter followed by discrete
Frechet, EDR, LCSS, DTW and ERP. Baselines are given identical inputs and the
same spatial tolerance. A baseline implemented sloppily would invalidate the
whole comparison, so their definitions, normalisations and endpoint rules are
documented per function and checked against hand-computed values.

## 5. What would count as a positive result

For level A only:

* a positive margin for `fed_both` over raw `frechet` on Recall@1 under the
  preregistered corruption regime, AND
* no material regression on clean (uncorrupted) queries.

Beating raw ordinary Frechet is NOT sufficient to call FED the best matcher.
If FED merely ties or loses to filtering, EDR, LCSS, DTW or ERP, that is the
result, it gets published as such, and the package is positioned as a faithful,
explainable implementation rather than a superior matcher.

## 6. Actual outcome of the pilot

Run: `runs/pilot`, report `runs/pilot/report.md`, seed 20240612.

* `fed_delete`, `fed_both`, `edr`, `dtw` and `erp` all reach Recall@1 = 1.000.
* Raw `frechet` reaches 0.188; `filter_frechet` 0.750; `lcss` 0.500.
* `fed_insert` abstains on **every** query (coverage 0.00), because insertion
  alone cannot remove a spike. Its zero is a total abstention, not a ranking
  failure, and the report says so explicitly.

Conclusions actually supported:

1. The mechanism works: on identical inputs, FED's edit objective is far more
   robust to isolated spikes than raw discrete Frechet (1.000 vs 0.188).
2. **FED does not beat the strong robust baselines here.** It ties EDR, DTW and
   ERP at ceiling. A tie is not a win.
3. **This regime does not discriminate** between the top five methods. Ceiling
   effects mean the pilot cannot rank them, and no ranking between them is
   claimed. A harder regime (smaller spikes, higher corruption rates, tighter
   corridors) would be needed to separate them, and is future work.
4. No confidence intervals are reported. The pilot's query count is small and
   its units are not independent enough to support them.

## 7. Rules that remain in force

* No claim of natural route recovery from level A evidence.
* No comparison of an edit count to a distance in metres as if commensurable.
* Numerical abstentions are reported, never dropped from a denominator.
* Per-query records are always persisted, so any aggregate can be recomputed.
* A missing mandatory field fails the report rather than rendering a partial run
  as if complete (`experiments/report.py::IncompleteRun`).
