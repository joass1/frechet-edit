# Limitations

Read this before using the package or citing anything from it.

## What is implemented

* Strong **discrete** Frechet edit distance, all three modes: `delete`,
  `insert`, `both`.
* Inserted vertices may be placed **anywhere** in the supported Euclidean space,
  via minimum-enclosing-ball geometry - not restricted to reference vertices.
* Replayable, independently verified minimal-edit witnesses.
* A certified numerical policy with explicit abstention.
* Strong **continuous** Frechet edit distance, **deletion only** (paper
  Section 4.1, Theorem 3), with exact comparisons, witnesses and an independent
  exact verifier. See `docs/continuous.md`.

## What is NOT implemented

| feature | status |
|---|---|
| continuous insertion and mixed edits | not implemented; they need the paper's minimum-link and canonical-subcurve machinery (`O(n m^5)` in the plane). Continuous DELETION is implemented. |
| weak traversal variants | not implemented; the source proves NP-hardness for the relevant weak variants |
| substitutions | not implemented; the authors defer the details, so this is a derivation project (S3), not a transcription |
| weighted edit costs | out of scope |
| editing both curves | out of scope |
| symmetrisation, temporal penalties, endpoint locking | out of scope |
| native / C++ / JIT backend | not started (S1) |
| map matching, streaming, learned embeddings | out of scope |

## Mathematical caveats

* **The result is a COUNT, not a distance.** It must never be compared
  numerically against metres, or against an ordinary Frechet distance, as
  though the scales were commensurable.
* **The measure is directed.** `FED(R, Q)` edits `Q` only. Swapping the
  arguments asks a different question, and the API is named so that this
  cannot happen by accident.
* **It is not a metric.** No symmetry, no identity of indiscernibles and no
  triangle inequality is claimed or tested. Do not put it in a metric index.
* **It is not robust to sampling density by construction.** Costs count
  vertices. Resampling changes the answer; that is an experimental condition to
  control, not an invariance to assume.
* **An edit objective cannot tell noise from intent.** A genuine short detour
  can be erased as cheaply as two noise spikes. This is a real failure mode,
  and `tests/unit/test_mixed.py::TestFailureModeFixtures` pins it deliberately
  rather than hiding it.
* **Insertion does not help every dropout.** Discrete Frechet already tolerates
  missing samples and stationary repetition; insertion helps only specific
  missing-geometry situations.

## Scope restrictions of this implementation

* Insertion-capable modes (`insert`, `both`) are certified for **dimensions 1
  to 8**. The bound is the cost of the exact enclosing-ball kernel, whose
  per-round work grows like `2**(d+2)` candidate subsets each solving a `d x d`
  rational system, so it stops being practical well before it stops being
  correct. Deletion-only has no such bound, because it needs no enclosing ball.
  Beyond the cap, `UnsupportedDimensionError` is raised rather than silently
  degrading. This is a product restriction, not a limitation of the theorem.
* `delta` must be finite and strictly positive. Zero-threshold support is not
  in this release.
* Public inputs must be non-empty and finite.

## Numerical caveats

* Geometric predicates are decided in float64 with a rigorous error bound, then
  in exact rational arithmetic at the boundary. Genuinely undecidable cases
  above the exact cap return `status="numerically_ambiguous"` with `cost=None`.
  **Ambiguity is never rewritten as infinity or as a feasible answer.**
* A certified optimal cost can come with `witness_status="unavailable"` when no
  float64 point near an inserted centre can be certified within `delta`. That
  is a representability limit, NOT infeasibility, and `delta` is never adjusted
  to rescue a witness.
* `numeric_policy="fast"` is float-only and is NOT certified. It exists to
  measure the cost of certification, not as a default.

## Evidence caveats

* Empirical evidence reaches **level B**: real GeoLife trajectory geometry with
  ground truth known by construction from injected corruption
  (`docs/results-geolife.md`), on top of the level A synthetic pilot. That
  supports "recovers the source trajectory under corruption" and nothing about
  matching routes in the wild; level C is blocked. An earlier revision of this
  file said no real data had been used, which stopped being true when level B
  was run. See `docs/experiment-protocol.md`.
* **No trajectory data is distributed here.** GeoLife's licence forbids
  redistributing it or derivatives. See `docs/data-and-labels.md`.
* The Course Check web app (`docs/web-app.md`) runs on **synthetic** scenarios.
  It demonstrates the method; it is not evidence about real athletes, vehicles
  or courses.
* In the level A pilot, FED **tied** EDR, DTW and ERP at ceiling and did not
  beat them. It decisively beat raw discrete Frechet. Ceiling effects mean the
  pilot cannot rank the top methods, and no such ranking is claimed.
* Performance numbers in `docs/performance.md` are single-machine measurements
  on curves of at most 400 points. They are not throughput guarantees and must
  not be extrapolated.
* The continuous solver is `O(k^2 m n)` for a deletion budget `k` (Theorem 3),
  `O(m n^3)` in the worst case. It is practical for budgets up to a few tens
  at a few hundred vertices; `max_deletions` bounds it, and a result of
  `budget_exceeded` is not a claim of infeasibility.
* In deletion-only mode the "optimised" backend is slightly slower than the
  reference backend. That is measured and reported rather than omitted.

## Correctness evidence, and its limits

* Optimality is established by agreement with independent oracles that share no
  recurrence with the solver, over exhaustive small families and randomised
  larger ones. That establishes optimality **for the cases tested**. It is not
  a proof of the implementation for all inputs.
* Witness feasibility alone never establishes minimality; only oracle agreement
  does.
* One real defect was found this way: the collapsed single-table form of the
  recurrence is UNSOUND once insertions are allowed. See `docs/recurrences.md`
  section 3.2.

  An earlier revision of this file said the question of whether the PUBLISHED
  proofs carry the restriction was "an open verification item, not a claimed
  erratum". That is no longer the position and the sentence was wrong to leave
  standing. Both published versions were subsequently checked directly and both
  display the unrestricted vertical predecessor, so the defect is in the
  publication and not only in this project's shorthand. It is now a claimed
  erratum, stated in `docs/errata-insertion-recurrence.md` with the scope it
  does and does not cover. The first author was contacted about it in
  September 2026.

## Provenance

This is an INDEPENDENT implementation written from the published description of
Fox, Nayyeri, Perry and Raichel, *Frechet Edit Distance*, SoCG 2024. No code
from the authors was used. A targeted search found no paper-specific public
implementation; that is a negative search result, not a claim that none exists,
and no "first implementation" claim is made. The authors have not been
contacted.
