# Limitations

Read this before using the package or citing anything from it.

## What is implemented

* Strong **discrete** Frechet edit distance, all three modes: `delete`,
  `insert`, `both`.
* Inserted vertices may be placed **anywhere** in the supported Euclidean space,
  via minimum-enclosing-ball geometry - not restricted to reference vertices.
* Replayable, independently verified minimal-edit witnesses.
* A certified numerical policy with explicit abstention.

## What is NOT implemented

| feature | status |
|---|---|
| continuous (polygonal) Frechet edit distance | not implemented; stretch goal S2 |
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
  and 2 only**, because that is where the enclosing-ball backend is certified.
  This is a product restriction of this package, not a limitation of the
  underlying theorem, and an unsupported dimension raises
  `UnsupportedDimensionError` rather than being silently degraded.
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

* The only empirical evidence in this repository is **level A: synthetic**.
  See `docs/experiment-protocol.md`.
* **No real trajectory data was used or downloaded.** Nothing here supports a
  claim about GPS traces, GeoLife, T-Drive, route recovery, or any deployment.
  See `docs/data-and-labels.md`.
* In the level A pilot, FED **tied** EDR, DTW and ERP at ceiling and did not
  beat them. It decisively beat raw discrete Frechet. Ceiling effects mean the
  pilot cannot rank the top methods, and no such ranking is claimed.
* Performance numbers in `docs/performance.md` are single-machine measurements
  on curves of at most 400 points. They are not throughput guarantees and must
  not be extrapolated.
* In deletion-only mode the "optimised" backend is slightly slower than the
  reference backend. That is measured and reported rather than omitted.

## Correctness evidence, and its limits

* Optimality is established by agreement with independent oracles that share no
  recurrence with the solver, over exhaustive small families and randomised
  larger ones. That establishes optimality **for the cases tested**. It is not
  a proof of the implementation for all inputs.
* Witness feasibility alone never establishes minimality; only oracle agreement
  does.
* One real defect was found this way and is preserved: a collapsed
  single-table form of the recurrence is UNSOUND once insertions are allowed.
  See `docs/recurrences.md` section 3.2. Whether the published proofs carry the
  restriction that this project's shorthand dropped is recorded as an open
  verification item in `docs/paper-map.md`, not as a claimed erratum.

## Provenance

This is an INDEPENDENT implementation written from the published description of
Fox, Nayyeri, Perry and Raichel, *Frechet Edit Distance*, SoCG 2024. No code
from the authors was used. A targeted search found no paper-specific public
implementation; that is a negative search result, not a claim that none exists,
and no "first implementation" claim is made. The authors have not been
contacted.
