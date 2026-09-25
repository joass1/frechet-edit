# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-25

### Fixed

- **The certified point predicate gave wrong answers near underflow.** Its
  float64 tier accepted a decision whenever the squared distance and `delta**2`
  were separated by a *relative* margin, which is not a bound once those
  squares are subnormal (coordinates near `1e-162`). A 283849-case probe found
  1555 confident wrong decisions, and the public API returned `cost=0` for
  instances that are infeasible. Tier 1 now adds an absolute underflow
  allowance and sends overflowed squares to the exact tier; the enclosing-ball
  YES certificate uses the same test. Normal-range inputs are decided exactly
  as before. Pinned in `tests/unit/test_numeric_boundary.py`.

### Added

- `continuous_edit_distance`: strong **continuous** Frechet edit distance,
  deletion only (paper Section 4.1, Theorem 3), via the weighted DAG complex
  and product free-space reachability, evaluated in anti-diagonal wavefronts.
  Every comparison is exact: interval endpoints are ranked per segment with
  outward-rounded float enclosures and an exact `x + s*sqrt(y)` fallback.
  Optional `max_deletions` with a distinct `budget_exceeded` status.
- `continuous_frechet_within` (ordinary continuous Frechet decision),
  `continuous_frechet_le` and `verify_continuous_witness` (an independent exact
  Alt-Godau verifier that shares no code with the solver); `verify_witness`
  routes continuous results to it. `UnsupportedOperationError` for continuous
  insertion, which is not implemented.
- `docs/continuous.md`, including one deviation from the paper's prose:
  explicit product-vertex propagation, needed for paths that wait at the last
  observation vertex.
- Course Check (`webapp/`, `docs/web-app.md`): a FastAPI + Leaflet web app that
  audits a GPS track against a course with the continuous measure, with five
  synthetic Marina Bay scenarios, GPX/GeoJSON/CSV upload, glitch ledger,
  offset trace, GPX export, and unit, API and real-browser tests. Optional
  extras `app` and `e2e`; not part of the installed package. Hardened after an
  independent security review: bounded simplification work (an O(n^2)
  Douglas-Peucker input had taken 315 s), a byte-counted body cap that chunked
  requests cannot bypass, and refusal of cross-site browser POSTs.

### Changed

- `docs/limitations.md` no longer says that no real data was used; level B
  GeoLife evidence has existed since 0.1.0.

## [0.1.0] - 2026-09-14

First published release. Alpha: the discrete library is complete and tested,
but nothing in it has been independently reviewed. See
[docs/limitations.md](docs/limitations.md) and
[docs/reviews/STATUS.md](docs/reviews/STATUS.md) before relying on it.

### Added

- **Dimensions 1 to 8** for insertion-capable modes. The first release plan
  restricted `insert` and `both` to the plane; `_meb` now computes the exact
  minimum enclosing ball in any dimension by support refinement. Deletion has
  no dimension bound at all. The remaining cap is a cost bound, not a
  correctness one.
- `docs/errata-insertion-recurrence.md`, recording that the recurrence
  published for the insertion and mixed variants is unsound, with a minimal
  counterexample, a worked trace and the verification counts behind it. The
  complexity theorems are not affected.
- `experiments/geolife.py` and `experiments/configs/geolife.toml`, an evidence
  level B evaluation on real GeoLife trajectory geometry. No trajectory data is
  distributed with this package and none may be.
- `experiments/freespace_viz.py`, free-space diagram rendering for the
  continuous Frechet distance, cross-checked against the exact Alt-Godau
  decision procedure.
- `discrete_edit_distance` implementing the strong discrete Frechet edit
  distance in all three modes (`delete`, `insert`, `both`).
- Arbitrary-location vertex insertion via minimum-enclosing-ball geometry, so a
  single inserted point can cover a contiguous block of reference vertices.
- Replayable minimal-edit witnesses: edit script, edited curve and coupling.
- `verify_witness`, an independent verifier that re-derives every claim and
  rejects tampering. It shares no recurrence with the solver.
- A three-tier certified numerical policy: float64 with a rigorous error bound,
  exact rational fallback at the boundary, and explicit abstention
  (`status="numerically_ambiguous"`) rather than a guess.
- Two backends that must agree on cost: `reference` (readable full tables,
  used for witnesses) and `python` (rolling `O(m)` storage plus a monotone
  minimum queue).
- Independent brute-force oracles under `tests/oracles`, written from the
  specification without sight of the dynamic program.
- Baseline implementations for comparison: ordinary and continuous discrete
  Frechet, DTW, EDR, LCSS, ERP, and preregistered smoothing filters.
- A synthetic retrieval harness with hard negatives, tie-aware metrics,
  abstention accounting and fail-closed reporting.

### Fixed during development

- **Corrected an unsound recurrence.** A collapsed single-table form whose keep
  branch uses the unrestricted `F(i-1, j)` as its vertical predecessor
  underestimates the optimum once insertions are allowed. Minimal
  counterexample: `R = [0, 1, 0]`, `Q = [0]`, `delta = 0.4`, insert-only, where
  the collapsed form returns 1 and the true optimum is 2. Found by exhaustive
  test and confirmed against an independent oracle. The implementation uses the
  layered `K`/`P`/`X` form throughout. See `docs/recurrences.md` section 3.2.

### Known limitations

- Insertion-capable modes are certified for dimensions 1 to 8. The cap is the
  cost of the exact enclosing-ball kernel, not a correctness boundary.
  Deletion-only has no dimension bound.
- Continuous variants, weak variants and substitutions are not implemented.
- No native extension. Profiling showed the bottleneck was per-call overhead on
  small inputs, which a compiled extension would not help; see
  `docs/performance.md`.
- Empirical evidence reaches level B: real GeoLife trajectory geometry with
  ground truth known by construction from the injected corruption. That is NOT
  evidence of natural route recovery. Level C, which needs blinded annotation,
  remains blocked. No trajectory data is distributed here.
- **Nothing in this release has been independently reviewed.** Every phase in
  `docs/reviews/STATUS.md` sits at READY_FOR_REVIEW. A passing test suite is
  evidence, not a signoff.
