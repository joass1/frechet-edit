# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

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
