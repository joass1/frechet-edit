# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased] - 0.1.0.dev0

Initial development release. Not published to any package index.

### Added

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

- Insertion-capable modes are certified for dimensions 1 and 2 only.
- Continuous variants, weak variants and substitutions are not implemented.
- Only synthetic (evidence level A) empirical results exist. No real trajectory
  data was used or downloaded. See `docs/limitations.md`.
