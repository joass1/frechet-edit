# Phase status and evidence

Status vocabulary: NOT_STARTED, IN_PROGRESS, READY_FOR_REVIEW, BLOCKED,
FAILED, PASSED. **Nothing is promoted to PASSED without an independent
review**, so everything below is READY_FOR_REVIEW at best regardless of how
green the test suite is. A passing suite is evidence, not a signoff.

| phase | scope | status | evidence |
|---|---|---|---|
| P0 | contract, recurrences, witness invariants, numerics, oracle protocol, environment | READY_FOR_REVIEW | `docs/definition.md`, `docs/recurrences.md`, `docs/witness-invariants.md`, `docs/numerics.md`, `docs/oracle-protocol.md`, `docs/paper-map.md` |
| P1 | deletion-only spine, witnesses, verifier | READY_FOR_REVIEW | `tests/unit/test_deletion.py`, `test_witness.py`, exhaustive Oracle A agreement |
| P2 | real-data pilot | READY_FOR_REVIEW | GeoLife 1.3 obtained locally (gitignored); `experiments/geolife.py`, `experiments/test_geolife.py`, `docs/results-geolife.md` |
| P3 | insertion geometry, mixed edits, oracles | READY_FOR_REVIEW | `tests/unit/test_geometry.py`, `test_insertion.py`, `test_mixed.py`, Oracle B agreement |
| P4 | optimised backend, measured performance | READY_FOR_REVIEW | `docs/performance.md`, `benchmarks/`, `tests/unit/test_minqueue.py` |
| P5 | held-out application validation | levels A and B done; **BLOCKED** at level C | `docs/experiment-protocol.md`, `docs/results.md` (A), `docs/results-geolife.md` (B) |
| P6-L | library release | READY_FOR_REVIEW | wheel + sdist built, fresh-env install verified, `tests/integration/` |
| P6-A | application-evidence release | **NOT CLAIMED** | level B evidence now exists; level C does not, and no release claims it |
| S1 | native backend | NOT_STARTED, and deliberately so | profiling found the bottleneck was per-call overhead on tiny inputs, which a compiled extension makes worse, not better; see `docs/performance.md` |
| S2 | continuous deletion | READY_FOR_REVIEW | `docs/continuous.md`, `src/frechet_edit/_continuous.py`, `tests/unit/test_continuous.py` (oracle agreement, exhaustive 1-D slow tier), mutation testing, one independent adversarial review (below) |
| APP | Course Check web app (real-world use of S2) | READY_FOR_REVIEW | `docs/web-app.md`, `webapp/`, `webapp/tests/` incl. real-browser tests; synthetic scenarios only |
| S3 | substitutions | NOT_STARTED | - |

## Verification actually run

```
828 tests collected in total (library, experiments, web app), split as:
pytest -q -m "not slow"                          813 passed (incl. 8 browser tests)
pytest -q -m slow                                 15 passed  (exhaustive oracle tier, 4m24s)
pytest -q -m "not slow and not realdata and not e2e"   802 selected  (CI's fast selection)
pytest -m e2e webapp/tests/test_e2e.py             8 passed  (Edge via Playwright; 3 consecutive runs)
pytest --cov=frechet_edit --cov=webapp            94% overall; every library module 96-100%
ruff check src tests experiments benchmarks examples webapp   All checks passed
mypy                                    Success: no issues found in 16 source files
mypy --strict webapp app modules        Success: no issues found in 6 source files
python -m build / twine check           wheel + sdist PASSED; wheel ships the 4 new modules, not webapp/
fresh venv + wheel install              examples and the README continuous example run outside the checkout
```

Mutation testing of the continuous suite: 10 injected bugs; 9 non-equivalent
mutants caught (one only after a test was added for it), 1 provably
equivalent. See `docs/continuous.md` section 5.

The earlier note here recorded "2 passed" for the slow tier, which was the count
before the later slow tests were added; it is corrected above. `pytest-cov` is
now a declared dev dependency, so the coverage figure is reproducible from a
clean checkout rather than from an ad-hoc install.

Environment: Python 3.11.9, NumPy 2.4.6, pytest 9.1.1, Hypothesis 6.168.0,
FastAPI 0.141.1, Playwright 1.63.0 with Microsoft Edge, Windows 11. The CI
changes for the web app (extras `app`/`e2e`, a browser job) have not yet run on
GitHub; the counts above are local.

CI has now been executed for the first time, and its first run **failed**, which
is the point of running it. Two real defects that local development could not
have surfaced:

1. **Python 3.10 was broken on every platform.** `experiments/evaluate.py`
   imported `tomllib` at module scope, which is 3.11+, so importing the module
   raised on 3.10 and collection failed. The package declares
   `requires-python = ">=3.10"`, so this was a genuine unsupported-version bug
   and not a CI misconfiguration. Fixed by parsing TOML lazily inside the CLI
   entry point, where it is the only thing that needs a parser.
2. **The strict type check was broken by a dependency, not by this code.** NumPy
   2.5's own stubs use PEP 695 `type` statements, which mypy can only parse when
   targeting Python 3.12 or later, so `python_version = "3.10"` failed inside
   numpy before reaching this package. The type target is now 3.12; actual 3.10
   support is verified by running the suite on 3.10, which is the check that
   tests the claim.

Both were reproduced locally in a clean interpreter before being fixed, not
guessed at from the CI summary. **CI is now green on all nine matrix jobs**
(Linux/Windows/macOS x Python 3.10/3.11/3.12), plus the slow oracle tier and the
lint/type/build job.

The three `realdata` tests need a locally provisioned archive and are deselected
in CI by marker, so a missing dataset can never read as a passing real-data
gate.

## Findings about the published algorithm

**The published insertion and mixed recurrences are unsound (severity: CRITICAL
for anyone transcribing them).** Re-checked against the SoCG version of record
and the arXiv full version: both display the unrestricted `IedDP(i-1, j)` as the
vertical predecessor of the keep branch, so the counterexample below applies to
the publication and not merely to this project's earlier shorthand. Verified by
three implementations sharing no recurrence - a verbatim transcription of the
published form, a definitional brute force with no dynamic program, and this
package. Over 23480 comparisons the published form was wrong 294 times, always
an under-estimate; this package was wrong zero times. The published deletion
recurrence is sound and was confirmed so. Theorems 20 and 21 are NOT refuted:
the layered correction keeps the same `O(m^2 + mn)` bound. Full statement in
`docs/errata-insertion-recurrence.md`; evidence in
`tests/property/test_published_recurrence.py`. The authors have not been
contacted.

## Independent review actually obtained

**The erratum claim only.** One independent reviewer examined
`docs/errata-insertion-recurrence.md` and returned **CONFIRMED**. What it
actually did, which is the part that matters:

* re-derived the true optimum of the witness by hand from the coupling
  definition, keeping the inserted point symbolic and covering both insertion
  positions, so the impossibility of one insertion is shown for every real `x`
  rather than for sampled candidates;
* fetched the LIPIcs **conference PDF** and extracted its text directly, rather
  than trusting this repository's quotations, and matched the displayed
  recurrence and base cases character-for-character against
  `tests/oracles/published_recurrence.py`;
* looked for a repair elsewhere in the paper - later passage, section 5.3,
  footnote, appendix, the proof of Theorem 20 - and found none;
* attacked the brute force's completeness, including proving and then
  empirically testing the `max_ins = m` cap against a 3x wider cap over 250
  random instances, with no mismatch;
* ran its own adversarial instances in 1-D, 2-D and mixed mode, plus a
  300-instance randomised search, finding no package disagreement.

It raised four issues, all now fixed: the "gap is 1" reading (the error is in
fact unbounded, section 7.1), shared geometric primitives between the two
oracles (now literally independent), and stale pre-resolution framing in
`limitations.md` and `recurrences.md`.

**This does not promote any phase to PASSED.** It reviewed one claim, not the
layered recurrence's correctness argument, not the geometry kernel, and not the
package as a whole. Two earlier reviews aimed at those targets were terminated
by API rate limits before reaching a verdict and have not been rerun. Everything
in the table above therefore stays at READY_FOR_REVIEW.

**The continuous deletion implementation (S2).** One independent adversarial
reviewer, briefed to find correctness bugs rather than style, read
`_continuous`, `_freespace`, `_quadroot`, `_interval`, the underflow change in
`_numerics`, the verifier and the oracle, then ran about 29000 differential
cases against the brute-force oracle and the independent verifier: integer
grids with tangent deltas, uniform floats in 1-3 dimensions, duplicate and
zero-length segments, the waiting-at-the-last-vertex family and a mirrored
waiting-at-the-first-vertex variant it constructed, power-of-two rescaling to
`2^-535` and `2^500`, budgets up to about 30 deletions at `m, n` up to 40.
For every optimal cost it also re-ran the solver at `cost - 1` and confirmed
no solution, a minimality check independent of oracle agreement. It
re-derived the ring-buffer sizing and the traceback's covering argument by
hand. Result: **no CRITICAL, HIGH or MEDIUM findings, no open suspicions.** Its
own caveat, recorded here: oracle-checked cases were bounded to `n <= 7`, and
coordinate distributions were simple. This promotes nothing to PASSED.

**The Course Check web app (APP), security.** One independent security
reviewer attacked `webapp/` with live requests against a running server. It
found three problems, all now fixed with regression tests:

1. CRITICAL: an algorithmic denial of service. Douglas-Peucker course
   simplification is O(n^2) on a square wave; a 50 000-point course inside
   every documented limit took 315 s of CPU, and the two-slot semaphore
   bounded the number of audits but not their duration. Each attempt now
   stops once it keeps too many points, tolerances are tried from largest to
   smallest, and all attempts share a work budget. The same request now
   completes in 0.7 s over real HTTP.
2. HIGH: the body-size cap read only `Content-Length`, so a chunked 150 MB
   body was buffered whole. Bytes are now counted as received; a 524 MB
   chunked push peaked the server at 84 MB and got 413.
3. MEDIUM: multipart uploads are CORS "simple" requests, so another site
   could drive a visitor's browser to POST here. Cross-site browser POSTs are
   now refused by `Sec-Fetch-Site`.

It verified by testing that XXE and DOCTYPE bypasses (lowercase, comment
split, UTF-16), header injection, static path traversal, frontend XSS and
error leakage are not exploitable, and that the CSP is sound.

## Findings about the published algorithm: continuous deletion

The prose of Section 4.1 propagates reachability cell by cell. Read literally,
that loses a path that waits at a DAG vertex with no outgoing edge (the copies
of the last observation vertex) while the reference advances: `<0, 1, 0>`
against `<9, 0.5>` at `delta = 0.5` has optimum 1 and a cells-only reading
reports it infeasible. The implementation propagates product vertices
explicitly (`docs/continuous.md` section 2.1), and mutation testing confirms a
cells-only variant fails the pinned test. This is an omission in the prose
about an edge case, not a flaw in Theorem 3, and is **not** claimed as an
erratum.

## Defects found in the TESTS (not the package)

**A property test asserted something untrue of floating point (fixed).**
`test_common_translation_preserves_cost` asserted that translating both curves
by a common offset leaves the cost unchanged. Hypothesis eventually found
`ref = [(0, 0)]`, `obs = [(1.4e-45, 1)]`, `delta = 1`, `shift = (1, 0)`, where
the cost is 2 before the shift and 0 after. Both answers are CORRECT: exactly,
`dist^2 = 1 + (1.4e-45)^2 > 1` before, while `1.4e-45 + 1.0` rounds to `1.0`,
so after the shift the points sit exactly on `delta` and the closed comparison
admits them. float64 addition is lossy, so the translated input is a different
geometry rather than the same one moved, and the invariance was never a
property of the measure. Both the translation and scaling tests now require the
transform to be exact - verified against rational arithmetic - before asserting
anything about it, and the scaling test draws powers of two so the requirement
costs no filtering. The instance is pinned in
`tests/unit/test_numeric_boundary.py::TestALossyTranslationIsNotTheSameProblem`
so that the behaviour is not later "fixed" into being wrong.

This one is worth noting for what it says about the rest: the package's exact
predicate policy is what made the two cases distinguishable at all. In pure
float64 both configurations compare equal.

## Defects found and fixed during this work

1. **Unsound collapsed recurrence (severity: CRITICAL, fixed).** The collapsed
   single-table form whose keep branch uses the unrestricted `F(i-1, j)` as its
   vertical predecessor underestimates the optimum once insertions are allowed.
   Found by exhaustive equivalence testing, confirmed against an independent
   oracle. Minimal counterexample `R=[0,1,0]`, `Q=[0]`, `delta=0.4`,
   insert-only: collapsed says 1, truth is 2. The layered `K`/`P`/`X` form is
   used throughout and is proved correct in `docs/recurrences.md` section 2.
   `docs/recurrences.md` section 3 was amended; the earlier claim that the two
   forms are equivalent in all modes was **wrong** and is now marked as such.

2. **Certified geometry was needlessly slow (severity: MEDIUM, fixed).**
   `meb_radius_le` converted whole blocks to exact rationals on every call.
   Restructured so the NO certificate uses at most three points and the YES
   certificate is a vectorised float64 check with a rigorous error bound. Exact
   arithmetic over the full block now runs only for genuinely boundary cases.
   Mixed mode at `m = n = 400`, `delta = 10`: 514 ms to 367 ms.

3. **The certified point predicate was not certified near underflow
   (severity: CRITICAL, fixed).** Tier 1 accepted a float decision whenever
   the squared distance and `delta**2` were separated by a relative margin.
   Once those squares are subnormal (coordinates near `1e-162`) a rounding
   error is an absolute half-ulp of the smallest subnormal, which can be most
   of the value, and the margin certifies nothing. A 283849-case probe against
   exact arithmetic found 1555 confident wrong decisions; end to end, two
   single-point curves farther apart than `delta` got `cost=0,
   status="optimal"` in every mode, where the truth is infeasible,
   infeasible and 2. Found while reviewing the numerics for the continuous
   work, not by the existing suite: every earlier test lived in the normal
   range. Fixed by an absolute underflow allowance in `_numerics.float_tier`,
   shared with the ball predicate's YES certificate; overflowed squares now go
   to the exact tier. Re-probed at 270000 cases from `1e-300` to `1e154` with
   no disagreement, and an independent reviewer's 22000 targeted cases found
   none. Pinned in
   `tests/unit/test_numeric_boundary.py::TestUnderflowDoesNotDefeatTheErrorBound`,
   including an exact power-of-two rescale into the subnormal range that must
   leave every cost unchanged.

## Known negative and neutral results (not defects)

* In the level A pilot, `fed_both` **ties** EDR, DTW and ERP at ceiling and
  does not beat them. It does beat raw discrete Frechet by a wide margin
  (Recall@1 1.000 vs 0.188). The regime does not discriminate the top methods.
* `fed_insert` abstains on every spiked query, because insertion cannot remove
  an outlier. Reported as coverage 0.00, not as a ranking failure.
* In deletion-only mode the optimised backend is slightly SLOWER than the
  reference backend (82 ms vs 65 ms at `m = n = 400`), since deletion never
  uses the `mu` window. Reported rather than omitted.

## What an independent reviewer should attack first

1. The completeness argument for the layered form in `docs/recurrences.md`
   section 2, especially the claim that an inserted point is always coupled to
   a contiguous reference block.
2. Oracle independence: confirm `tests/oracles/` shares no recurrence with
   `src/frechet_edit/`, and that the Oracle B candidate-completeness lemma
   (`docs/oracle-protocol.md` B.1) and insertion bound (B.2) are sound.
3. The numerical error bound in `_numerics._gamma` and the YES/NO certificates
   in `_geometry.meb_radius_le`. A wrong bound would silently accept wrong
   answers, and no test can catch what the bound itself gets wrong.
4. Whether the published proofs carry the "ends with sigma_j" restriction that
   this project's collapsed shorthand dropped. Recorded as an open item in
   `docs/paper-map.md`; the authors have **not** been contacted.
5. The retrieval metrics in `experiments/metrics.py`, particularly the tie-aware
   expectations, against the hand-computed values in `experiments/test_metrics.py`.

## Claims policy

Any public claim about this work must stay inside what this file and
`docs/limitations.md` support. In particular: no "first implementation" claim
(the search for prior implementations was negative, not exhaustive); no claim
that the authors have confirmed the erratum (the first author was contacted in
September 2026); and no claim about real-world route matching beyond evidence
level B.
