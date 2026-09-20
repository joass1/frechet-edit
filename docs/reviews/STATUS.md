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
| S2 | continuous deletion | NOT_STARTED | free-space RENDERING only, `experiments/freespace_viz.py`; no continuous edit algorithm |
| S3 | substitutions | NOT_STARTED | - |

## Verification actually run

```
577 tests collected in total, split as:
pytest -q -m "not slow"                     567 passed, 10 deselected
pytest -q -m slow                            10 passed  (exhaustive oracle tier)
pytest -q -m "not slow and not realdata"    564 passed, 13 deselected  (CI's selection)
pytest --cov=frechet_edit                    99% line coverage over src/frechet_edit
ruff check src tests experiments benchmarks examples   All checks passed
mypy                          Success: no issues found in 12 source files
python -m build               wheel + sdist
python -m twine check dist/*  PASSED
fresh venv + wheel install    examples run from outside the checkout
```

The earlier note here recorded "2 passed" for the slow tier, which was the count
before the later slow tests were added; it is corrected above. `pytest-cov` is
now a declared dev dependency, so the coverage figure is reproducible from a
clean checkout rather than from an ad-hoc install.

Environment: Python 3.11.9, NumPy 2.4.6, pytest 9.1.1, Hypothesis 6.168.0,
Windows 11.

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

No resume, publication, or public novelty claim has been made or authorised.
Nothing has been published to any package index, no repository has been created,
and the authors of the paper have not been contacted. See `docs/limitations.md`.
