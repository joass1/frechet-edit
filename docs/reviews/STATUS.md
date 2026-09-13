# Phase status and evidence

Status vocabulary: NOT_STARTED, IN_PROGRESS, READY_FOR_REVIEW, BLOCKED,
FAILED, PASSED. **Nothing is promoted to PASSED without an independent
review**, so everything below is READY_FOR_REVIEW at best regardless of how
green the test suite is. A passing suite is evidence, not a signoff.

| phase | scope | status | evidence |
|---|---|---|---|
| P0 | contract, recurrences, witness invariants, numerics, oracle protocol, environment | READY_FOR_REVIEW | `docs/definition.md`, `docs/recurrences.md`, `docs/witness-invariants.md`, `docs/numerics.md`, `docs/oracle-protocol.md`, `docs/paper-map.md` |
| P1 | deletion-only spine, witnesses, verifier | READY_FOR_REVIEW | `tests/unit/test_deletion.py`, `test_witness.py`, exhaustive Oracle A agreement |
| P2 | real-data pilot | **BLOCKED** | no licensed dataset present; see `docs/data-and-labels.md` |
| P3 | insertion geometry, mixed edits, oracles | READY_FOR_REVIEW | `tests/unit/test_geometry.py`, `test_insertion.py`, `test_mixed.py`, Oracle B agreement |
| P4 | optimised backend, measured performance | READY_FOR_REVIEW | `docs/performance.md`, `benchmarks/`, `tests/unit/test_minqueue.py` |
| P5 | held-out application validation | **BLOCKED** at levels B and C; level A done | `docs/experiment-protocol.md`, `runs/pilot/report.md` |
| P6-L | library release | READY_FOR_REVIEW | wheel + sdist built, fresh-env install verified, `tests/integration/` |
| P6-A | application-evidence release | **NOT CLAIMED** | requires P2/P5 evidence that does not exist |
| S1 | native backend | NOT_STARTED | - |
| S2 | continuous deletion | NOT_STARTED | - |
| S3 | substitutions | NOT_STARTED | - |

## Verification actually run

```
pytest -q -m "not slow"     398 passed, 4 deselected
pytest -q -m slow             2 passed  (exhaustive oracle tier, ~87 s)
ruff check src tests experiments benchmarks examples   All checks passed
mypy                          Success: no issues found in 11 source files
python -m build               wheel + sdist
python -m twine check dist/*  PASSED
fresh venv + wheel install    examples run from outside the checkout
```

Environment: Python 3.11.9, NumPy 2.4.6, pytest 9.1.1, Hypothesis 6.168.0,
Windows 11. CI is configured for Linux/Windows/macOS on Python 3.10-3.12 but
**has not been executed** - no CI run exists yet.

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
