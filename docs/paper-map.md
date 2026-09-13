# Source traceability map

Primary source: Kyle Fox, Amir Nayyeri, Hannah Miller Perry (Perry), Benjamin
Raichel, *Fréchet Edit Distance*, SoCG 2024, LIPIcs vol. 293, 58:1-58:15,
DOI 10.4230/LIPIcs.SoCG.2024.58. Full version: arXiv:2403.12878.

This package is an INDEPENDENT implementation written from the published
description. No code from the authors was used; this planning pass found no
paper-specific public implementation, which is a negative search result and not
a claim that none exists.

| Topic | SoCG version | arXiv full version | Code | Tests |
|---|---|---|---|---|
| Definition, direction, unit edit costs | Sec. 2 | Sec. 2 | `docs/definition.md`, `api.py` | `tests/unit/test_contract.py` |
| Discrete deletion DP | Sec. 5.1, Thm. 9 | Sec. 5.1, Thm. 17 | `_reference_dp.py` branch `X`/`K` | `tests/unit/test_deletion.py` |
| Insertion geometry `mu(i)` | Sec. 5.2 | Sec. 5.2, Lem. 18 | `_geometry.py: mu_indices` | `tests/unit/test_geometry.py` |
| Amortised minimum queue | abbreviated | Sec. 5.2, Lem. 19 | `_minqueue.py` | `tests/unit/test_minqueue.py` |
| Discrete insertion | Thm. 10 | Thm. 20 | `_reference_dp.py` branch `P` | `tests/unit/test_insertion.py` |
| Discrete mixed edits | Thm. 11 | Sec. 5.3, Thm. 21 | `_reference_dp.py` (all branches) | `tests/unit/test_mixed.py` |
| Continuous deletion | Sec. 4.1, Thm. 3 | numbering to re-verify | NOT IMPLEMENTED (stretch S2) | - |
| Weak variants (NP-hardness) | Sec. 3 / 6 | Sec. 3 / 6 | NOT IMPLEMENTED, out of scope | - |
| Substitutions | deferred by the authors | not supplied | NOT IMPLEMENTED (stretch S3) | - |

## Stated asymptotics (targets, not measurements)

- discrete deletion: `O(mn)`
- discrete insertion / mixed, fixed dimension: `O(m^2 + mn)`

Measured behaviour is reported separately in `docs/performance.md`. A running-
time plot is not a proof of an asymptotic bound.

## Deliberate deviations

1. **Layered states (REQUIRED, not cosmetic).** The implementation splits the
   prefix value `F(i,j)` into `K/P/X` by the identity of the last edited
   element. This began as a traceback convenience, but P3 established that it
   is necessary for CORRECTNESS in insertion and mixed modes: the collapsed
   single-table form, whose keep branch takes the unrestricted `F(i-1,j)` as
   its vertical predecessor, is unsound. The minimal counterexample is
   `R = [0, 1, 0]`, `Q = [0]`, `delta = 0.4`, insert-only, where the collapsed
   form returns 1 and the true optimum, confirmed by the independent oracle, is
   2. See `docs/recurrences.md` Sec. 3.2 and
   `tests/unit/test_equivalence.py::TestCollapsedFormIsUnsoundWithInsertions`.
   For deletion-only the two forms do agree, and that is checked exhaustively.
2. **Dimension scope.** Insertion and mixed modes are restricted to d in {1,2}
   because only those minimum-enclosing-ball backends are certified here. The
   theorem is not dimension-restricted in that way.
3. **Numerical policy.** The paper works in a real-RAM model. This
   implementation adds an explicit certified/abstaining predicate policy
   (`docs/numerics.md`); abstention is a reported status, not an answer.

## Open questions (not resolved by guessing)

- Whether a journal version supplies the substitution recurrence. Unresolved;
  S3 remains a derivation project, not a transcription.
- **How the source states the vertical predecessor of the keep branch.** This
  project's collapsed shorthand used `F(i-1,j)` and is demonstrably unsound
  (deviation 1). That is a refutation of THIS project's transcription, not a
  claimed erratum in the publication: the published DP state may already carry
  the "edited prefix ends with sigma_j" restriction in prose or in its state
  definition. To be re-checked against the full version before any public
  statement. The implementation is unaffected either way, because the layered
  form is proved correct independently and agrees with the oracle.
- Exact full-version theorem numbering for the continuous deletion result.
  Recorded as "to re-verify" rather than asserted.

Author correspondence has NOT been sent. Any such message requires the
execution user's explicit authorisation.
