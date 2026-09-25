# frechet-edit

**Strong Fréchet edit distance, discrete and continuous, with verifiable minimal-edit witnesses.**

How many vertices must you insert into or delete from a noisy observed curve to
bring it within `δ` of a reference curve, and *which* ones?

```python
import numpy as np
from frechet_edit import discrete_edit_distance, verify_witness

reference   = np.array([[0.0], [1.0], [2.0]])
observation = np.array([[0.0], [1.0], [100.0], [2.0]])   # one spike

result = discrete_edit_distance(reference, observation, delta=0.1,
                                operations="delete", return_witness=True)

result.cost                       # 1
[e.as_dict() for e in result.edits]   # [{'op': 'delete', 'index': 2}]
verify_witness(reference, observation, result).ok      # True
```

Ordinary discrete Fréchet distance between those two curves is **98**. One bad
sample destroys it. The edit distance instead says: one deletion, and here is
which vertex.

This is an independent implementation of the discrete algorithms, and of the
continuous deletion algorithm, of Fox, Nayyeri, Perry and Raichel, *Fréchet
Edit Distance*, SoCG 2024
([DOI](https://doi.org/10.4230/LIPIcs.SoCG.2024.58),
[arXiv:2403.12878](https://arxiv.org/abs/2403.12878)).

**See it used:** [Course Check](https://github.com/joass1/frechet-edit/blob/main/docs/web-app.md), a web app that audits a GPS
track against a planned course with it, and names the glitches.

## What makes it more than a distance function

1. **All three edit modes** (`delete`, `insert`, `both`), not just
   deletion-based denoising.
2. **Inserted vertices go anywhere.** A single inserted point can cover a whole
   run of reference vertices, found via minimum-enclosing-ball geometry. It is
   *not* restricted to copying reference vertices:

   ```python
   R, Q = np.array([[0.0], [2.0], [4.0]]), np.array([[0.0]])
   r = discrete_edit_distance(R, Q, delta=1.0, operations="insert", return_witness=True)
   r.cost                 # 1  (inserting reference vertices would cost 2)
   r.edits[0].point       # (3.0,)  not a vertex of either curve
   ```
3. **Replayable witnesses.** Every optimal cost can come with an edit script, the
   edited curve, and a coupling, checked by a verifier that shares no code with
   the solver and *rejects tampering*.
4. **Optimality is cross-checked** against brute-force oracles written from the
   definition alone, without sight of the dynamic program.
5. **Certified numerics.** Boundary predicates fall back to exact rational
   arithmetic; genuinely undecidable ones **abstain** rather than guess. `δ` is
   never quietly adjusted.
6. **Continuous curves too.** `continuous_edit_distance` treats both inputs as
   polygonal curves (paper Section 4.1, Theorem 3), so a sparse reference
   polyline and a dense observation of the same path agree, where discrete
   Fréchet forces vertex-to-vertex matches:

   ```python
   from frechet_edit import continuous_edit_distance
   route = np.array([[0.0, 0.0], [10.0, 0.0]])                    # 2 vertices
   gps   = np.array([[0, 0], [2, .1], [4, 0], [5, 30], [6, 0], [8, -.1], [10, 0]])
   continuous_edit_distance(route, gps, 1.0, return_witness=True).edits
   # (Deletion(index=3),)     discrete deletion needs 5 for the same pair
   ```

## Install

```bash
pip install -e .            # from a checkout
pip install -e ".[dev]"     # plus pytest, hypothesis, ruff, mypy
```

Requires Python ≥ 3.10 and NumPy. Pure Python, no compiler.

## API

```python
discrete_edit_distance(
    reference, observation, delta, *,
    operations="both",          # "delete" | "insert" | "both"
    return_witness=False,
    backend="python",           # "python" (rolling storage) | "reference" (full tables)
    numeric_policy="certified", # "certified" | "fast" (NOT certified)
) -> EditResult
```

`EditResult` carries `status` (`"optimal"` / `"infeasible"` /
`"numerically_ambiguous"`), `cost`, `witness_status`, `edited_curve`, `edits`,
`coupling` and `stats`. `status` and `witness_status` are **independent**: a
cost can be certified optimal while no representable certified witness exists,
and that is never reported as infeasibility.

Also exported: `ordinary_discrete_frechet`, `verify_witness`, `replay`.

```python
continuous_edit_distance(
    reference, observation, delta, *,
    operations="delete",        # only "delete"; insertion raises UnsupportedOperationError
    return_witness=False,
    max_deletions=None,         # optional cap; "budget_exceeded" is NOT "infeasible"
) -> EditResult                 # backend="continuous", exact comparisons throughout

continuous_frechet_within(a, b, delta) -> bool     # ordinary continuous Fréchet <= delta
verify_continuous_witness(reference, observation, result)  # also via verify_witness
```

The continuous contract, algorithm and evidence are in
[`docs/continuous.md`](https://github.com/joass1/frechet-edit/blob/main/docs/continuous.md).

Read [`docs/definition.md`](https://github.com/joass1/frechet-edit/blob/main/docs/definition.md) before relying on the
semantics. Three points catch people out:

- **The cost is a count of edits, not a distance.** Never compare it to metres.
- **It is directed.** Only `observation` is edited; swapping arguments asks a
  different question.
- **It is not a metric.** No symmetry, no triangle inequality.

## Documentation

| document | what it covers |
|---|---|
| [definition.md](https://github.com/joass1/frechet-edit/blob/main/docs/definition.md) | the normative public contract |
| [continuous.md](https://github.com/joass1/frechet-edit/blob/main/docs/continuous.md) | the continuous deletion variant: contract, algorithm, exact numerics, evidence |
| [web-app.md](https://github.com/joass1/frechet-edit/blob/main/docs/web-app.md) | **Course Check**: the GPS route-audit web app, how to run it and test it |
| [recurrences.md](https://github.com/joass1/frechet-edit/blob/main/docs/recurrences.md) | the dynamic programs, with proofs |
| [witness-invariants.md](https://github.com/joass1/frechet-edit/blob/main/docs/witness-invariants.md) | witness semantics and replay |
| [numerics.md](https://github.com/joass1/frechet-edit/blob/main/docs/numerics.md) | the three-tier certified predicate policy |
| [oracle-protocol.md](https://github.com/joass1/frechet-edit/blob/main/docs/oracle-protocol.md) | the independent oracles and their completeness proofs |
| [paper-map.md](https://github.com/joass1/frechet-edit/blob/main/docs/paper-map.md) | source traceability and deliberate deviations |
| [performance.md](https://github.com/joass1/frechet-edit/blob/main/docs/performance.md) | measured time and memory |
| [experiment-protocol.md](https://github.com/joass1/frechet-edit/blob/main/docs/experiment-protocol.md) | preregistration and actual outcomes |
| [data-and-labels.md](https://github.com/joass1/frechet-edit/blob/main/docs/data-and-labels.md) | why there is no real data here |
| [results.md](https://github.com/joass1/frechet-edit/blob/main/docs/results.md) | level A pilot results, losses included |
| [results-geolife.md](https://github.com/joass1/frechet-edit/blob/main/docs/results-geolife.md) | level B results on real GeoLife geometry |
| [errata-insertion-recurrence.md](https://github.com/joass1/frechet-edit/blob/main/docs/errata-insertion-recurrence.md) | **the published recurrence is unsound**, with the counterexample |
| [limitations.md](https://github.com/joass1/frechet-edit/blob/main/docs/limitations.md) | **read this one** |
| [reviews/STATUS.md](https://github.com/joass1/frechet-edit/blob/main/docs/reviews/STATUS.md) | phase status, defects found, what to attack |

## A finding worth knowing

**The insertion recurrence as published is unsound.** The "keep" branch takes
the unrestricted `IedDP(i-1, j)` as its vertical predecessor, but an optimal
solution for that prefix may end with a newly *inserted* point, and a monotone
coupling cannot step back past it to reuse `sigma_j`. The recurrence admits
solutions that do not exist, so it can under-report.

```
pi = <0, 1, 0>,  sigma = <0>,  delta = 0.4,  insertions only
published recurrence: 1          true optimum: 2
```

Verified against the SoCG version of record and the arXiv full version, which
display the same recurrence, by three implementations sharing no recurrence: a
verbatim transcription of the published form, a definitional brute force with no
dynamic program, and this package. Across 23480 comparisons the published form
was wrong 294 times, **always an under-estimate**; this package was wrong zero
times. The published *deletion* recurrence is sound, and was confirmed so.

Theorems 20 and 21 are **not** refuted. The fix is to split the DP state by what
the edited prefix ends with - layered `K` / `P` / `X` - which runs in the same
`O(m^2 + mn)` bound. The paper's own parenthetical flags the hazard; a single
table indexed by `(i, j)` simply cannot express the restriction.

Full write-up in
[errata-insertion-recurrence.md](https://github.com/joass1/frechet-edit/blob/main/docs/errata-insertion-recurrence.md); run the
evidence with `pytest tests/property/test_published_recurrence.py`. The first
author was contacted about it in September 2026.

## Free-space diagrams

`experiments/freespace_viz.py` renders the continuous Fréchet free-space diagram,
which is the clearest picture of why a single outlier is so destructive: one
spike severs the free band, so no monotone path exists and ordinary Fréchet
reports a large distance, even though the curves agree everywhere else.

![Free-space diagram with one outlier vertex](https://raw.githubusercontent.com/joass1/frechet-edit/main/docs/images/freespace_spike.png)

Green is reachable from the origin, cream is free but unreachable, dark is
blocked. Regenerate with `python -m experiments.freespace_viz`. This draws the
*continuous* distance. Of the continuous **edit** variants, deletion is
implemented (`continuous_edit_distance`); insertion and mixed are not.

## Course Check: a real-world use

![Course Check auditing a run with six GPS spikes](https://raw.githubusercontent.com/joass1/frechet-edit/main/docs/images/course-check.png)

A web app that answers *did this GPS track follow that course, in order, and
which fixes were glitches?* with continuous Fréchet edit distance, next to the
order-blind and glitch-intolerant checks that fail. Five synthetic scenarios
around Marina Bay, Singapore, each isolate one failure: urban-canyon spikes,
sparse logging, a shortcut, and a two-lap race run as one lap, which passes
both naive checks and is still caught. Or upload your own GPX, GeoJSON or CSV.

```bash
pip install -e ".[dev,app]"
python -m webapp             # http://127.0.0.1:8000
```

Walkthrough, API, limits and tests: [docs/web-app.md](https://github.com/joass1/frechet-edit/blob/main/docs/web-app.md).

## Tests

```bash
pytest -q -m "not slow"                  # fast suite: library, web app, and browser tests
pytest -q -m "not slow and not e2e"      # the same without a browser
pytest -q -m slow                        # exhaustive oracle tier (~4 min)
pytest -m e2e webapp/tests/test_e2e.py   # only the browser tests (needs .[e2e] and Edge/Chrome)
ruff check src tests experiments benchmarks examples webapp && mypy
```

The suite includes exhaustive oracle agreement over all 1-D instances up to
`m ≤ 4`, `n ≤ 3`, property-based invariants, geometric degeneracy fixtures
(collinear, cocircular, obtuse-triangle MEB, exact-boundary `δ`), and witness
tamper-rejection tests.

## Status and honest scope

Alpha. The **library** is the deliverable and it is tested. The empirical side
is deliberately limited:

- Evidence reaches **level B**: real GeoLife trajectory geometry, with ground
  truth known by construction from the injected corruption. That supports
  "recovers the source trajectory under corruption"; it does **not** support any
  claim about matching routes in the wild. **Level C** - blinded route-identity
  annotation - is still blocked, and no dataset supplies those labels.
- No trajectory data is in this repository and none may be. GeoLife's licence
  permits non-commercial research and forbids redistributing the data *or any
  derivative work*. Bring your own archive; see
  [data-and-labels.md](https://github.com/joass1/frechet-edit/blob/main/docs/data-and-labels.md).
- On real geometry, FED **tied** EDR, DTW and ERP at ceiling and did **not**
  beat them. It beat raw discrete Fréchet decisively (Recall@1 1.000 vs 0.115).
  `fed_insert` abstained on every query, because insertion cannot remove an
  outlier; that is reported as coverage 0.00, not hidden.
- Continuous **deletion** is implemented and oracle-checked
  ([continuous.md](https://github.com/joass1/frechet-edit/blob/main/docs/continuous.md)). Continuous insertion and mixed edits,
  weak variants and substitutions are **not implemented**.
- The Course Check web app runs on **synthetic** scenarios with ground truth by
  construction. It demonstrates the method; it is not evidence that the method
  is right about real athletes or vehicles.

See [limitations.md](https://github.com/joass1/frechet-edit/blob/main/docs/limitations.md). No claim is made that this is the
first or only implementation of these algorithms.

## Licence

MIT for the original code in this repository; see [LICENSE](https://github.com/joass1/frechet-edit/blob/main/LICENSE). The
licence of the paper does not license other software; the algorithms are
implemented here from their published description.
