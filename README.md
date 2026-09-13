# frechet-edit

**Strong discrete Fréchet edit distance, with verifiable minimal-edit witnesses.**

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

This is an independent implementation of the discrete algorithms of Fox,
Nayyeri, Perry and Raichel, *Fréchet Edit Distance*, SoCG 2024
([DOI](https://doi.org/10.4230/LIPIcs.SoCG.2024.58),
[arXiv:2403.12878](https://arxiv.org/abs/2403.12878)).

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

Read [`docs/definition.md`](docs/definition.md) before relying on the
semantics. Three points catch people out:

- **The cost is a count of edits, not a distance.** Never compare it to metres.
- **It is directed.** Only `observation` is edited; swapping arguments asks a
  different question.
- **It is not a metric.** No symmetry, no triangle inequality.

## Documentation

| document | what it covers |
|---|---|
| [definition.md](docs/definition.md) | the normative public contract |
| [recurrences.md](docs/recurrences.md) | the dynamic programs, with proofs |
| [witness-invariants.md](docs/witness-invariants.md) | witness semantics and replay |
| [numerics.md](docs/numerics.md) | the three-tier certified predicate policy |
| [oracle-protocol.md](docs/oracle-protocol.md) | the independent oracles and their completeness proofs |
| [paper-map.md](docs/paper-map.md) | source traceability and deliberate deviations |
| [performance.md](docs/performance.md) | measured time and memory |
| [experiment-protocol.md](docs/experiment-protocol.md) | preregistration and actual outcomes |
| [data-and-labels.md](docs/data-and-labels.md) | why there is no real data here |
| [results.md](docs/results.md) | the actual pilot results, losses included |
| [limitations.md](docs/limitations.md) | **read this one** |
| [reviews/STATUS.md](docs/reviews/STATUS.md) | phase status, defects found, what to attack |

## A finding worth knowing

While building this, an exhaustive test plus an independent oracle refuted a
plausible-looking *collapsed* form of the recurrence: the one whose "keep"
branch takes the unrestricted `F(i-1, j)` as its vertical predecessor. It is
**unsound** once insertions are allowed.

```
R = [0, 1, 0],  Q = [0],  delta = 0.4,  insert-only
collapsed form: 1        layered form: 2        oracle: 2
```

The state must record whether the edited prefix *ends with* the observation
vertex being kept. The implementation therefore splits the DP into layered
`K` / `P` / `X` states. Details, counterexample and the open source-verification
item are in [`docs/recurrences.md` §3.2](docs/recurrences.md).

## Tests

```bash
pytest -q -m "not slow"    # fast suite
pytest -q -m slow          # exhaustive oracle tier (~90 s)
ruff check src tests && mypy
```

The suite includes exhaustive oracle agreement over all 1-D instances up to
`m ≤ 4`, `n ≤ 3`, property-based invariants, geometric degeneracy fixtures
(collinear, cocircular, obtuse-triangle MEB, exact-boundary `δ`), and witness
tamper-rejection tests.

## Status and honest scope

Alpha. The **library** is the deliverable and it is tested. The empirical side
is deliberately limited:

- Only **synthetic** evidence exists (level A). No real trajectory data was used
  or downloaded. GeoLife and T-Drive forbid redistribution of the data *and*
  of derivative works.
- In the pilot, FED **tied** EDR, DTW and ERP at ceiling and did **not** beat
  them. It beat raw discrete Fréchet decisively (Recall@1 1.000 vs 0.188).
- Continuous variants, weak variants and substitutions are **not implemented**.

See [limitations.md](docs/limitations.md). No claim is made that this is the
first or only implementation of these algorithms.

## Licence

MIT for the original code in this repository; see [LICENSE](LICENSE). The
licence of the paper does not license other software; the algorithms are
implemented here from their published description.
