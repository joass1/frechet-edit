# Continuous deletion-only Fréchet edit distance

Status: implemented, oracle-checked, READY_FOR_REVIEW. Source: Fox, Nayyeri,
Perry and Raichel, *Fréchet Edit Distance*, SoCG 2024, Section 4.1 and
Theorem 3 (arXiv:2403.12878, same section).

## 1. Contract

```python
continuous_edit_distance(reference, observation, delta, *,
                         operations="delete", return_witness=False,
                         max_deletions=None) -> EditResult
continuous_frechet_within(a, b, delta) -> bool      # ordinary d_F <= delta
verify_continuous_witness(reference, observation, result) -> VerificationReport
```

* Both inputs are **polygonal curves**: linear between consecutive vertices.
  `d_F` is the strong continuous Fréchet distance of Alt and Godau.
* The cost is the fewest vertices of `observation` whose deletion leaves a
  **non-empty** curve `Q'` with `d_F(reference, Q') <= delta`. Deleting every
  vertex is not a solution, so the answer can be `infeasible` (`cost = inf`).
* A single retained vertex is a curve; its distance to a polyline is the
  farthest vertex of that polyline, by convexity.
* Only `operations="delete"`. `"insert"` and `"both"` raise
  `UnsupportedOperationError`: the paper's continuous insertion needs
  minimum-link machinery (`O(n m^5)` in the plane) that is not implemented,
  and nothing silently falls back to deletion or to the discrete measure.
* Any dimension. `delta` finite and strictly positive; comparisons closed.
* `max_deletions` caps the search. If no solution uses at most that many
  deletions the status is `budget_exceeded` with `cost=None`. That is **not**
  a claim of infeasibility, and it is never reported as one.
* `numeric_policy` is always `"certified"`; there is no abstention, because
  every comparison below is decided exactly.

Why it exists alongside the discrete measure: discrete Fréchet matches
vertices to vertices, so a sparse reference polyline against a densely sampled
observation of the *same* path is far apart. In `benchmarks`-style GPS
scenarios (a 20-60 vertex route against 100-600 noisy fixes with a few
injected spikes) discrete deletion needed 148-290 deletions or was infeasible,
while continuous deletion found the injected spikes exactly.

## 2. Algorithm (paper Section 4.1)

The observation `sigma` becomes the **complete weighted DAG complex** for a
budget `k`: vertex copies `(j, l)`, meaning "at observation vertex `j`, having
deleted `l` vertices so far", and an edge `(i, l) -> (j, l + j - i - 1)` for
every `i < j` with `l + j - i - 1 <= k`. Start vertices are `(j, j)` (the first
`j` vertices deleted) paired with `pi_1`; reaching `(j, l)` paired with `pi_m`
costs `l + (n - 1 - j)`. Reachability is propagated through the free space of
the product of this complex with `pi` (the paper's Theorem 2), and the answer
for budget `k` is the cheapest reachable end vertex.

The solver searches budgets `0, 1, 2, 4, ...`. A budget-`k` run returns the
**exact** optimum whenever it is at most `k`, so no bisection is needed. The
discrete deletion distance is computed first (`O(mn)`, certified) and caps the
search, since a subsequence within discrete Fréchet `delta` is within
continuous Fréchet `delta` too. Budget `0` is the ordinary continuous Fréchet
decision (`continuous_frechet_within`).

Complexity: `O(k^2 m n)` per budget, as in Theorem 3; `O(m n^3)` in the worst
case, when `k` approaches `n`. Only `k + 3` wavefronts of state are kept for
the cost; a witness records the full `(m, n, k+1, k+1)` table of the final run
and is refused above `CONTINUOUS_WITNESS_MAX_BYTES` (512 MiB), in which case
the exact cost is still returned with `witness_status="unavailable"`.

### 2.1 Deviation: explicit vertex propagation

The paper propagates reachability **cell by cell**, initialising the edges
leaving each start vertex. Read literally, that loses a path that *waits* at a
DAG vertex with no outgoing edge (the copies of the last observation vertex)
while the reference advances past one of its own vertices: no cell carries the
horizontal edges of such a vertex beyond the first. The smallest instance:

```
reference = <0, 1, 0>,  observation = <9, 0.5>,  delta = 0.5
only solution: delete 9, keep <0.5>     cost 1
cells-only reading                      infeasible
```

This implementation propagates product **vertices** explicitly: a vertex is
reachable if it is free and is a start, or ends a reachable horizontal or
vertical edge, and a reachable vertex makes each outgoing edge fully
reachable (its free interval contains the vertex, and is convex). For every
vertex that has an outgoing edge this adds nothing new: waiting there is
already carried along the edge's `t = 0` end. The pinned test is
`tests/unit/test_continuous.py::TestPaperEdgeCases::test_waiting_at_the_last_vertex_through_several_reference_vertices`,
and mutation testing confirms the cells-only variant fails it. This is a gap
in the prose description of an edge case, not a flaw in Theorem 3, and is not
claimed as an erratum.

### 2.2 Evaluation order

Cell `(a, j)` depends only on `(a, i < j)` and `(a - 1, j)`, so every cell with
the same `a + j` is independent. The solver processes one such anti-diagonal
per step, vectorised over its cells, the `k + 1` gaps and the `k + 1` copies.

## 3. Witnesses

With `return_witness=True` the final run records its reachability table and
`traceback` walks it backwards from the chosen end vertex. At each edge it
moves to the contributor that produced the edge's **lowest** reachable rank.
Every contribution to an edge reaches up to the same free upper endpoint, so
that contributor alone covers the whole reachable part, and the walk never
needs to track a specific point. The retained observation vertices are those
the walk visits; the API cross-checks that their number matches the cost.

A witness is independently verified by `verify_continuous_witness` (also
reached through `verify_witness`), which replays the deletions and decides
`d_F <= delta` on the edited curve with its own exact Alt-Godau sweep. That
sweep shares no code with the solver and uses a different exact comparison
technique (section 4).

## 4. Numerics

Every free-interval endpoint is `0`, `1`, or a root of a quadratic with float64
(hence rational) coefficients, `x ± sqrt(y)` with `x`, `y` rational. The
propagation only ever compares endpoints lying on the **same** segment, so each
segment's endpoints are ranked once, exactly, and the propagation compares
integers.

* **Vertex freedom** uses the package's certified point predicate
  (`docs/numerics.md`), including its underflow allowance.
* **Enclosures.** `_interval` computes every root with outward-rounded
  interval arithmetic. IEEE round-to-nearest is within half an ulp, so one
  `nextafter` step outward after every operation is a rigorous bound; `nan`
  from overflow widens to the whole line. A whole-line enclosure is never
  wrong, it only forces the exact tier.
* **Exact tier.** Enclosures that overlap are ordered by
  `_quadroot.compare`, which decides `sign(x1 + s1 sqrt(y1) - x2 - s2 sqrt(y2))`
  by squaring with sign bookkeeping, in rational arithmetic. Equal values get
  equal ranks, which the closed comparisons require: tangencies and
  single-point free intervals are common on integer inputs, and several tests
  pin them.
* **Emptiness** of an interval whose endpoints are both outside the ball
  (`D >= 0` and `0 < -B < A`) is decided on the enclosures when they are
  conclusive, exactly otherwise.

The independent checkers use a different technique: they normalise rational
square roots, declare two irrational `x + s sqrt(y)` equal only when the
triples coincide (anything else would make a root rational), and order unequal
values by integer-square-root refinement. A shared wrong answer would need two
different wrong proofs.

## 5. Evidence

| check | where |
|---|---|
| exact comparator vs 120-digit decimal, constructed ties | `tests/unit/test_continuous.py::TestQuadRootComparison` |
| interval enclosures contain exact values, incl. near underflow and overflow | `TestIntervalEnclosures` |
| exact dense ranking vs exact sort, forced overlaps | `TestDenseRanks` |
| ordinary decision vs oracle and verifier, exhaustive 1-D, random 2-D with tangencies | `TestOrdinaryDecision` |
| deletion cost vs subsequence-enumeration oracle, exhaustive 1-D (slow tier: all `m <= 3`, `n <= 4`, five deltas), random 2-D and 3-D | `TestDeletionAgainstTheOracle` |
| paper edge cases: waiting at the last vertex, `m = 1`, `n = 1`, prefix and suffix deletion, degenerate segments | `TestPaperEdgeCases` |
| never more than discrete deletion; zero iff ordinary `d_F <= delta`; non-increasing in delta; reversal invariance; exact rescaling by `2^-535` | `TestProperties` |
| tampered witnesses rejected | `TestWitnessRejection` |
| oracle self-checks: closed forms, `d_F <= d_DF`, dense-resampling convergence, symmetry | `tests/oracles/test_continuous_oracle_selfcheck.py` |

**Mutation testing.** Ten deliberate bugs were injected into a copy of the
package and the fast suite was run against each. Eight were caught at once,
including the cells-only reading of Section 4.1, a miscounted suffix, a wrong
copy offset, an early exit before every start vertex is seen, ranking equal
endpoints as distinct, and trusting the float emptiness test. Two survived:

* a **strict** comparison inside a cell (`>` for `>=`). That was a real gap in
  the tests, and
  `test_a_path_through_a_single_point_needs_the_closed_cell_comparison` was
  added because of it; the mutant is now caught.
* making the exact interior test non-strict. That is an **equivalent
  mutant**: if neither endpoint is free, the parabola's vertex cannot sit at
  `t = 0` or `t = 1`, because that would force `f(0) <= 0` or `f(1) <= 0`. No
  test can distinguish it.

So all nine non-equivalent mutants are now caught.

## 6. Limits

* Deletion only. Continuous insertion and mixed edits are not implemented.
* `O(k^2 m n)`: practical for budgets up to a few tens at a few hundred
  vertices; `max_deletions` bounds the running time.
* Weak (non-monotone) variants are NP-hard in the paper and are not attempted.
