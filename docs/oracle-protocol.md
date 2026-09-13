# Independent oracle protocol (P0)

Status: APPROVED for P1 (Oracle A) and P3 (Oracles B, C).

An oracle exists to DISAGREE with the production code. It may not import
`_reference_dp`, `_discrete`, `_minqueue`, `_traceback` or `_geometry`, and may
not reuse their recurrences. It may import `_validation` only for input shape
helpers. Agreement between two implementations of the same recurrence is not
evidence.

## Oracle C - exact minimum enclosing ball (`tests/oracles/enclosing_circle.py`)

Method: for `d = 1`, the interval midpoint. For `d = 2`, enumerate every
1-point, 2-point (diameter) and 3-point (circumcircle) candidate, keep the
candidates that enclose all of S under exact rational comparison, and return
the one of least squared radius. All arithmetic is `fractions.Fraction`, so
both the decision and the centre are exact.

Correctness rests on the standard fact that the minimum enclosing ball of a
finite planar set is determined by a support set of at most 3 points, and is
the smallest enclosing candidate over that finite family.

Cost `O(|S|^4)` rational operations. Oracle use is restricted to `|S| <= 12`.

## Oracle A - deletion (`tests/oracles/subsequence.py`)

For every non-empty subsequence S of Q, compute `discrete_frechet(R, S)` with a
separate, deliberately naive O(mk) dynamic program written directly from the
coupling definition (`tests/oracles/frechet.py`), and keep the S with fewest
deletions satisfying `<= delta`. Return `inf` when no subsequence works.

Enumeration size `2^n - 1`; oracle use is restricted to `n <= 12`.

This tests the deletion DP without duplicating it: it never forms the table,
never uses the three predecessor cases, and decides feasibility only through an
independent ordinary Frechet computation.

## Oracle B - insertion and mixed (`tests/oracles/block_edit_search.py`)

### B.1 Finite candidate lemma (proved here, used by the oracle)

*Claim.* If any feasible edited curve of cost `c` exists, then one of cost `c`
exists in which every inserted point is the exact minimum-enclosing-ball centre
of a contiguous block of R.

*Proof.* Fix a feasible edited curve `Q'` and a coupling. Every element of `Q'`
is coupled to at least one reference index, and by monotonicity the set of
reference indices coupled to one element is a contiguous block `[a,b]`. Let `p`
be an inserted element with block `[a,b]`. Feasibility gives
`max_{a<=t<=b} ||p - R[t]|| <= delta`, so the minimum enclosing ball of
`R[a..b]` has radius `<= delta`, and its centre `c` also satisfies
`max_t ||c - R[t]|| <= delta`. Replacing `p` by `c` changes no other coupled
pair, because `p` is coupled to nothing outside `[a,b]`. Repeating for every
inserted element gives a curve of the same cost using only block centres. QED

Hence `CANDIDATES = { MEB_centre(R[a..b]) : 0 <= a <= b < m, radius <= delta }`,
of size at most `m(m+1)/2`, is complete. Centres are computed by Oracle C.
Candidate coordinates may be reused any number of times and in any gap; the
lemma places no restriction on which block a candidate is later coupled to.

### B.2 Insertion count bound (proved here)

*Claim.* An optimal solution uses at most `m` insertions.

*Proof.* Suppose an inserted element `p` exclusively covers no reference index,
i.e. every index of its block `[a,b]` is also coupled to some other element.
Delete `p` from `Q'` and from the coupling; every reference index remains
covered and the coupling stays monotone, giving a feasible curve of cost
`c - 1`, contradicting optimality. So in an optimal solution every inserted
element exclusively covers at least one of the `m` reference indices, and those
exclusive sets are disjoint. QED

Therefore the oracle searches insertion counts `0..m`, deletion counts `0..n`
(mixed) or `0` (insertion-only), and total cost is bounded by `m + n`.

### B.3 Search

Iterative deepening on total cost `c = 0, 1, 2, ...`:

1. enumerate retained subsequences of Q consistent with the deletion budget
   (all of Q when `operations="insert"`);
2. enumerate ordered placements of the remaining budget of insertions into the
   `n + 1` original gaps, drawing points from `CANDIDATES` with repetition;
3. build the final curve by replay (the same replay semantics as
   `witness-invariants.md`, re-implemented locally);
4. accept the first curve whose independent `discrete_frechet(R, curve)` is
   `<= delta` under exact rational comparison.

Limits: `MAX_STATES = 100_000` curves evaluated and `TIMEOUT_S = 10.0` per
case. Exceeding either raises `OracleBudgetExceeded`. A test that hits the
budget is reported as BLOCKED evidence and is NOT counted as agreement; it must
not be caught and turned into a skip that reads as a pass.

Default exhaustive sizes: `m, n <= 3` in the always-on suite, `m, n <= 4` in
the `slow` marker. Sizes are raised only after measuring the actual state
counts, which the oracle reports.

## What oracle agreement establishes

Cost equality between the production API and Oracles A/B on a case establishes
optimality FOR THAT CASE. Witness feasibility alone never establishes
minimality. A single disagreement is preserved verbatim as a regression fixture
in `tests/unit/` before any fix is attempted.
