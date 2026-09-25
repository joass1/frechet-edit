# Public definition and contract (P0)

Status: APPROVED for P1 (deletion) and P3 (insertion/mixed) scope.
This document is the specification. Code must follow it; where code and this
document disagree, this document wins until it is amended under review.

## 1. Objects

- `reference` R = (R_0, ..., R_{m-1}), m >= 1, the FIXED curve.
- `observation` Q = (Q_0, ..., Q_{n-1}), n >= 1, the EDITABLE curve.
- Both are sequences of points in R^d, same d, finite float64 coordinates.
- `delta` is a finite, strictly positive Euclidean threshold in coordinate units.

Indices in this document and in the public API are ZERO-BASED. The recurrence
document uses ONE-BASED prefix lengths; the mapping is stated there.

## 2. Discrete Frechet (the residual geometric constraint)

A *coupling* of A (length a >= 1) and B (length b >= 1) is a sequence of index
pairs starting at (0,0), ending at (a-1,b-1), where each successive pair
advances the first index by 1, the second by 1, or both by 1 (the "strong"
/ standard discrete traversal). Its cost is the maximum Euclidean distance over
its pairs. `discrete_frechet(A,B)` is the minimum cost over couplings.

`discrete_frechet` of an empty curve against a non-empty curve is undefined and
is treated as INFEASIBLE. Empty vs empty is 0 and occurs only as an internal
DP state, never as public input.

## 3. The objective

    FED_delta(R, Q) = min { edit_count(Q -> Q') : discrete_frechet(R, Q') <= delta }

- Edits apply to Q ONLY. R is never modified.
- One vertex deletion costs 1. One vertex insertion costs 1.
- An inserted vertex may be placed at ANY point of the supported Euclidean
  space. It is NOT restricted to reference vertices or to observation vertices.
- Insertions and deletions are allowed at any position, including before the
  first and after the last original vertex (prefix and suffix editing).
- `operations` selects the allowed edit set: "delete", "insert", or "both".

The returned cost is an INTEGER COUNT OF EDITS. It is not a distance in
coordinate units and must never be compared numerically against meters or
against an ordinary Frechet distance.

## 4. Directedness and non-metric status

`FED_delta(R, Q)` is directed: swapping the arguments changes the problem.
No symmetry, no identity of indiscernibles, and no triangle inequality are
claimed or tested. The public API takes `reference` and `observation` as
keyword-friendly named parameters so the direction cannot be transposed by
accident.

## 5. Supported dimensions

- `operations="delete"`: any d >= 1 (deletion needs only point distances).
- `operations="insert"` and `"both"`: 1 <= d <= 8. The cap is the cost of the
  exact minimum-enclosing-ball kernel, whose per-round work grows like
  `2**(d+2)` candidate subsets each solving a `d x d` rational system. This is a
  PRODUCT SCOPE restriction, not a limitation of the underlying theorem, which
  holds in any fixed dimension.
- An unsupported dimension raises `UnsupportedDimensionError`. It is never
  silently downgraded, projected, or reported as infeasible.

Coordinates are Euclidean. Latitude/longitude MUST be projected to a local
metric CRS by the caller. The core never applies haversine distance and never
treats degrees as meters.

## 6. Edge behavior (normative)

- Empty public `reference` or `observation` -> `ValueError`. Internal
  empty-prefix DP states are legal and are required by the recurrences; in
  particular "delete every original Q vertex, then insert" is a valid solution
  of the mixed mode and must be reachable.
- Singletons and repeated/duplicate vertices are supported and are never
  deduplicated, sorted, reversed, or otherwise mutated. Multiplicity matters:
  costs count vertices.
- `delta <= 0`, NaN or infinite `delta` -> `ValueError`. Zero-threshold support
  is not part of this release.
- NaN or infinite coordinates, ragged arrays, mismatched dimensions, unknown
  `operations` / `backend` / `numeric_policy` -> `ValueError`.
- Deletion-only and insertion-only can be mathematically INFEASIBLE. That is
  reported as `status="infeasible"`, `cost=inf`; it is not an exception.
- Mixed mode on valid finite non-empty inputs is always feasible with
  cost <= m + n (delete all of Q, then insert a cover of R).
- Comparisons are CLOSED: a pair is acceptable when distance <= delta. No
  hidden epsilon ever enlarges or shrinks delta. Numerically undecidable
  comparisons are resolved by the policy in `numerics.md`, never by nudging
  delta.

## 7. Result contract

`EditResult` fields:

| field | meaning |
|---|---|
| `status` | `"optimal"` / `"infeasible"` / `"numerically_ambiguous"` |
| `cost` | `int` when optimal; `math.inf` when infeasible; `None` when ambiguous |
| `witness_status` | `"not_requested"` / `"certified"` / `"unavailable"` |
| `edited_curve` | `np.ndarray` (k,d) when a certified witness exists, else `None` |
| `edits` | tuple of `Deletion` / `Insertion` records, else `None` |
| `coupling` | tuple of `(i, t)` pairs over R and the edited curve, else `None` |
| `mode`, `delta`, `dimension`, `backend`, `numeric_policy` | echoed inputs |
| `stats` | counters (states, enclosing-ball calls, exact fallbacks, ...) |

`status` and `witness_status` are independent. A cost may be certified optimal
while no representable certified witness can be emitted (see `numerics.md`);
that is `status="optimal"`, `witness_status="unavailable"`, and it must NOT be
reported as mathematical infeasibility.

`numerically_ambiguous` means the numerical policy was exhausted. It is never
converted into `inf`, into a feasible answer, or into an exception by a backend.

JSON serialization (`EditResult.to_json_obj()`) never emits a bare `Infinity`
token: an infeasible cost serializes as `null` alongside `status`.

## 8. Witness conventions

- A `Deletion` names the ORIGINAL zero-based index of the removed Q vertex.
- An `Insertion` names a `gap` g in `0..n`, meaning "immediately before original
  index g" (`g = n` means after the last original vertex), an `order` integer
  giving the stable position within that gap, and float64 `point` coordinates.
- Replay is defined against the ORIGINAL Q and original indices. It must work
  when every original vertex is deleted.
- Gaps are emitted in ascending `gap`, then ascending `order`. That order is
  retained even when all intervening original vertices are deleted.
- Retained original vertices appear unchanged and in their original relative
  order in `edited_curve`.
- `coupling` indexes R and the FINAL edited curve, starts at (0,0), ends at
  (m-1, k-1), uses only the three legal monotone steps, and every pair is
  within delta. Costs count EDITS, not coupling moves.
- Tie-breaking is deterministic for a fixed backend and input, and documented,
  but different backends may legitimately return different optimal scripts of
  equal cost. Only the cost is canonical.

## 9. Out of scope for this release

Weighted edit costs, substitutions, simultaneous editing of both curves,
symmetrization, temporal penalties, endpoint locking, continuous (polygonal)
insertion and mixed variants, weak traversals, online/streaming input, and map
matching. The continuous DELETION variant is implemented under its own
contract, `docs/continuous.md`; nothing in this document applies to it unless
that one says so. Any
application-level composite score must be given its own name and must not be
called FED.
