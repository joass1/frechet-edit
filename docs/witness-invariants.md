# Witness invariants and reconstruction (P0)

Status: APPROVED for P1 and P3.

## 1. What a witness is

A witness is a tuple `(edits, edited_curve, coupling)` such that:

W1. `edits` replayed against the ORIGINAL observation Q reproduces
    `edited_curve` exactly.
W2. `len(edits) == cost`.
W3. `coupling` is a valid strong discrete coupling of R and `edited_curve`:
    it starts at `(0,0)`, ends at `(m-1, k-1)`, and each step advances the
    reference index by 1, the edited index by 1, or both by 1.
W4. every coupled pair satisfies `dist(R[i], edited_curve[t]) <= delta` under
    the certified predicate.
W5. every retained original vertex appears in `edited_curve` unchanged and in
    its original relative order.
W6. the multiset of deleted original indices and the sequence of insertions are
    consistent with `mode` (no insertions in delete-only, no deletions in
    insert-only).

A witness satisfying W1-W6 proves FEASIBILITY at its own cost. It does not by
itself prove MINIMALITY; minimality comes from the DP plus oracle agreement.

## 2. Replay semantics

Replay is a pure function of the ORIGINAL curve and the edit records:

```
kept = [q for idx, q in enumerate(Q) if idx not in deleted]
out  = []
for gap g in 0..n:
    for ins in insertions with gap == g, ordered by ins.order:
        out.append(ins.point)
    if g < n and g not in deleted:
        out.append(Q[g])
```

Consequences that are normative, not incidental:

- indices never shift: a deletion of index 3 does not renumber index 4.
- `gap = n` places points after the last original vertex; `gap = 0` before the
  first.
- the procedure is well defined when `deleted == {0..n-1}`; the output is then
  the inserted points alone, in gap-then-order sequence.
- ordering across gaps is ascending `gap`; within a gap ascending `order`.
  This ordering is preserved even when all intervening original vertices are
  deleted, which is exactly the "delete all and rebuild" solution.

## 3. Reconstruction from the layered DP

Traceback walks the layered states, never the collapsed value table, so the
identity of the last edited element is always known:

| state | recorded parent | emitted |
|---|---|---|
| `X(i,j)` | `F(i, j-1)` | `Deletion(index=j-1)` |
| `K(i,j)` via vertical | `K(i-1, j)` | nothing (sigma_j already kept) |
| `K(i,j)` via diagonal | `F(i-1, j-1)` | keep sigma_j; coupling pair (i,j) |
| `K(i,j)` via horizontal | `F(i, j-1)` | keep sigma_j; coupling pair (i,j) |
| `P(i,j)` with block k | `F(k-1, j)` | one `Insertion` in gap j, centre c |

Invariants maintained during the walk:

T1. Entering `K(i,j)` guarantees sigma_j is retained in the reconstructed
    solution; entering `X(i,j)` guarantees it is deleted. Because the DP stores
    K separately from F, these can never both be asserted for the same j. This
    is the reason the collapsed `F(i-1,j)` predecessor is not used for
    traceback.
T2. Each `P` step emits exactly one insertion and consumes reference block
    `k..i`; its coupling pairs are `(k,t), (k+1,t), ..., (i,t)` for the new
    element index `t`, i.e. a run of vertical steps.
T3. Each `K` chain emits exactly one edited element and a run of coupling pairs
    `(i',j)` for the consecutive i' covered by the vertical chain.
T4. The walk terminates at `F(0,0)`; reaching any `+inf` state during traceback
    is a programming error and raises, it is not silently absorbed.

Because the walk is built backwards, insertion `order` values are assigned
after the walk by numbering each gap's insertions in the order they appear in
the forward (reversed) sequence.

## 4. Tie-breaking

Within a state the branches are evaluated in the fixed order
`K, P, X` and the FIRST strict minimum wins; ties therefore resolve towards
keeping an original vertex, then towards insertion, then towards deletion.
This is deterministic for a fixed backend and input. It is NOT part of the
mathematical contract: another backend may return a different optimal script
of the same cost, and tests must compare costs, not scripts.

## 5. Independent verification

`frechet_edit.verify.verify_witness` re-derives everything from scratch:

- it replays `edits` itself rather than trusting `edited_curve`;
- it recounts edits rather than trusting `cost`;
- it re-checks every coupling step and distance with the certified predicate;
- it recomputes ordinary discrete Frechet on the edited curve with a separate,
  deliberately simple implementation and checks it is `<= delta`;
- it does NOT call the edit DP, the minimum queue, or any stored parent table.

The verifier must REJECT tampering. `tests/unit/test_witness.py` mutates, one
at a time: a deletion index, an inserted coordinate, a gap number, an insertion
order within a gap, a coupling endpoint, a coupling step (making it non-
monotone or a double jump), and the reported cost. Every mutation must produce
a failure report naming the violated invariant.
