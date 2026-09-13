# Recurrences, base cases and their justification (P0, amended during P3)

Status: APPROVED for P1 and P3.
Amendment 1 (P3): section 3 previously claimed the collapsed and layered forms
are equivalent in all modes. That claim was FALSE for insertion and mixed
modes and was refuted by an exhaustive test plus an independent oracle. The
counterexample and the corrected statement are in section 3.

Notation. R = pi_1..pi_m (fixed), Q = sigma_1..sigma_n (editable), both
ONE-BASED in this document. `close(i,j)` abbreviates `dist(pi_i, sigma_j) <= delta`
under the predicate policy of `numerics.md`.

## 0. The state

For a prefix pair (i, j), 0 <= i <= m, 0 <= j <= n, define

    F(i,j) = min edits applied inside sigma_1..sigma_j such that the resulting
             edited prefix E satisfies discrete_frechet(pi_1..pi_i, E) <= delta,
             with F(0,0) = 0 and E required to be empty exactly when i = 0.

F(0,j) therefore forces E empty, i.e. all j original vertices deleted.
F(i,0) for i > 0 forces E to consist of inserted points only.
The answer is F(m,n).

This state is sufficient because a discrete Frechet coupling is monotone: the
cost of completing a traversal depends only on how far each curve has been
consumed, not on how the consumed prefix was edited.

`mu(i)` is the smallest t in 1..i such that the minimum enclosing ball of
pi_t..pi_i has radius <= delta. A single inserted point p can be coupled to a
contiguous block pi_k..pi_i iff max_t ||p - pi_t|| <= delta, and such a p exists
iff that block's minimum enclosing ball has radius <= delta. So the feasible
blocks ending at i are exactly k in [mu(i), i]. `mu` is non-decreasing because
enclosing radii grow with i for fixed t.

## 1. The three moves

At state (i, j) there are three ways the edited prefix can be extended:

    (D)  delete sigma_j                              cost 1,  j >= 1
    (M)  keep sigma_j and couple it to pi_i          cost 0,  needs close(i,j)
    (I)  append one inserted point covering pi_k..pi_i, mu(i) <= k <= i
                                                     cost 1,  i >= 1

Mode restriction: deletion-only drops (I); insertion-only drops (D); "both"
keeps all three. Deletion-only additionally has F(i,0) = +infinity for i > 0.

The subtlety is the PREDECESSOR of (M). Because the last coupled pair is
forced to be (pi_i, sigma_j), the three legal incoming coupling steps are

    vertical   (pi_{i-1}, sigma_j) -> (pi_i, sigma_j)
    diagonal   (pi_{i-1}, prev)    -> (pi_i, sigma_j)
    horizontal (pi_i,     prev)    -> (pi_i, sigma_j)

and the vertical one requires the predecessor state to END WITH sigma_j. A
plain `F(i-1, j)` does not express that requirement; section 3 shows that using
it is not merely loose but actually WRONG once insertions are allowed.

## 2. Layered form (implemented)

Split F by what the edited prefix ends with:

    K(i,j) = F restricted to solutions whose edited prefix ENDS WITH the
             retained original vertex sigma_j
    P(i,j) = F restricted to solutions whose edited prefix ENDS WITH an
             inserted point (which lies in the gap after sigma_j)
    X(i,j) = F restricted to solutions in which sigma_j is DELETED

    F(i,j) = min( K(i,j), P(i,j), X(i,j) )

    X(i,j) = 1 + F(i, j-1)                                         j >= 1
    K(i,j) = +inf if not close(i,j), else
             min( K(i-1,j), F(i-1,j-1), F(i,j-1) )                 i,j >= 1
    P(i,j) = 1 + min_{ mu(i) <= k <= i } F(k-1, j)                 i >= 1

    F(0,0) = 0;  K(0,*) = K(*,0) = +inf;  P(0,*) = +inf;  X(*,0) = +inf
    hence F(0,j) = X(0,j) = j and F(i,0) = P(i,0).

`K(i-1,j)` is the exact vertical predecessor: it is by definition the best
solution that both consumes pi_1..pi_{i-1} and ends at sigma_j.

**Soundness.** Each branch exhibits a concrete feasible edited prefix, so
F(i,j) is at most the stated minimum. **Completeness.** Take an optimal solution
for F(i,j) and inspect the last element of its edited prefix. It is sigma_j
retained (-> K, whose three predecessors enumerate the three possible incoming
coupling steps, the vertical one landing in K by construction), or an inserted
point p (-> P: by monotonicity p is coupled to a contiguous suffix block
pi_k..pi_i, so that block's enclosing radius is <= delta, hence k >= mu(i), and
the state before p is (k-1, j)), or sigma_j is deleted (-> X). The enumeration
is exhaustive, so F(i,j) is at least the stated minimum.

## 3. Relation to the collapsed single-table form

Write the collapsed recurrence as

    Fc(i,j) = min( 1 + Fc(i,j-1),
                   min(Fc(i-1,j-1), Fc(i-1,j), Fc(i,j-1)) if close(i,j),
                   1 + min_{mu(i) <= k <= i} Fc(k-1,j) )

i.e. the layered form with the vertical predecessor `K(i-1,j)` replaced by the
unrestricted `F(i-1,j)`.

### 3.1 Deletion-only: the two forms agree

With no insertions, a solution counted by F(i-1,j) either ends with sigma_j
retained - in which case appending the vertical step is legal and costs nothing
- or has sigma_j deleted. In the second case, un-deleting sigma_j and coupling
it to pi_i saves one deletion, and the resulting solution is exactly one
counted by F(i-1,j-1), so F(i-1,j-1) <= F(i-1,j) - 1 and the diagonal term
already dominates. Hence substituting F(i-1,j) for K(i-1,j) never lowers the
minimum below the true optimum, and it obviously never raises it.

`tests/unit/test_equivalence.py::test_deletion_mode_forms_agree` checks this
exhaustively on all 1-D instances with m <= 3, n <= 2 over three thresholds,
and on random 2-D instances.

### 3.2 Insertion and mixed: the collapsed form is UNSOUND

Once insertions exist, a solution counted by F(i-1,j) may end with an INSERTED
point that sits after sigma_j. Such a solution can be extended neither by a
vertical step at sigma_j (sigma_j is no longer last) nor for free in any other
way, yet the collapsed recurrence credits it as if it could.

Minimal counterexample, found by exhaustive search and confirmed against the
independent oracle of `docs/oracle-protocol.md`:

    R = [0, 1, 0],  Q = [0],  delta = 0.4,  operations = "insert"

    layered form  : 2     <- correct
    collapsed form: 1     <- wrong, too cheap
    Oracle B      : 2

Why. `close(3,1)` holds, because pi_3 = 0 and sigma_1 = 0 coincide, so the
collapsed keep branch offers `Fc(3,1) <= Fc(2,1) = 1`. But the solution
achieving F(2,1) = 1 is "keep sigma_1 at 0, then insert a point at 1 to cover
pi_2"; its edited prefix ends with that inserted point. Coupling pi_3 to
sigma_1 afterwards would require going BACKWARDS past the inserted point, which
no monotone coupling allows. The true answer needs a second insertion, because
one inserted point cannot cover both pi_2 = 1 and pi_3 = 0 when delta = 0.4
(their enclosing radius is 0.5).

Exhaustive search over the same family finds 30 such disagreements, and in
every one the oracle sides with the layered form.

**Consequence.** The layered split is not a presentational convenience. It is
REQUIRED for correctness in insertion and mixed modes, and the implementation
uses it in both `_reference_dp` and `_discrete`.

**Source status.** This document's collapsed form is this project's own
shorthand, written while translating the published description. The
counterexample refutes THAT transcription. It is not evidence about what the
authors wrote: the published proofs may well carry the "ends with sigma_j"
restriction in prose or in the definition of the DP state. This is recorded as
an open verification item in `docs/paper-map.md` rather than as a claimed
erratum, and the resolution is the same either way, since the layered form is
proved correct in section 2 and agrees with an independent oracle.

## 4. Domination note

When close(i,j) holds, (M) yields at most F(i,j-1), while (D) yields
1 + F(i,j-1). Deleting a delta-close vertex is therefore never strictly
necessary, but (D) is still evaluated unconditionally: it costs nothing, keeps
the code uniform, and makes the "delete everything and rebuild" solution
explicit.

## 5. What this recurrence is NOT

It is not LCS, Levenshtein, EDR or DTW. Those charge a cost per unmatched
element and advance both sequences in lockstep on a match. Here a single
retained vertex may be coupled to an unbounded run of reference vertices (the
K(i-1,j) chain) and a single INSERTED vertex may cover an unbounded contiguous
reference block (the mu-window). Replacing (I) by "insert a copy of a reference
vertex" gives a different, strictly larger objective; the separating fixture is
R = [0, 2, 4], Q = [0], delta = 1, where one inserted point at 3 covers both
pi_2 and pi_3 for a cost of 1 while reference-vertex insertion needs 2. It is
tested in `tests/unit/test_insertion.py` and guarded in
`tests/unit/test_equivalence.py::test_the_collapsed_model_is_not_vacuous`.

## 6. Complexity

- Deletion-only: O(mn) time, O(m) rolling score storage.
- Insertion / mixed: O(m^2) for all mu(i) (O(m) enclosing-ball calls by the
  two-pointer argument, each linear in the block length) plus O(mn) for the DP,
  provided the mu-window minimum is maintained by a monotone deque rather than
  rescanned. Column-major (j outer, i inner) evaluation is required because
  P(i,j) reads F(k-1,j) from the SAME column, and K(i,j) reads K(i-1,j) from it.
- Score-only working storage is O(m); witness storage is O(mn) parent records.

These are algorithmic targets derived from the structure, not measurements.
Measured numbers live in `docs/performance.md`.
