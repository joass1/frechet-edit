# Erratum: the published insertion recurrence is unsound

Status: VERIFIED against both published versions, by three mutually independent
implementations. This document states precisely what is and is not being
claimed. The authors have **not** been contacted; see section 9.

## 1. Summary

The dynamic-programming recurrence published for the **insertion-only**
(`IedDP`) and **insertion-and-deletion** (`edDP`) variants of the strong
discrete Fréchet edit distance does not compute the quantity it is defined to
compute. It can return a value strictly smaller than the true optimum.

The smallest witness is

```
pi = <0, 1, 0>      sigma = <0>      delta = 0.4      insertions only
published recurrence: 1          true optimum: 2
```

Scope of the defect:

| Variant | Section | Recurrence | Verdict |
|---|---|---|---|
| Deletions only (`DedDP`) | 5.1 | no insertion branch | **sound** |
| Insertions only (`IedDP`) | 5.2 | has insertion branch | **unsound** |
| Insertions and deletions (`edDP`) | 5.3 | has insertion branch | **unsound** |

**What is not claimed.** Theorems 10/20 and 11/21 assert that `IedDF` and `edDF`
are computable in `O(m^2 + mn)` time. Those statements are **not** refuted. The
corrected recurrence in section 6 below runs in the same asymptotic bound, so
the results stand; what fails is the recurrence offered as the proof. This is an
erratum in the presentation of the algorithm, not a counterexample to the
theorems.

## 2. Source

Emily Fox, Amir Nayyeri, Jonathan James Perry, Benjamin Raichel, *Fréchet Edit
Distance*, SoCG 2024, LIPIcs vol. 293, 58:1-58:15,
DOI 10.4230/LIPIcs.SoCG.2024.58; full version arXiv:2403.12878.

Both versions were checked. The conference version of record (section 5.2) and
the arXiv full version (section 5.2) display the **same** recurrence with the
same vertical predecessor term. arXiv lists only `v1`; there is no revision that
alters it.

## 3. What the paper states

Quoting the full version, section 5.2, with `IedDP(i,j) := IedDF(pi[1,i], sigma[1,j])`:

> As before, assume there is a set of insertions changing `sigma[1,j]` to `sigma'`
> where `d_DF(pi[1,i], sigma') <= delta`. Suppose `sigma'` ends with `sigma_j`,
> implying `||sigma_j - pi_i|| <= delta`. **(It is important to note for later that
> if `||sigma_j - pi_i|| <= delta` it does not imply `sigma'` ends with `sigma_j`.)**
> We get the three standard cases for computing the discrete Fréchet distance as
> before.

and then, in the `otherwise` branch (the branch taken when `||sigma_j - pi_i|| <= delta`):

```
IedDP(i,j) = min { IedDP(i, j-1),
                   IedDP(i-1, j),
                   IedDP(i-1, j-1),
                   1 + min_{mu(i) <= k <= i} IedDP(k-1, j) }
```

The parenthetical remark identifies exactly the hazard that makes this wrong.
The three "standard cases" are derived **under the hypothesis that `sigma'` ends
with `sigma_j`**, but the branch guard that admits them is `||sigma_j - pi_i|| <= delta`,
which the authors correctly note is a weaker condition. The remark is flagged
"for later" and no later passage constrains the branch: the text proceeds to
Lemma 18, Lemma 19, and Theorem 20 on efficiency only, and section 5.3 repeats
the same structure for `edDP`.

## 4. Why it fails

`IedDP(i-1, j)` is the unrestricted optimum for the prefix pair
`(pi[1,i-1], sigma[1,j])`. An optimal edited curve realising that value may end
with a **newly inserted** point rather than with `sigma_j`.

Using such a value as the vertical predecessor silently asserts that `sigma_j`
is still available to be coupled with `pi_i`. It is not. In the discrete Fréchet
correspondence the coupling is monotone, so once the edited curve has advanced
past `sigma_j` to an inserted point, no later step may go back and couple
`sigma_j` to `pi_i`. The recurrence permits a step the coupling forbids, admits
solutions that do not exist, and therefore under-reports.

The defect is confined to the insertion branch. In the deletion-only recurrence
of section 5.1 there is no insertion branch at all, so every element of the
edited curve is an original `sigma` point and the predecessor can never "end with
an inserted point". That is why `DedDP` is sound, and it was verified to be
sound (section 7).

## 5. The counterexample, worked

`pi = <0, 1, 0>`, `sigma = <0>`, `delta = 0.4`, insertions only.

First, the true optimum is 2. The points `0` and `1` are `1` apart, and
`delta = 0.4`, so no single point of the edited curve can be within `delta` of
both: a ball covering both needs radius `0.5 > 0.4`. Covering `pi = <0, 1, 0>`
therefore needs three distinct edited-curve points, in order. `sigma` supplies
one. So **2 insertions** are necessary, and 2 suffice: `sigma' = <0, 1, 0>`.

Now the published recurrence. `mu(1) = 1`, `mu(2) = 2`, `mu(3) = 3`, since no
two consecutive points of `pi` fit in a ball of radius `0.4`.

```
IedDP(0,0) = 0                                                  base
IedDP(0,1) = inf                                                base
IedDP(1,0) = 1                                                  insertion branch
IedDP(1,1) = 0     otherwise: min{1, inf, 0, inf}
IedDP(2,0) = 2                                                  insertion branch
IedDP(2,1) = 1     ||sigma_1 - pi_2|| = 1 > 0.4, insertion branch
IedDP(3,0) = 3                                                  insertion branch
IedDP(3,1) = 1     otherwise: min{3, IedDP(2,1)=1, 2, 2}   <-- takes IedDP(i-1,j)
```

Result `1`; truth `2`.

The offending value is `IedDP(2,1) = 1`. It is realised only by
`sigma' = <0, x>` with `x ≈ 1` an inserted point covering `pi_2`. The final line
then couples `pi_3` to `sigma_1 = 0` — but `sigma_1` was already consumed by
`pi_1`, and the curve has since advanced to `x`. Monotonicity forbids returning
to it. The recurrence does not notice, because a single table indexed by
`(i, j)` cannot record whether the optimal prefix solution ended with `sigma_j`
or with an inserted point.

## 6. The correction

Split the state by the identity of the last element of the edited prefix. This
package uses three layers:

* `K(i,j)` - optimal solutions whose edited prefix ends with the kept point `sigma_j`
* `P(i,j)` - optimal solutions whose edited prefix ends with an inserted point
* `X(i,j)` - the unrestricted minimum, `min(K, P)`

The keep branches then draw their vertical predecessor from `K` only, never from
`X`. The insertion branch continues to draw from `X`, which is correct: an
inserted point may follow anything. Full statement and correctness argument in
`docs/recurrences.md` section 2; the unsoundness of the collapsed form is
section 3.2.

This changes the constant factor, not the asymptotics: `O(m^2)` to precompute
all `mu(i)` (Lemma 18, unchanged) plus `O(mn)` for the layered DP, i.e. the
`O(m^2 + mn)` of Theorems 20 and 21. The published complexity results are
therefore unaffected.

### 6.1 Exactly one term is at fault

It is worth being precise about how small the repair is, because it pins the
cause rather than merely removing the symptom. Writing the keep branch as

```
K(i,j) = min( X(i, j-1),  K(i-1, j),  X(i-1, j-1) )     when ||sigma_j - pi_i|| <= delta
P(i,j) = 1 + min_{mu(i) <= k <= i} X(k-1, j)
X(i,j) = min( K(i,j), P(i,j) )
```

only the **middle term of `K`** differs from the published form. The other two
keep terms legitimately draw on the unrestricted `X`:

* `X(i, j-1)` appends `sigma_j` after whatever the predecessor ended with,
  which is always allowed, since `sigma_j` follows it in the original order;
* `X(i-1, j-1)` does the same while also advancing `pi`;
* `K(i-1, j)` is different in kind. It couples `pi_i` to a `sigma_j` that is
  **already** the last element of the predecessor's edited curve, so the
  predecessor must itself end with `sigma_j`. A predecessor ending in an
  inserted point cannot supply that, and `X(i-1, j)` includes exactly those.

Changing that one term and nothing else makes the recurrence agree with the
definition on all 4320 instances of the insertion sweep, with zero failures.
The insertion branch, the `mu(i)` construction of Lemma 18 and the queue of
Lemma 19 are all untouched and all correct.

## 7. Verification

Three implementations that share no recurrence were compared:

1. **The published recurrence**, transcribed verbatim from the PDF text into
   `tests/property/test_published_recurrence.py`, written without reference to
   this package's solver.
2. **A definitional brute force**, which contains no dynamic program at all: it
   enumerates edited curves and monotone staircase couplings directly, placing
   each inserted point at the centre of the minimum enclosing ball of the block
   it covers, with exact rational arithmetic for the radius test. It imports
   nothing but `itertools` and `fractions`; its geometry is written from
   scratch in that file rather than shared with the transcription above, so
   "shares no recurrence" is literal rather than approximate.
3. **This package** (`frechet_edit.discrete_edit_distance`).

Results:

| Sweep | Comparisons | Published recurrence | This package |
|---|---|---|---|
| Deletion, exhaustive 1-D, `delta` in {0.4, 1.0, 2.0} | 14040 | 0 wrong | 0 wrong |
| Insertion + mixed, exhaustive 1-D, `delta` in {0.4, 1.0, 2.0} | 8640 | **292 wrong** | 0 wrong |
| Insertion + mixed, random 2-D | 800 | **2 wrong** | 0 wrong |

Every discrepancy is an **under**-estimate (292 under, 0 over), which is what
the diagnosis predicts: the `min` ranges over a strictly too permissive set of
predecessors. The largest gap in that sweep is 1. Failure density rises as
`delta` falls relative to the spacing of `pi`: at `delta = 2.0` on the
`{0, 1, 2.5}` alphabet no failure occurs at all, because `mu(i)` collapses and
the insertion branch stops competing.

The package agreed with the definitional brute force on all 23480 comparisons.

### 7.1 The error is not bounded by 1

"Largest gap 1" is true of the small exhaustive sweep above and false in
general, so it should not be read as a characterisation of the defect. On
alternating curves the gap grows without bound. Take
`pi = <0, 1, 0, 1, ...>` of length `m`, `sigma = <0>`, `delta = 0.4`, insertions
only:

| `m` | published | true optimum | gap |
|---|---|---|---|
| 3 | 1 | 2 | 1 |
| 4 | 2 | 3 | 1 |
| 5 | 2 | 4 | 2 |
| 6 | 3 | 5 | 2 |
| 7 | 3 | 6 | 3 |
| 8 | 4 | 7 | 3 |
| 9 | 4 | 8 | 4 |

The true optimum is `m - 1`: consecutive vertices are 1 apart and `2 * delta`
is 0.8, so no point can cover two of them, every vertex needs its own
edited-curve point, and `sigma` supplies one. The published form returns about
`(m - 1) / 2`, because it re-uses the single kept vertex once per alternation
instead of once in total. The ratio therefore approaches **2**, and the
absolute error grows linearly in `m`.

This matters for how the defect is described. It is not an off-by-one at a
numerical boundary; it is a structural error whose size scales with how often
the optimal solution would have to return to a vertex it has already passed.

Reproduced in
`tests/property/test_published_recurrence.py::TestTheErrorIsNotBoundedByOne`.

## 8. Reproducing

```
pytest tests/property/test_published_recurrence.py -v
```

## 9. Status of this finding

* Verified against the conference version of record and the arXiv full version.
* The first author was contacted about this finding in September 2026.
* No claim of priority is made. No search was performed for existing errata, so
  the possibility that this is already known is open, and a negative search
  would not establish novelty in any case.
* The correct reading may be that the recurrence is shorthand for a state that
  the authors intended to carry the "ends with `sigma_j`" restriction, which
  their parenthetical shows they had in mind. That would make this a defect of
  presentation rather than of the underlying algorithm. Either way the displayed
  recurrence cannot be implemented as written, and the restriction cannot be
  expressed without changing the state space.
