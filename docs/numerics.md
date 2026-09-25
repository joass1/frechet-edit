# Numerical policy (P0)

Status: APPROVED for P1 and P3.

Every geometric decision in this package is one of two predicates:

- **PD** (point predicate): `||a - b|| <= delta`, for a, b float64 points.
- **BD** (ball predicate): `radius(MEB(S)) <= delta`, for a finite S of float64
  points, together with a witness centre when the answer is yes.

`delta` is never modified to make a predicate come out a particular way.

## 1. Three-tier decision procedure

Tier 1 - float64 with a rigorous error bound.
  PD is decided on squared quantities. With `u = 2^-53` and dimension d, the
  computed `D2 = fl(sum_i (a_i - b_i)^2)` satisfies
      |D2_exact - D2| <= gamma * D2,  gamma = (d + 3) u / (1 - (d + 3) u)
  (one rounding for each subtraction, one for each square, d-1 for the
  additions, bounded termwise). `delta^2` carries relative error <= u.
  A safety factor of 4 is applied. The decision is accepted when the two
  intervals `[D2(1-4g) - s, D2(1+4g) + s]` and `[delta^2(1-4u) - s,
  delta^2(1+4u) + s]` are disjoint. Otherwise the case is BOUNDARY and falls to
  tier 2.

  The absolute term `s = 4 (d + 4) eta`, with `eta = 2^-1074` the smallest
  subnormal, is **required**, not decorative. The relative bound above holds
  only while nothing underflows; a product whose result is subnormal carries an
  absolute error of up to `eta / 2`, which can be most of the value. Without
  `s`, squares in the subnormal range (coordinates near `1e-162`) produced
  confident wrong decisions - 1555 in a 283849-case probe - and the public API
  returned `cost=0` for infeasible instances. For normal-range values `s` is far
  below one ulp and changes no decision. A square that overflows to `inf` is
  never decided in tier 1. The same test (`_numerics.float_tier`) is used by the
  YES certificate of the ball predicate. Pinned in
  `tests/unit/test_numeric_boundary.py::TestUnderflowDoesNotDefeatTheErrorBound`.

Tier 2 - exact rational arithmetic.
  float64 values are exact binary rationals, so `fractions.Fraction` decides PD
  with no error at all: compare `sum (a_i - b_i)^2` against `delta^2` in Q.
  This is the *certification*, not merely "more precision".

  For BD the same tier is reached through the support-set characterisation: in
  dimension 1 the MEB of S is `[min S, max S]`, centre `(min+max)/2`, radius
  `(max-min)/2`, all exactly rational. In dimension 2 the MEB is determined by
  a support set of 2 points (a diameter) or 3 points (a circumcircle); the
  centre of each candidate is a rational function of the coordinates, so both
  the candidate centre and its squared radius are exactly rational. Enumerating
  all pairs and triples, discarding candidates that do not enclose S under
  exact PD, and taking the minimum squared radius decides BD exactly.

Tier 3 - explicit abstention.
  Exact BD enumeration is `O(|S|^3)` rational operations. It is capped by
  `EXACT_BALL_MAX_POINTS` (default 48) and by a per-call budget. When a
  BOUNDARY ball decision exceeds the cap, the computation does not guess: the
  call returns `status="numerically_ambiguous"`, `cost=None`. That result is
  never rewritten to `inf`, never rewritten to a feasible cost, and never
  silently dropped from an experiment's denominator.

`numeric_policy="certified"` (default) runs all three tiers.
`numeric_policy="fast"` runs tier 1 only and resolves BOUNDARY cases by the
plain float comparison; it is documented as NOT certified and is provided for
benchmarking the cost of certification, never as the default.

## 2. Minimum enclosing ball backends

1D: exact interval midpoint. Deterministic, exact, no randomness.

2D: Welzl's move-to-front randomised incremental algorithm in float64 for the
fast path. Its expected time is linear; no deterministic worst-case linear-time
claim is made. The shuffle is driven by a `random.Random(seed)` instance whose
seed is part of the result metadata, so runs are reproducible. The float result
is used only to (a) produce a candidate centre and (b) decide the non-boundary
cases; boundary cases go to the exact enumeration above, which does not depend
on the random order at all.

An enclosing candidate proves feasibility only after EVERY member of S has been
checked against it. Infeasibility (radius > delta) is asserted only from the
exact minimum over the complete candidate set, never from one bad centre.

## 3. Witness centre representability

A rational centre that certifies `radius <= delta` need not still certify it
after rounding to float64. Therefore every emitted inserted point is
re-validated with exact PD against its whole reference block. If the directly
rounded centre fails, the implementation tries a small deterministic set of
neighbouring float64 candidates (componentwise `nextafter` moves towards the
block's exact centroid, at most 3^d - 1 of them). If none is certified, the
COST is still reported as optimal and `witness_status="unavailable"` with the
reason recorded in `stats`. The threshold is not adjusted, and the failure is
not reported as infeasibility.

## 4. Degeneracy cases that must be tested

Collinear points; cocircular points; duplicated support points; near-zero
triangle area; large common translation; coordinates of very disparate
magnitude; delta immediately below, exactly at, and immediately above a
critical radius; a delta equal to a representable distance exactly.
`tests/unit/test_geometry.py` owns these.

## 5. Status decision table

| tier 1 | tier 2 | tier 3 | reported |
|---|---|---|---|
| decisive | - | - | the decision, `exact_fallbacks` unchanged |
| boundary | decisive | - | the decision, `exact_fallbacks` incremented |
| boundary | over cap | reached | `numerically_ambiguous`, `cost=None` |

A `numerically_ambiguous` outcome in an experiment is an ABSTENTION: it stays
in the denominator and is reported as a coverage failure, never discarded.
