# Measured performance (P4)

Everything below is a MEASUREMENT on one machine, not a guarantee and not a
proof of an asymptotic bound. The structural argument is in
`docs/recurrences.md` section 6; this file is only evidence that the
implementation behaves the way that argument says it should.

Reproduce with:

```
python benchmarks/bench_discrete.py --sizes 100 200 400 --repeats 5
python benchmarks/bench_memory.py
```

## Environment

| key | value |
|---|---|
| frechet-edit | 0.1.0 |
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| OS | Windows 11 (10.0.26200) |
| state | warm: same process, repeated calls, no JIT |
| seed | 20240612 |

Curves are persistent random walks of equal length (`m = n`) in 2-D, with 5%
of the observation's vertices displaced by Gaussian noise of scale 25.

## Time

p50 / p95 wall-clock milliseconds per call, 5 repeats.

| mode | backend | m = n | delta | p50 ms | p95 ms |
|---|---|---|---|---|---|
| delete | python | 400 | 2 | 42.8 | 46.5 |
| delete | reference | 400 | 2 | 72.2 | 74.3 |
| insert | python | 400 | 2 | 218.5 | 251.5 |
| insert | reference | 400 | 2 | 230.6 | 234.5 |
| both | python | 400 | 2 | 224.0 | 356.3 |
| both | reference | 400 | 2 | 306.6 | 378.9 |
| both | python | 400 | 10 | 252.7 | 253.2 |
| both | reference | 400 | 10 | 475.8 | 550.5 |
| insert | python | 400 | 10 | 264.0 | 300.4 |
| insert | reference | 400 | 10 | 374.3 | 386.0 |

Honest reading:

* The optimised backend wins where it is supposed to: at `delta = 10` the
  `mu`-window is wide, and replacing the rescan with the monotone queue gives
  about **1.9x** on mixed mode at `m = n = 400` (253 ms vs 476 ms).
* At `delta = 2` the windows are narrow, so a rescan of two or three cells is
  cheaper than the queue's method calls, and the two backends are within noise.
* In **deletion-only** mode the optimised backend is now FASTER than the
  reference (43 ms vs 72 ms). An earlier revision of this document reported the
  opposite, 82 ms against 65 ms, and that reversal is a real change rather than
  noise: the support-set generator these numbers share was rewritten (see
  below), and the reference backend leans on it harder.
* These are milliseconds on curves of a few hundred points in pure Python. No
  low-latency claim is made, and none should be made from this table.

## The support-set generator, and a failed optimisation

Profiling a mixed-mode solve at `m = n = 400` put the largest single cost not in
the dynamic program but in `_geometry._welzl_support_2d`, the float64 pass that
proposes a candidate support set for each block. It accounted for roughly **40
per cent of total runtime** across about **1.46 million** containment checks -
more than the recurrence it exists to serve.

The first fix attempted was the obvious one: vectorise each scan so numpy finds
the first violating point in one call. It was **measured and rejected**. It ran
roughly **twice as slow** for blocks of ten points or fewer, which is nearly all
of them, because `mu_indices` calls this on short windows and numpy's per-call
overhead swamps the work. It only began to win past about a hundred points, a
block size that barely occurs.

The fix that worked went the other way: remove numpy from the function entirely
and do plain Python float arithmetic on unpacked coordinates. In two dimensions
each containment test is a handful of multiplications with no array allocation,
no dtype dispatch and no ufunc call.

Measured on the function alone, identical support sets on 3000 random blocks:

| block size | before | after | speedup |
|---|---|---|---|
| 2 | 13.5 us | 7.9 us | 1.7x |
| 3 | 33.7 us | 9.3 us | 3.6x |
| 5 | 35.2 us | 10.3 us | 3.4x |
| 10 | 198.6 us | 35.3 us | 5.6x |
| 40 | 632.3 us | 66.6 us | 9.5x |
| 200 | 3236.3 us | 292.7 us | 11.1x |

End to end the gain depends entirely on how large the blocks are, which depends
on `delta` relative to the curve's scale. On the frozen benchmark above, mixed
mode at `m = n = 400`, `delta = 10` went from 367 ms to 253 ms. On random-walk
curves with wider windows, a controlled A/B of the same solve measured **3.5x**
(1176 ms to 335 ms). Both numbers are real; neither generalises, and the honest
summary is a range, not a headline.

**No native extension was written.** The original plan for this phase was a C++
backend, and profiling is the reason it was not: the bottleneck was per-call
overhead on tiny inputs, which a compiled extension addresses by adding a
foreign-call boundary in exactly the wrong place. S1 remains open, and should
stay open until a profile justifies it.

## Instrumentation supporting the complexity argument

Counters are returned in `EditResult.stats`.

| m = n | enclosing-ball calls | ratio to m | exact fallbacks |
|---|---|---|---|
| 100 | 177-194 | 1.8-1.9 | 0 |
| 200 | 378-394 | 1.9-2.0 | 0 |
| 400 | 778-794 | 1.9-2.0 | 0 |

The call count tracks `2m`, which is what the two-pointer argument predicts:
each iteration advances either `i` or `mu`, and neither moves backwards, so the
total is at most `2m`. Zero exact fallbacks on this data means no geometric
predicate landed near the `delta` boundary; the exact tier is reserved for
cases that do, and `tests/unit/test_geometry.py` exercises it directly.

## Memory

Peak allocation via `tracemalloc`, `m = 120`, `delta = 4`, mode `both`.

| n | score-only KiB | witness KiB | ratio |
|---|---|---|---|
| 50 | 64.9 | 483.9 | 7.5x |
| 100 | 37.8 | 821.6 | 21.8x |
| 200 | 14.0 | 1583.3 | 113.5x |
| 400 | 14.0 | 3113.2 | 222.2x |
| 800 | 14.0 | 6176.1 | 440.8x |

* **Score-only peak is flat at 14.0 KiB while `n` grows 4x** (200 to 800).
  That is the `O(m)` rolling-column design behaving as designed: no dense
  `m x n` distance matrix is ever built, and each column materialises one
  boolean row of length `m`.
* The `n = 50` and `n = 100` rows are larger because they are dominated by
  one-time allocations (import, NumPy warm-up), not by the DP. They are
  excluded from the trend rather than quietly reported as an improvement.
* **Witness storage grows linearly in `n`**, as documented: requesting a
  witness switches to the reference backend and its `O(mn)` parent tables.
  That is a deliberate trade, not a regression - a score-only query and a
  witness query on the same input return the same cost, which
  `tests/property/test_invariants.py` checks.

## What is not measured here

* Cold-start cost, import time, and first-call warm-up.
* Anything on real trajectory data; there is none in this repository.
* Any native or JIT backend. None exists; stretch goal S1 is not started.
* Throughput on curves of thousands of points. The numbers above stop at 400
  and must not be extrapolated.

## Continuous deletion (`continuous_edit_distance`)

`python benchmarks/bench_continuous.py`: a sparse reference route of `m`
vertices against a dense noisy observation of the same path (`n` fixes, 3 m
noise) with injected spikes, `delta = 25`, witness included. One run per row
on a laptop that was also running other jobs, so read the times as an order of
magnitude.

| m | n | spikes | discrete deletion | continuous deletion | time | budgets tried |
|---|---|---|---|---|---|---|
| 20 | 100 | 2 | infeasible | 2 | 0.11 s | 0, 1, 2 |
| 30 | 200 | 3 | 148 | 3 | 0.21 s | 0, 1, 2, 4 |
| 40 | 300 | 5 | 197 | 4 | 0.29 s | 0, 1, 2, 4 |
| 60 | 400 | 8 | 184 | 8 | 0.98 s | 0, 1, 2, 4, 8 |
| 60 | 600 | 10 | 290 | 10 | 2.29 s | 0, 1, 2, 4, 8, 16 |

The discrete column is not a performance comparison: it shows that discrete
Fréchet, matching vertices to vertices, cannot align a sparse reference with
dense samples at all. In the 5-spike row one injected offset was only 28.9 m,
and its component away from the route is under 25 m, so it is not a glitch at
this `delta`; the four deleted indices are exactly the other four injected
ones.

Cost grows as `O(k^2 m n)` in the budget `k` (Theorem 3), and the budget
search tries `0, 1, 2, 4, ...`, so a result costs roughly one run at a budget
below twice the optimum. Where an answer is infeasible within the cap, the
search runs to the cap: the web app's worst case, 150 course vertices against
600 fixes with a cap of 25, measured 3.8 s. Exact arithmetic is used only
where float enclosures overlap; repeated points are ranked once, which cut the
two-lap web scenario from 3.2 s to 1.5 s.
