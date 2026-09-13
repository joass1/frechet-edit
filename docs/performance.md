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
| frechet-edit | 0.1.0.dev0 |
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
| delete | python | 400 | 2 | 82.0 | 94.3 |
| delete | reference | 400 | 2 | 64.9 | 66.4 |
| insert | python | 400 | 2 | 250.0 | 307.1 |
| insert | reference | 400 | 2 | 260.6 | 309.9 |
| both | python | 400 | 2 | 253.1 | 310.2 |
| both | reference | 400 | 2 | 268.3 | 309.8 |
| both | python | 400 | 10 | 367.2 | 385.3 |
| both | reference | 400 | 10 | 513.9 | 542.9 |
| insert | python | 400 | 10 | 347.3 | 413.0 |
| insert | reference | 400 | 10 | 466.6 | 484.7 |

Honest reading:

* The optimised backend wins where it is supposed to: at `delta = 10` the
  `mu`-window is wide, and replacing the rescan with the monotone queue gives
  about **1.4x** on mixed mode at `m = n = 400` (367 ms vs 514 ms).
* At `delta = 2` the windows are narrow, so a rescan of two or three cells is
  cheaper than the queue's method calls, and the two backends are within noise.
* In **deletion-only** mode the optimised backend is slightly SLOWER than the
  reference (82 ms vs 65 ms). Deletion does not use the `mu` window at all, so
  the optimised path buys nothing there and pays for rolling-list indexing. The
  reference backend is a legitimate choice for delete-only work.
* These are milliseconds on curves of a few hundred points in pure Python. No
  low-latency claim is made, and none should be made from this table.

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
