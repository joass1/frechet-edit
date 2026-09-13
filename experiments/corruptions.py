"""Seeded, reproducible corruption operators for curve experiments.

EVIDENCE LEVEL
--------------
Everything in this module operates on SYNTHETIC curves. Nothing here has ever
seen a real GPS trace. Results derived from it are EVIDENCE LEVEL A (synthetic
mechanism tests) in the taxonomy of the project plan, section 4.2. They are NOT
level B (real-trace-derived controlled corruption) and NOT level C (natural
repeated-route evaluation).

CONVENTIONS
-----------
* A *curve* is a ``(k, d)`` array of finite float64 coordinates, ``k >= 1``.
* Every stochastic operator takes an explicit ``numpy.random.Generator``. Given
  the same generator state and the same arguments the output is bit-identical.
  No operator reads the global numpy random state.
* Inputs are never mutated. A fresh float64 array is returned every time.
* Every operator returns ``(corrupted_curve, CorruptionRecord)``. The record
  names the ORIGINAL zero-based indices that were displaced or removed, the
  OUTPUT positions that hold fabricated points, and a full index map from
  output position back to original index.

THE INJECTED COUNT IS NOT AN EDIT-COUNT TARGET
----------------------------------------------
``CorruptionRecord.n_injected`` counts what this module DID. It is NOT a
prediction of the optimal cost returned by ``discrete_edit_distance``, and any
test asserting equality between the two would be wrong. The project plan says
so explicitly (section 4.2): "Correct edit count need not equal injected edit
count because repeated samples and delta create alternative optimal repairs."
Concretely:

* a displaced vertex that still lands within ``delta`` of the reference needs
  no edit at all;
* two adjacent dropped samples may cost zero edits, because a discrete Frechet
  coupling may simply stall on a neighbouring vertex;
* one deletion can absorb several injected anomalies when the reference has a
  repeated vertex to stall on;
* ``delta`` is a free parameter, so the same corruption has different optimal
  repair costs at different thresholds.

Use the records for EDIT LOCALIZATION analysis (which indices did a method
flag?), never as an expected cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CorruptionRecord",
    "PipelineResult",
    "apply_pipeline",
    "burst",
    "drop_samples",
    "jitter",
    "prefix_suffix_noise",
    "resample",
    "spike",
]


# --------------------------------------------------------------------------
# record type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorruptionRecord:
    """What one corruption operator did, in the index space of ITS OWN input.

    Fields
    ------
    operator
        The operator name, e.g. ``"spike"``.
    params
        The operator arguments actually used, JSON-safe.
    original_length, corrupted_length
        Point counts before and after.
    displaced_original_indices
        Original indices whose coordinates were MOVED but which survive.
    removed_original_indices
        Original indices that were DELETED.
    affected_original_indices
        The sorted union of the two above. This is the localization ground
        truth: exactly these original indices were touched.
    fabricated_output_indices
        Positions in the OUTPUT curve holding points that did not exist in the
        input (interpolated or invented).
    output_to_original
        Length ``corrupted_length``. Entry ``i`` is the original index that
        output point ``i`` came from, or ``None`` when it was fabricated. The
        non-``None`` entries are strictly increasing.
    n_injected
        How many anomalies this operator injected. NOT an optimal edit count;
        see the module docstring.
    """

    operator: str
    params: dict[str, Any]
    original_length: int
    corrupted_length: int
    displaced_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]
    fabricated_output_indices: tuple[int, ...]
    output_to_original: tuple[int | None, ...]
    n_injected: int

    @property
    def affected_original_indices(self) -> tuple[int, ...]:
        """Sorted union of displaced and removed original indices."""
        return tuple(
            sorted(set(self.displaced_original_indices) | set(self.removed_original_indices))
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {
            "operator": self.operator,
            "params": dict(self.params),
            "original_length": self.original_length,
            "corrupted_length": self.corrupted_length,
            "displaced_original_indices": list(self.displaced_original_indices),
            "removed_original_indices": list(self.removed_original_indices),
            "affected_original_indices": list(self.affected_original_indices),
            "fabricated_output_indices": list(self.fabricated_output_indices),
            "output_to_original": list(self.output_to_original),
            "n_injected": self.n_injected,
            "note": (
                "n_injected is what was injected, NOT the optimal edit count; "
                "delta and repeated samples permit cheaper repairs"
            ),
        }


def _validate_record(record: CorruptionRecord) -> CorruptionRecord:
    """Internal self-check: a record must describe a consistent index map."""
    k = record.original_length
    if len(record.output_to_original) != record.corrupted_length:
        raise AssertionError(f"{record.operator}: index map length disagrees with output length")
    seen: list[int] = []
    for position, source in enumerate(record.output_to_original):
        if source is None:
            if position not in record.fabricated_output_indices:
                raise AssertionError(f"{record.operator}: unlisted fabricated position {position}")
            continue
        if not 0 <= source < k:
            raise AssertionError(f"{record.operator}: index map names {source}, outside 0..{k - 1}")
        seen.append(source)
    if seen != sorted(seen) or len(set(seen)) != len(seen):
        raise AssertionError(f"{record.operator}: index map is not strictly increasing")
    surviving = set(seen)
    for index in record.removed_original_indices:
        if index in surviving:
            raise AssertionError(f"{record.operator}: index {index} is both removed and present")
    for index in record.affected_original_indices:
        if not 0 <= index < k:
            raise AssertionError(f"{record.operator}: affected index {index} does not exist")
    return record


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _as_curve(curve: Any) -> np.ndarray:
    """Validate and copy a curve argument to a fresh ``(k, d)`` float64 array."""
    try:
        arr = np.array(curve, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("curve must be a numeric (k, d) array") from exc
    if arr.ndim != 2:
        raise ValueError(f"curve must be a 2-D (k, d) array, got ndim={arr.ndim}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"curve must be non-empty in both axes, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("curve contains NaN or infinite coordinates")
    return arr


def _as_generator(rng: Any) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise ValueError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    return rng


def _non_negative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return number


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {number}")
    return number


def _unit_vectors(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    """``count`` independent uniform directions on the unit sphere of R^dim."""
    raw = rng.normal(0.0, 1.0, size=(count, dim))
    norms = np.sqrt(np.sum(raw * raw, axis=1))
    degenerate = norms <= 0.0
    if np.any(degenerate):
        # Measure-zero fallback so a direction always exists and stays seeded.
        raw[degenerate] = 0.0
        raw[degenerate, 0] = 1.0
        norms[degenerate] = 1.0
    return raw / norms[:, None]


def _interior_indices(length: int) -> np.ndarray:
    """Indices ``1..length-2``.

    Endpoints are excluded by default because every Frechet coupling PINS them
    (``docs/definition.md`` section 2), so corrupting an endpoint changes the
    problem qualitatively. Endpoint corruption is a separate ablation, selected
    with ``include_endpoints=True`` or by ``prefix_suffix_noise``.
    """
    return np.arange(1, max(length - 1, 1), dtype=np.int64)


def _identity_map(length: int) -> tuple[int | None, ...]:
    return tuple(range(length))


# --------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------


def spike(
    curve: Any,
    rng: Any,
    n_spikes: int,
    magnitude: float,
    *,
    include_endpoints: bool = False,
) -> tuple[np.ndarray, CorruptionRecord]:
    """Displace ``n_spikes`` isolated vertices by exactly ``magnitude``.

    Each chosen vertex is moved along an independent uniform random direction,
    by exactly ``magnitude`` coordinate units. Vertices are chosen WITHOUT
    replacement, so ``n_spikes`` distinct original indices are displaced. The
    point count is unchanged.

    ``include_endpoints=False`` (the default) keeps index 0 and index ``k-1``
    untouched, because Frechet couplings pin the endpoints.

    Raises ``ValueError`` when ``n_spikes`` exceeds the number of eligible
    vertices; it never silently injects fewer than requested.
    """
    arr = _as_curve(curve)
    generator = _as_generator(rng)
    count = _count(n_spikes, "n_spikes")
    size = _non_negative(magnitude, "magnitude")
    length, dim = arr.shape

    if include_endpoints:
        pool = np.arange(length, dtype=np.int64)
    elif length <= 2:
        pool = np.empty(0, dtype=np.int64)
    else:
        pool = _interior_indices(length)
    if count > pool.size:
        raise ValueError(
            f"cannot place {count} spikes: only {pool.size} eligible vertices "
            f"(length {length}, include_endpoints={include_endpoints})"
        )

    chosen = np.sort(generator.choice(pool, size=count, replace=False)) if count else pool[:0]
    out = arr.copy()
    if count:
        out[chosen] = arr[chosen] + size * _unit_vectors(generator, count, dim)

    record = CorruptionRecord(
        operator="spike",
        params={
            "n_spikes": count,
            "magnitude": size,
            "include_endpoints": include_endpoints,
        },
        original_length=length,
        corrupted_length=length,
        displaced_original_indices=tuple(int(i) for i in chosen),
        removed_original_indices=(),
        fabricated_output_indices=(),
        output_to_original=_identity_map(length),
        n_injected=count,
    )
    return out, _validate_record(record)


def burst(
    curve: Any,
    rng: Any,
    length: int,
    magnitude: float,
    *,
    include_endpoints: bool = False,
) -> tuple[np.ndarray, CorruptionRecord]:
    """Offset one CONTIGUOUS run of ``length`` vertices by a shared vector.

    Every vertex of the run is translated by the SAME vector, of norm exactly
    ``magnitude`` and uniformly random direction. This models a correlated
    loss-of-lock excursion rather than ``length`` independent spikes, and it is
    a materially harder repair: the run has no internal inconsistency, only a
    collective one.

    ``n_injected`` is the run length. It is not an edit count: a single stall in
    the coupling can sometimes absorb several run vertices.
    """
    arr = _as_curve(curve)
    generator = _as_generator(rng)
    run = _count(length, "length")
    size = _non_negative(magnitude, "magnitude")
    total, dim = arr.shape

    lo = 0 if include_endpoints else 1
    hi = total if include_endpoints else total - 1  # exclusive
    span = max(hi - lo, 0)
    if run > span:
        raise ValueError(
            f"cannot place a burst of {run} vertices: only {span} eligible "
            f"positions (length {total}, include_endpoints={include_endpoints})"
        )

    out = arr.copy()
    if run:
        start = int(generator.integers(lo, hi - run + 1))
        indices = np.arange(start, start + run, dtype=np.int64)
        offset = size * _unit_vectors(generator, 1, dim)[0]
        out[indices] = arr[indices] + offset
    else:
        indices = np.empty(0, dtype=np.int64)

    record = CorruptionRecord(
        operator="burst",
        params={
            "length": run,
            "magnitude": size,
            "include_endpoints": include_endpoints,
        },
        original_length=total,
        corrupted_length=total,
        displaced_original_indices=tuple(int(i) for i in indices),
        removed_original_indices=(),
        fabricated_output_indices=(),
        output_to_original=_identity_map(total),
        n_injected=run,
    )
    return out, _validate_record(record)


def drop_samples(
    curve: Any,
    rng: Any,
    fraction: float,
    *,
    include_endpoints: bool = False,
) -> tuple[np.ndarray, CorruptionRecord]:
    """Delete a ``fraction`` of the vertices, chosen uniformly without replacement.

    The number dropped is ``round(fraction * eligible)``, where ``eligible``
    excludes the two endpoints unless ``include_endpoints=True``. The output is
    shorter than the input; ``output_to_original`` records which original index
    every surviving point came from.

    Dropping samples is exactly the situation INSERTION is meant to repair, but
    insertion does not help every dropout: stationary-point duplication and
    missed samples that discrete Frechet already tolerates need no insertion
    at all. Expect a repair cost well below the number dropped.
    """
    arr = _as_curve(curve)
    generator = _as_generator(rng)
    share = _non_negative(fraction, "fraction")
    if share > 1.0:
        raise ValueError(f"fraction must be <= 1, got {share}")
    total = arr.shape[0]

    if include_endpoints:
        pool = np.arange(total, dtype=np.int64)
    elif total <= 2:
        pool = np.empty(0, dtype=np.int64)
    else:
        pool = _interior_indices(total)
    n_drop = round(share * pool.size)
    if include_endpoints and n_drop >= total:
        n_drop = total - 1  # never return an empty curve

    dropped = np.sort(generator.choice(pool, size=n_drop, replace=False)) if n_drop else pool[:0]
    keep_mask = np.ones(total, dtype=bool)
    keep_mask[dropped] = False
    kept = np.flatnonzero(keep_mask)
    out = arr[kept].copy()

    record = CorruptionRecord(
        operator="drop_samples",
        params={
            "fraction": share,
            "include_endpoints": include_endpoints,
            "n_dropped": n_drop,
        },
        original_length=total,
        corrupted_length=int(kept.size),
        displaced_original_indices=(),
        removed_original_indices=tuple(int(i) for i in dropped),
        fabricated_output_indices=(),
        output_to_original=tuple(int(i) for i in kept),
        n_injected=n_drop,
    )
    return out, _validate_record(record)


def resample(curve: Any, factor: Any) -> tuple[np.ndarray, CorruptionRecord]:
    """Change the sampling RATE. Deterministic: it takes no generator.

    * ``factor`` an integer ``>= 1``: UPSAMPLE. Each edge is split into
      ``factor`` equal parts, so ``factor - 1`` linearly interpolated points are
      fabricated per edge. No original vertex moves or disappears.
    * ``0 < factor < 1``: DOWNSAMPLE. Every ``round(1/factor)``-th vertex is
      kept; the first and last vertices are always kept. Skipped originals are
      recorded as removed.

    Sampling-rate mismatch is an experiment CONDITION, not noise: it changes
    edit costs for every method, so it must be applied identically across
    methods and reported as its own arm rather than folded into a baseline.
    """
    arr = _as_curve(curve)
    ratio = float(factor)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"factor must be finite and strictly positive, got {factor!r}")
    total = arr.shape[0]

    if ratio >= 1.0:
        if not ratio.is_integer():
            raise ValueError(f"an upsampling factor must be an integer, got {factor!r}")
        step = int(ratio)
        if step == 1 or total == 1:
            record = CorruptionRecord(
                operator="resample",
                params={"factor": ratio, "direction": "identity"},
                original_length=total,
                corrupted_length=total,
                displaced_original_indices=(),
                removed_original_indices=(),
                fabricated_output_indices=(),
                output_to_original=_identity_map(total),
                n_injected=0,
            )
            return arr.copy(), _validate_record(record)

        pieces: list[np.ndarray] = []
        index_map: list[int | None] = []
        for i in range(total - 1):
            start, end = arr[i], arr[i + 1]
            for s in range(step):
                pieces.append(start + (end - start) * (s / step))
                index_map.append(i if s == 0 else None)
        pieces.append(arr[total - 1])
        index_map.append(total - 1)
        out = np.asarray(pieces, dtype=np.float64)
        fabricated = tuple(i for i, src in enumerate(index_map) if src is None)
        record = CorruptionRecord(
            operator="resample",
            params={"factor": ratio, "direction": "upsample", "step": step},
            original_length=total,
            corrupted_length=int(out.shape[0]),
            displaced_original_indices=(),
            removed_original_indices=(),
            fabricated_output_indices=fabricated,
            output_to_original=tuple(index_map),
            n_injected=len(fabricated),
        )
        return out, _validate_record(record)

    stride = max(round(1.0 / ratio), 2)
    kept = sorted(set(range(0, total, stride)) | {0, total - 1})
    kept_set = set(kept)
    removed = [i for i in range(total) if i not in kept_set]
    out = arr[np.asarray(kept, dtype=np.int64)].copy()
    record = CorruptionRecord(
        operator="resample",
        params={"factor": ratio, "direction": "downsample", "stride": stride},
        original_length=total,
        corrupted_length=int(out.shape[0]),
        displaced_original_indices=(),
        removed_original_indices=tuple(removed),
        fabricated_output_indices=(),
        output_to_original=tuple(kept),
        n_injected=len(removed),
    )
    return out, _validate_record(record)


def jitter(curve: Any, rng: Any, sigma: float) -> tuple[np.ndarray, CorruptionRecord]:
    """Add independent Gaussian noise of standard deviation ``sigma`` to EVERY coordinate.

    This is BACKGROUND noise, not a localizable anomaly. Every original index is
    reported as displaced, and ``n_injected`` is therefore ``0``: there is no
    discrete anomaly to repair and no sensible per-index localization target.
    Reporting jitter as ``k`` injected edits would be meaningless.
    """
    arr = _as_curve(curve)
    generator = _as_generator(rng)
    scale = _non_negative(sigma, "sigma")
    total = arr.shape[0]
    out = arr + generator.normal(0.0, scale, size=arr.shape) if scale > 0.0 else arr.copy()

    record = CorruptionRecord(
        operator="jitter",
        params={"sigma": scale},
        original_length=total,
        corrupted_length=total,
        displaced_original_indices=tuple(range(total)) if scale > 0.0 else (),
        removed_original_indices=(),
        fabricated_output_indices=(),
        output_to_original=_identity_map(total),
        n_injected=0,
    )
    return out, _validate_record(record)


def prefix_suffix_noise(
    curve: Any,
    rng: Any,
    n: int,
    *,
    step: float | None = None,
) -> tuple[np.ndarray, CorruptionRecord]:
    """Prepend and append ``n`` fabricated wandering points at each end.

    Models a receiver that logged garbage before departure and after arrival.
    The fabricated points form an undirected random walk of step length
    ``step`` (default: the median spacing of the input curve) leading into the
    first original vertex and away from the last one.

    No ORIGINAL index is touched, so ``affected_original_indices`` is empty; the
    damage lives entirely at fabricated OUTPUT positions. This corruption is
    especially punishing for the Frechet family, whose couplings pin the
    endpoints, and nearly free for LCSS, whose prefixes and suffixes cost
    nothing. That asymmetry is a real property of the measures, not a bug.
    """
    arr = _as_curve(curve)
    generator = _as_generator(rng)
    count = _count(n, "n")
    total, dim = arr.shape

    if step is None:
        spacing = 1.0
        if total >= 2:
            gaps = np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))
            if gaps.size:
                spacing = float(np.median(gaps))
        if not math.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
    else:
        spacing = _non_negative(step, "step")

    if count == 0:
        record = CorruptionRecord(
            operator="prefix_suffix_noise",
            params={"n": 0, "step": spacing},
            original_length=total,
            corrupted_length=total,
            displaced_original_indices=(),
            removed_original_indices=(),
            fabricated_output_indices=(),
            output_to_original=_identity_map(total),
            n_injected=0,
        )
        return arr.copy(), _validate_record(record)

    prefix_steps = spacing * _unit_vectors(generator, count, dim)
    suffix_steps = spacing * _unit_vectors(generator, count, dim)

    # Walk backwards out of the first vertex, then reverse so the walk arrives.
    back = arr[0] - np.cumsum(prefix_steps, axis=0)
    prefix = back[::-1]
    suffix = arr[total - 1] + np.cumsum(suffix_steps, axis=0)

    out = np.vstack([prefix, arr, suffix])
    index_map: list[int | None] = [None] * count + list(range(total)) + [None] * count
    fabricated = tuple(i for i, src in enumerate(index_map) if src is None)

    record = CorruptionRecord(
        operator="prefix_suffix_noise",
        params={"n": count, "step": spacing},
        original_length=total,
        corrupted_length=int(out.shape[0]),
        displaced_original_indices=(),
        removed_original_indices=(),
        fabricated_output_indices=fabricated,
        output_to_original=tuple(index_map),
        n_injected=2 * count,
    )
    return out, _validate_record(record)


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """Result of applying several operators in order.

    ``records[i]`` describes step ``i`` in the index space of THAT STEP'S INPUT,
    not of the original curve. ``output_to_original`` is the composed map, from
    a final output position back to an index of the curve the pipeline started
    from (``None`` when the point was fabricated at some step).
    """

    curve: np.ndarray
    records: tuple[CorruptionRecord, ...]
    output_to_original: tuple[int | None, ...]

    @property
    def first_step_affected_indices(self) -> tuple[int, ...]:
        """Original indices touched by step 0 only.

        Later steps index an already-corrupted curve, so their indices are NOT
        comparable with the original ones without re-mapping. Composing
        localization ground truth across steps is deliberately not attempted
        here: it would be guesswork for operators that change the point count.
        """
        return self.records[0].affected_original_indices if self.records else ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": [r.as_dict() for r in self.records],
            "output_to_original": list(self.output_to_original),
            "final_length": int(self.curve.shape[0]),
        }


def apply_pipeline(
    curve: Any,
    steps: Sequence[Callable[[np.ndarray], tuple[np.ndarray, CorruptionRecord]]],
) -> PipelineResult:
    """Apply ``steps`` in order, composing their index maps back to the original.

    Each step is a callable taking a curve and returning
    ``(curve, CorruptionRecord)``; bind the generator and parameters with
    ``functools.partial`` at the call site so the seeding stays explicit.
    """
    current = _as_curve(curve)
    composed: tuple[int | None, ...] = _identity_map(current.shape[0])
    records: list[CorruptionRecord] = []
    for step in steps:
        current, record = step(current)
        current = _as_curve(current)
        records.append(record)
        composed = tuple(
            None if src is None else composed[src] for src in record.output_to_original
        )
    return PipelineResult(curve=current, records=tuple(records), output_to_original=composed)
