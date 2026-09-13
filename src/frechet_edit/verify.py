"""Independent verification of an :class:`~frechet_edit._types.EditResult`.

This module deliberately shares NOTHING with the solver beyond input
validation and the certified point predicate. It does not import the dynamic
program, the minimum queue, or the traceback, and it never trusts a stored
value: it replays the edits itself, recounts them, re-checks every coupling
step, and recomputes ordinary discrete Frechet on the edited curve with its own
plain implementation.

See ``docs/witness-invariants.md`` section 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._numerics import NumericPolicy, exact_within, within
from ._types import Deletion, EditResult, Insertion


@dataclass(slots=True)
class VerificationReport:
    """Outcome of verification. ``ok`` is true only if ``violations`` is empty."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    replayed_curve: np.ndarray | None = None
    ordinary_frechet: float | None = None

    def __bool__(self) -> bool:
        return self.ok

    def raise_for_status(self) -> None:
        if not self.ok:
            raise AssertionError("witness verification failed:\n  " + "\n  ".join(self.violations))


def replay(
    observation: np.ndarray,
    edits: tuple[Deletion | Insertion, ...],
) -> np.ndarray:
    """Apply ``edits`` to the ORIGINAL observation.

    Indices never shift: deleting index 3 does not renumber index 4. Insertion
    gap ``g`` places a point immediately before original index ``g``; ``g == n``
    places it after the last original vertex. Ordering is ascending gap, then
    ascending order within a gap, and that is retained even when every original
    vertex is deleted.
    """
    observation = np.asarray(observation, dtype=np.float64)
    n = len(observation)

    deleted: set[int] = set()
    for edit in edits:
        if isinstance(edit, Deletion):
            if not 0 <= edit.index < n:
                raise ValueError(f"deletion index {edit.index} out of range 0..{n - 1}")
            if edit.index in deleted:
                raise ValueError(f"observation index {edit.index} deleted more than once")
            deleted.add(edit.index)

    by_gap: dict[int, list[Insertion]] = {}
    for edit in edits:
        if isinstance(edit, Insertion):
            if not 0 <= edit.gap <= n:
                raise ValueError(f"insertion gap {edit.gap} out of range 0..{n}")
            by_gap.setdefault(edit.gap, []).append(edit)
    for gap, group in by_gap.items():
        orders = sorted(ins.order for ins in group)
        if orders != list(range(len(orders))):
            raise ValueError(f"insertions in gap {gap} have non-contiguous order values {orders}")

    out: list[np.ndarray] = []
    dim = observation.shape[1]
    for gap in range(n + 1):
        for ins in sorted(by_gap.get(gap, []), key=lambda e: e.order):
            point = np.asarray(ins.point, dtype=np.float64)
            if point.shape != (dim,):
                raise ValueError(f"inserted point {ins.point!r} is not {dim}-dimensional")
            out.append(point)
        if gap < n and gap not in deleted:
            out.append(observation[gap])
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, dim), dtype=np.float64)


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Ordinary strong discrete Frechet distance, written straight from the definition.

    A separate, deliberately simple implementation used only to cross-check
    witnesses. Empty against non-empty is ``inf``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0 if len(a) == len(b) else float("inf")

    prev = [float("inf")] * len(b)
    for i in range(len(a)):
        cur = [float("inf")] * len(b)
        for j in range(len(b)):
            d = float(np.linalg.norm(a[i] - b[j]))
            if i == 0 and j == 0:
                cur[j] = d
            elif i == 0:
                cur[j] = max(cur[j - 1], d)
            elif j == 0:
                cur[j] = max(prev[j], d)
            else:
                cur[j] = max(min(prev[j], prev[j - 1], cur[j - 1]), d)
        prev = cur
    return prev[-1]


def _check_coupling(
    reference: np.ndarray,
    edited: np.ndarray,
    coupling: tuple[tuple[int, int], ...],
    delta: float,
    policy: NumericPolicy,
    violations: list[str],
) -> None:
    m, k = len(reference), len(edited)
    if not coupling:
        violations.append("coupling is empty")
        return
    if coupling[0] != (0, 0):
        violations.append(f"coupling starts at {coupling[0]}, expected (0, 0)")
    if coupling[-1] != (m - 1, k - 1):
        violations.append(f"coupling ends at {coupling[-1]}, expected {(m - 1, k - 1)}")

    for idx, (ri, ei) in enumerate(coupling):
        if not (0 <= ri < m and 0 <= ei < k):
            violations.append(f"coupling pair {idx} = {(ri, ei)} is out of range")
            return

    for idx in range(1, len(coupling)):
        pri, pei = coupling[idx - 1]
        ri, ei = coupling[idx]
        step = (ri - pri, ei - pei)
        if step not in {(1, 0), (0, 1), (1, 1)}:
            violations.append(
                f"coupling step {idx} is {step} from {(pri, pei)} to {(ri, ei)}; "
                "only (1,0), (0,1) and (1,1) are legal"
            )

    for idx, (ri, ei) in enumerate(coupling):
        if not within(reference[ri], edited[ei], delta, policy=policy):
            dist = float(np.linalg.norm(reference[ri] - edited[ei]))
            violations.append(
                f"coupling pair {idx} = {(ri, ei)} has distance {dist!r} > delta {delta!r}"
            )


def verify_witness(
    reference: np.ndarray,
    observation: np.ndarray,
    result: EditResult,
    *,
    policy: NumericPolicy = "certified",
    check_ordinary_frechet: bool = True,
) -> VerificationReport:
    """Re-derive and check every claim a witness makes.

    A passing report establishes FEASIBILITY at the reported cost. It does not
    establish minimality; that comes from oracle agreement.
    """
    violations: list[str] = []
    reference = np.asarray(reference, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)

    if result.witness_status != "certified":
        return VerificationReport(
            ok=False,
            violations=[f"witness_status is {result.witness_status!r}, nothing to verify"],
        )
    if result.edits is None or result.edited_curve is None or result.coupling is None:
        return VerificationReport(ok=False, violations=["witness fields are incomplete"])

    try:
        replayed = replay(observation, result.edits)
    except ValueError as exc:
        return VerificationReport(ok=False, violations=[f"replay failed: {exc}"])

    if replayed.shape != result.edited_curve.shape or not np.array_equal(
        replayed, result.edited_curve
    ):
        violations.append(
            f"replayed curve (shape {replayed.shape}) does not match the reported "
            f"edited_curve (shape {result.edited_curve.shape})"
        )

    if result.cost is None or len(result.edits) != result.cost:
        violations.append(f"witness has {len(result.edits)} edits but cost is {result.cost!r}")

    n_ins = sum(isinstance(e, Insertion) for e in result.edits)
    n_del = sum(isinstance(e, Deletion) for e in result.edits)
    if result.mode == "delete" and n_ins:
        violations.append(f"mode 'delete' but witness contains {n_ins} insertion(s)")
    if result.mode == "insert" and n_del:
        violations.append(f"mode 'insert' but witness contains {n_del} deletion(s)")

    deleted = {e.index for e in result.edits if isinstance(e, Deletion)}
    kept = [observation[idx] for idx in range(len(observation)) if idx not in deleted]
    pos = 0
    for point in replayed:
        if pos < len(kept) and np.array_equal(point, kept[pos]):
            pos += 1
    if pos != len(kept):
        violations.append("retained original vertices do not appear unchanged and in order")

    if len(replayed) == 0:
        violations.append("edited curve is empty, so no coupling with a non-empty reference exists")
    else:
        _check_coupling(reference, replayed, result.coupling, result.delta, policy, violations)

    ordinary: float | None = None
    if check_ordinary_frechet and len(replayed) > 0:
        ordinary = discrete_frechet(reference, replayed)
        if not exact_within([ordinary], [0.0], result.delta):
            violations.append(
                f"independent discrete Frechet of the edited curve is {ordinary!r}, "
                f"which exceeds delta {result.delta!r}"
            )

    return VerificationReport(
        ok=not violations,
        violations=violations,
        replayed_curve=replayed,
        ordinary_frechet=ordinary,
    )
