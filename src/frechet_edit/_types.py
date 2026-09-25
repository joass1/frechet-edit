"""Public result types and exceptions.

The contract these implement is ``docs/definition.md`` section 7. Two points
are load-bearing:

* ``status`` and ``witness_status`` are independent. A certified-optimal cost
  with an unavailable witness is ``status="optimal"``, and must never be
  presented as mathematical infeasibility.
* ``numerically_ambiguous`` is a reported outcome, not an exception and not a
  disguised infinity.
* ``budget_exceeded`` (continuous solver only) means "no solution within the
  caller's ``max_deletions``". It says nothing about larger budgets and is
  never a disguised infinity either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Operations = Literal["delete", "insert", "both"]
Status = Literal["optimal", "infeasible", "numerically_ambiguous", "budget_exceeded"]
WitnessStatus = Literal["not_requested", "certified", "unavailable"]

OPERATIONS: tuple[Operations, ...] = ("delete", "insert", "both")
BACKENDS: tuple[str, ...] = ("python", "reference")


class UnsupportedDimensionError(ValueError):
    """Raised when a mode is asked for a dimension its backend does not certify."""


class UnsupportedOperationError(ValueError):
    """Raised when an edit mode exists in the theory but not in this package.

    Continuous insertion and mixed edits need the paper's minimum-link and
    canonical-subcurve machinery; they are not implemented, and asking for them
    is an error rather than a silent fallback to deletion or to discrete.
    """


@dataclass(frozen=True, slots=True)
class Deletion:
    """Removal of one original observation vertex.

    ``index`` is the ORIGINAL zero-based index in the input observation and
    never shifts as other vertices are removed.
    """

    index: int

    def as_dict(self) -> dict[str, Any]:
        return {"op": "delete", "index": self.index}


@dataclass(frozen=True, slots=True)
class Insertion:
    """Insertion of one new vertex at an arbitrary Euclidean location.

    ``gap`` is in ``0..n`` and means "immediately before original index gap";
    ``gap == n`` means after the last original vertex. ``order`` disambiguates
    several insertions sharing a gap, ascending.
    """

    gap: int
    order: int
    point: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": "insert",
            "gap": self.gap,
            "order": self.order,
            "point": list(self.point),
        }


@dataclass(frozen=True, slots=True)
class EditResult:
    """Outcome of one edit-distance query. See ``docs/definition.md`` section 7."""

    status: Status
    cost: int | float | None
    mode: Operations
    delta: float
    dimension: int
    backend: str
    numeric_policy: str
    witness_status: WitnessStatus = "not_requested"
    edited_curve: np.ndarray | None = None
    edits: tuple[Deletion | Insertion, ...] | None = None
    coupling: tuple[tuple[int, int], ...] | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    detail: str | None = None

    @property
    def is_feasible(self) -> bool:
        """True only for a certified optimal cost. Ambiguity is not feasibility."""
        return self.status == "optimal"

    def to_json_obj(self) -> dict[str, Any]:
        """A JSON-safe dict. Never emits a bare ``Infinity`` token."""
        cost: int | None = int(self.cost) if self.status == "optimal" else None  # type: ignore[arg-type]
        return {
            "status": self.status,
            "cost": cost,
            "cost_is_infinite": self.status == "infeasible",
            "mode": self.mode,
            "delta": self.delta,
            "dimension": self.dimension,
            "backend": self.backend,
            "numeric_policy": self.numeric_policy,
            "witness_status": self.witness_status,
            "edited_curve": (
                None if self.edited_curve is None else self.edited_curve.tolist()
            ),
            "edits": None if self.edits is None else [e.as_dict() for e in self.edits],
            "coupling": (
                None if self.coupling is None else [list(p) for p in self.coupling]
            ),
            "stats": dict(self.stats),
            "detail": self.detail,
        }

    def __repr__(self) -> str:
        cost = "None"
        if self.cost is not None:
            cost = "inf" if math.isinf(float(self.cost)) else str(self.cost)
        return (
            f"EditResult(status={self.status!r}, cost={cost}, mode={self.mode!r}, "
            f"delta={self.delta!r}, witness_status={self.witness_status!r})"
        )
