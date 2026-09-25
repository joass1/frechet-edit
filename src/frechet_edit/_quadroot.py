"""Exact arithmetic on numbers of the form ``x + s * sqrt(y)``.

Every endpoint of a continuous free-space interval is a root of a quadratic
with float64 (hence rational) coefficients, so it has exactly this form with
``x`` and ``y`` rational. Deciding the order of two such endpoints is the only
irrational comparison the continuous solver ever needs, and it can be done with
no rounding at all by squaring with careful sign bookkeeping. That is what this
module does; ``docs/continuous.md`` section 4 states the policy.

Nothing here is approximate. Floats appear only in the enclosures produced
elsewhere to AVOID calling into this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

__all__ = ["ONE", "ZERO", "QuadNum", "compare", "sign_plus_root"]


@dataclass(frozen=True, slots=True)
class QuadNum:
    """The real number ``x + s * sqrt(y)`` with rational ``x``, ``y >= 0``.

    ``s`` is ``-1``, ``0`` or ``+1``. When ``s == 0`` or ``y == 0`` the number
    is the rational ``x``.
    """

    x: Fraction
    s: int
    y: Fraction

    def __post_init__(self) -> None:
        if self.s not in (-1, 0, 1):
            raise ValueError(f"s must be -1, 0 or 1, got {self.s!r}")
        if self.y < 0:
            raise ValueError(f"y must be non-negative, got {self.y!r}")

    @classmethod
    def rational(cls, value: Fraction | int) -> QuadNum:
        return cls(Fraction(value), 0, Fraction(0))


ZERO = QuadNum.rational(0)
ONE = QuadNum.rational(1)


def _sign(value: Fraction) -> int:
    return (value > 0) - (value < 0)


def sign_plus_root(w: Fraction, c: Fraction, y: Fraction) -> int:
    """Exact sign of ``w + c * sqrt(y)`` for rational ``w``, ``c`` and ``y >= 0``."""
    sw = _sign(w)
    sc = _sign(c)
    if sc == 0 or y == 0:
        return sw
    if sw == 0 or sw == sc:
        return sc
    # Opposite signs: the larger magnitude wins. Both sides are non-negative,
    # so comparing squares preserves the order.
    lhs = w * w
    rhs = c * c * y
    if lhs > rhs:
        return sw
    if lhs < rhs:
        return sc
    return 0


def compare(p: QuadNum, q: QuadNum) -> int:
    """Exact ``sign(p - q)``: ``-1``, ``0`` or ``+1``.

    Writes ``p - q = L - R`` with ``L = (xp - xq) + sp*sqrt(yp)`` and
    ``R = sq*sqrt(yq)``. When ``L`` and ``R`` have different signs the answer is
    immediate; when they share a sign, ``sign(L - R)`` is that sign times
    ``sign(L^2 - R^2)``, and ``L^2 - R^2`` again has the one-root form.
    """
    u = p.x - q.x
    a1 = p.s if p.y != 0 else 0
    a2 = q.s if q.y != 0 else 0
    if a2 == 0:
        return sign_plus_root(u, Fraction(a1), p.y)
    if a1 == 0:
        return sign_plus_root(u, Fraction(-a2), q.y)

    s_left = sign_plus_root(u, Fraction(a1), p.y)
    s_right = a2  # sign of a2 * sqrt(yq), with yq > 0
    if s_left != s_right:
        return 1 if s_left > s_right else -1
    # Same non-zero sign. L^2 - R^2 = (u^2 + yp - yq) + 2*a1*u*sqrt(yp).
    squares = sign_plus_root(u * u + p.y - q.y, Fraction(2 * a1) * u, p.y)
    return s_left * squares
