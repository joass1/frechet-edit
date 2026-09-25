"""Strong Frechet edit distance with verifiable minimal-edit witnesses.

An independent implementation of the discrete algorithms (deletion, insertion,
both) and of the continuous deletion-only algorithm of Fox, Nayyeri, Perry and
Raichel, *Frechet Edit Distance*, SoCG 2024
(DOI 10.4230/LIPIcs.SoCG.2024.58; full version arXiv:2403.12878).

    >>> import numpy as np
    >>> from frechet_edit import discrete_edit_distance
    >>> reference = np.array([[0.0], [1.0], [2.0]])
    >>> observation = np.array([[0.0], [1.0], [100.0], [2.0]])
    >>> result = discrete_edit_distance(reference, observation, delta=0.1)
    >>> result.status, result.cost
    ('optimal', 1)

The returned cost is a COUNT OF EDITS, not a distance. The measure is directed:
edits apply to ``observation`` only. See ``docs/definition.md`` and, for the
continuous variant, ``docs/continuous.md``.
"""

from ._numerics import NumericallyAmbiguous
from ._types import (
    Deletion,
    EditResult,
    Insertion,
    UnsupportedDimensionError,
    UnsupportedOperationError,
)
from .api import (
    continuous_edit_distance,
    continuous_frechet_within,
    discrete_edit_distance,
    ordinary_discrete_frechet,
)
from .verify import (
    VerificationReport,
    continuous_frechet_le,
    replay,
    verify_continuous_witness,
    verify_witness,
)

__version__ = "0.1.0"

__all__ = [
    "Deletion",
    "EditResult",
    "Insertion",
    "NumericallyAmbiguous",
    "UnsupportedDimensionError",
    "UnsupportedOperationError",
    "VerificationReport",
    "__version__",
    "continuous_edit_distance",
    "continuous_frechet_le",
    "continuous_frechet_within",
    "discrete_edit_distance",
    "ordinary_discrete_frechet",
    "replay",
    "verify_continuous_witness",
    "verify_witness",
]
