"""The matrix update: orthogonalization, row normalization, momentum form.

``DION3_UPDATE`` (the default) is row-selected NorMuon with heavy-ball momentum.
``MUON_UPDATE`` is Keller Jordan's Muon (microsoft/dion's ``Muon(nesterov=True,
adjust_lr="rms_norm")`` with Keller's iteration) once the groups update every
row (``param_groups(model, fraction=1.0)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from dionw.newton_schulz import NewtonSchulz

__all__ = [
    "DION3_UPDATE",
    "MUON_UPDATE",
    "MatrixUpdate",
    "MomentumForm",
    "RowNormalization",
]


class RowNormalization(Enum):
    """What a matrix step does to the orthogonalized rows."""

    # NorMuon: divide each row by the root of an EMA of its mean square, then
    # restore the block's Frobenius norm.
    NORMUON = "normuon"
    # None: the orthogonalized update as it is (Muon).
    NONE = "none"


class MomentumForm(Enum):
    """Which momentum a matrix step orthogonalizes.

    Both keep ``B = momentum * B + G``; ``HEAVY_BALL`` orthogonalizes ``B``,
    ``NESTEROV`` orthogonalizes ``G + momentum * B``. Nesterov needs every row
    per step: row selection's error feedback has no Nesterov form.
    """

    HEAVY_BALL = "heavy_ball"
    NESTEROV = "nesterov"


@dataclass(frozen=True)
class MatrixUpdate:
    """How a matrix step turns momentum into an update.

    Attributes:
        newton_schulz: The orthogonalization.
        row_normalization: NorMuon row normalization, or none (Muon).
        momentum_form: Heavy-ball or Nesterov momentum (Nesterov needs every
            row per step).
    """

    newton_schulz: NewtonSchulz
    row_normalization: RowNormalization
    momentum_form: MomentumForm


DION3_UPDATE: Final[MatrixUpdate] = MatrixUpdate(
    newton_schulz=NewtonSchulz.GRAM,
    row_normalization=RowNormalization.NORMUON,
    momentum_form=MomentumForm.HEAVY_BALL,
)
MUON_UPDATE: Final[MatrixUpdate] = MatrixUpdate(
    newton_schulz=NewtonSchulz.KELLER_TRITON,
    row_normalization=RowNormalization.NONE,
    momentum_form=MomentumForm.NESTEROV,
)
