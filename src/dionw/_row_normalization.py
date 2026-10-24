"""Step 4 of a matrix step: NorMuon row normalization or none, times the step."""

from __future__ import annotations

from typing import Any, Final

import torch
from torch import Tensor

from dionw._keys import MUON_BETA2_KEY, ROW_NORMALIZATION_KEY
from dionw.matrix_update import RowNormalization

__all__ = ["finish", "normalize_rows"]

NORMALIZATION_EPS: Final[float] = 1e-8


def finish(
    orthogonal: Tensor, variance: Tensor, group: dict[str, Any], step: Tensor
) -> tuple[Tensor, Tensor | None]:
    """Scale an orthogonalized ``[blocks, k, cols]`` batch into the float32 step.

    Returns:
        The step and, for NorMuon, the new ``[blocks, k, 1]`` variance (None
        without row normalization).

    Raises:
        RuntimeError: For an unhandled ``RowNormalization``.
    """
    match RowNormalization(group[ROW_NORMALIZATION_KEY]):
        case RowNormalization.NORMUON:
            return normalize_rows(
                orthogonal, variance, float(group[MUON_BETA2_KEY]), step
            )
        case RowNormalization.NONE:
            return orthogonal.float() * step, None
        case _ as unreachable:
            raise RuntimeError(f"Unhandled RowNormalization: {unreachable}")


@torch.compile(dynamic=False, fullgraph=True)
def normalize_rows(
    update: Tensor, variance: Tensor, beta2: float, step: Tensor
) -> tuple[Tensor, Tensor]:
    """NorMuon row normalization.

    Divides each row by the root of an EMA of its mean square, restores the
    block's Frobenius norm and multiplies by ``step`` (a 0-d device tensor, so a
    scheduled LR does not recompile).

    Returns:
        The float32 step and the new variance.
    """
    update = update.float()
    norm = torch.linalg.vector_norm(update, dim=(-2, -1), keepdim=True)
    variance = torch.lerp(variance, update.square().mean(-1, keepdim=True), 1 - beta2)
    normalized = update / (variance.sqrt() + NORMALIZATION_EPS)
    new_norm = torch.linalg.vector_norm(normalized, dim=(-2, -1), keepdim=True)
    restore = norm / new_norm.clamp_min(NORMALIZATION_EPS)
    return normalized * (restore * step), variance
