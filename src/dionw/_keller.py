"""Keller Jordan's Muon Newton-Schulz: five steps of one quintic, in bfloat16.

The iteration of Muon (github.com/KellerJordan/Muon): normalize by the
Frobenius norm, then five steps of
``x <- 3.4445 x - 4.7750 A x + 2.0315 A^2 x`` with ``A = x x^T`` on the wide
orientation. It pushes singular values into about [0.7, 1.2] rather than onto 1.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from dionw._polar_express import Coefficients, quintic_cublas, quintic_triton

__all__ = ["KELLER_COEFFICIENTS", "keller", "keller_triton"]

KELLER_COEFFICIENTS: Final[Coefficients] = ((3.4445, -4.7750, 2.0315),) * 5
# Muon divides by the Frobenius norm itself (no safety factor).
KELLER_NORM_SAFETY: Final[float] = 1.0


@torch.compile(dynamic=False, fullgraph=True)
def keller(matrix: Tensor, epsilon: float) -> Tensor:
    """Keller's Newton-Schulz with cuBLAS products (``quintic_cublas``)."""
    return quintic_cublas(matrix, epsilon, KELLER_COEFFICIENTS, KELLER_NORM_SAFETY)


@torch.compile(dynamic=False, fullgraph=True)
def keller_triton(matrix: Tensor, epsilon: float) -> Tensor:
    """Keller's Newton-Schulz with symmetric Triton products (``quintic_triton``)."""
    return quintic_triton(matrix, epsilon, KELLER_COEFFICIENTS, KELLER_NORM_SAFETY)
