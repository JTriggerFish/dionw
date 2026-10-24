"""The orthogonalization of Dion's matrix updates (``NewtonSchulz``).

Every variant runs five quintic steps on a batch of float32 matrices, rounding
them itself, and returns bfloat16:

- ``POLAR_EXPRESS``: microsoft/dion's Polar Express iteration (arXiv
  2505.16932) with cuBLAS products;
- ``POLAR_EXPRESS_TRITON``: the same with symmetric products in Triton;
- ``GRAM``: Dao-AILab's Gram Newton-Schulz, 1.3-2.7x faster than
  ``POLAR_EXPRESS_TRITON`` on wide blocks (RTX 5090);
- ``KELLER``: Keller Jordan's Muon iteration (one quintic, singular values
  pushed into about [0.7, 1.2], not onto 1) with cuBLAS products;
- ``KELLER_TRITON``: the same with symmetric products in Triton.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from torch import Tensor

from dionw._gram import gram_polar_express
from dionw._keller import keller, keller_triton
from dionw._polar_express import polar_express, polar_express_triton

__all__ = ["NewtonSchulz", "Orthogonalize", "newton_schulz_fn"]

# A batched orthogonalization: ([N, rows, cols] input, epsilon) -> bf16 output.
Orthogonalize = Callable[[Tensor, float], Tensor]


class NewtonSchulz(Enum):
    """The orthogonalization of matrix updates."""

    POLAR_EXPRESS = "polar_express"
    POLAR_EXPRESS_TRITON = "polar_express_triton"
    GRAM = "gram"
    KELLER = "keller"
    KELLER_TRITON = "keller_triton"


def newton_schulz_fn(kind: NewtonSchulz) -> Orthogonalize:
    """Return the orthogonalization for ``kind``.

    Args:
        kind: The variant.

    Returns:
        A function of a ``[N, rows, cols]`` batch and the epsilon added to its
        Frobenius norm, returning the bfloat16 polar factors.

    Raises:
        RuntimeError: For an unhandled kind.
    """
    match kind:
        case NewtonSchulz.POLAR_EXPRESS:
            return polar_express
        case NewtonSchulz.POLAR_EXPRESS_TRITON:
            return polar_express_triton
        case NewtonSchulz.GRAM:
            return gram_polar_express
        case NewtonSchulz.KELLER:
            return keller
        case NewtonSchulz.KELLER_TRITON:
            return keller_triton
        case _ as unreachable:
            raise RuntimeError(f"Unhandled NewtonSchulz: {unreachable}")
