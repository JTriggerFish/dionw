"""The orthogonalization of Dion's matrix updates (``NewtonSchulz``).

Every variant runs five Polar Express steps (arXiv 2505.16932) on a batch of
float32 matrices, rounding them itself, and returns bfloat16:

- ``POLAR_EXPRESS``: microsoft/dion's iteration with cuBLAS products;
- ``POLAR_EXPRESS_TRITON``: the same with symmetric products in Triton;
- ``GRAM``: Dao-AILab's Gram Newton-Schulz, 1.3-2.7x faster than
  ``POLAR_EXPRESS_TRITON`` on wide blocks (RTX 5090).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from torch import Tensor

from dionw._gram import gram_polar_express
from dionw._polar_express import polar_express, polar_express_triton

__all__ = ["NewtonSchulz", "Orthogonalize", "newton_schulz_fn"]

# A batched orthogonalization: ([N, rows, cols] input, epsilon) -> bf16 output.
Orthogonalize = Callable[[Tensor, float], Tensor]


class NewtonSchulz(Enum):
    """The orthogonalization of matrix updates."""

    POLAR_EXPRESS = "polar_express"
    POLAR_EXPRESS_TRITON = "polar_express_triton"
    GRAM = "gram"


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
        case _ as unreachable:
            raise RuntimeError(f"Unhandled NewtonSchulz: {unreachable}")
