"""Gram Newton-Schulz: Polar Express on the small Gram matrix of wide blocks.

Adapted from Dao-AILab's gram-newton-schulz (github.com/Dao-AILab/
gram-newton-schulz, by Jack Zhang, Noah Amsel, Berlin Chen and Tri Dao, declared
MIT). For a wide ``X`` (rows <= cols) it iterates on ``R = X X^T`` and a
polynomial ``Q`` in it, restarting from ``X <- Q X`` at iteration 2 for
stability, and returns ``Q X``: the Polar Express iteration with fewer FLOPs
when cols > rows. Tall inputs are transposed; square ones run the standard
iteration. The iterations run in float16 after a float32 normalization, as in
the reference, on the portable Triton kernels (the reference's CuTeDSL kernels,
sm90/sm100 only, are not used).
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from dionw._symmetric_kernels import ns_line_1, ns_line_2

__all__ = ["GRAM_COEFFICIENTS", "gram_polar_express"]

_GRAM_SAFETY: Final[float] = 1.05
_GRAM_UNMODIFIED: Final[tuple[tuple[float, float, float], ...]] = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
# Polar Express (arXiv 2505.16932) with the Gram reference's safety factor.
GRAM_COEFFICIENTS: Final[tuple[tuple[float, float, float], ...]] = tuple(
    (a / _GRAM_SAFETY, b / _GRAM_SAFETY**3, c / _GRAM_SAFETY**5)
    for a, b, c in _GRAM_UNMODIFIED
)
GRAM_RESTART_ITERATION: Final[int] = 2


def _standard(x: Tensor) -> Tensor:
    """Standard Polar Express on wide ``[N, rows, cols]`` float16 input."""
    for a, b, c in GRAM_COEFFICIENTS:
        gram = ns_line_1(x)
        poly = ns_line_2(gram, alpha=c, beta=b)
        x = torch.baddbmm(x, poly, x, beta=a)
    return x


def _gram(x: Tensor) -> Tensor:
    """Gram Newton-Schulz on wide ``[N, rows, cols]`` float16 input."""
    gram = ns_line_1(x)
    eye = torch.eye(x.size(-2), device=x.device, dtype=x.dtype).expand_as(gram)
    q = eye
    last = len(GRAM_COEFFICIENTS) - 1
    for i, (a, b, c) in enumerate(GRAM_COEFFICIENTS):
        if i == GRAM_RESTART_ITERATION:
            x = torch.bmm(q, x)
            gram = ns_line_1(x)
        z = ns_line_2(gram, alpha=c, beta=b)
        q = (
            z + a * eye
            if i in (0, GRAM_RESTART_ITERATION)
            else torch.baddbmm(q, q, z, beta=a)
        )
        if i < last and i + 1 != GRAM_RESTART_ITERATION:
            rz = torch.baddbmm(gram, gram, z, beta=a)
            gram = torch.baddbmm(rz, z, rz, beta=a)
    return torch.bmm(q, x)


@torch.compile(dynamic=False, fullgraph=True)
def gram_polar_express(matrix: Tensor, epsilon: float) -> Tensor:
    """Polar factor of each ``[rows, cols]`` matrix of a ``[N, rows, cols]`` batch.

    Args:
        matrix: The batch (any float dtype).
        epsilon: Added to the Frobenius norm before normalizing.

    Returns:
        The orthogonalized batch in bfloat16.
    """
    tall = matrix.size(-2) > matrix.size(-1)
    x = matrix.mT if tall else matrix
    x = x.float()
    x = (x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + epsilon)).half()
    x = x.contiguous()
    x = _gram(x) if x.size(-1) > x.size(-2) else _standard(x)
    return (x.mT if tall else x).bfloat16()
