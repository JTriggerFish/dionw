"""Polar Express (arXiv 2505.16932): five polynomial steps to the polar factor.

Copied from microsoft/dion (github.com/microsoft/dion, commit 7692479, MIT
License, Copyright (c) Microsoft Corporation) with the same numerics, in two
variants: cuBLAS products, and symmetric products in Triton.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

from dionw._symmetric_kernels import ns_line_1, ns_line_2

__all__ = ["POLAR_EXPRESS_COEFFICIENTS", "polar_express", "polar_express_triton"]

# Polar Express coefficients (num_iters=5, safety_factor=2e-2, cushion=2); the
# 1.02 safety factor is baked into all but the last polynomial.
POLAR_EXPRESS_COEFFICIENTS: Final[tuple[tuple[float, float, float], ...]] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
# Spectral-norm bound of the normalized input, matching the coefficients.
POLAR_EXPRESS_NORM_SAFETY: Final[float] = 1.02


@torch.compile(dynamic=False, fullgraph=True)
def polar_express(matrix: Tensor, epsilon: float) -> Tensor:
    """Polar Express in bfloat16 with cuBLAS products (``[..., rows, cols]``).

    Args:
        matrix: The (batched) matrix, any float dtype.
        epsilon: Added to the scaled Frobenius norm before normalizing.

    Returns:
        The orthogonalized input in bfloat16.
    """
    x = matrix.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * POLAR_EXPRESS_NORM_SAFETY + epsilon)
    if matrix.size(-2) > matrix.size(-1):
        # Tall: the small cols x cols product, multiplied from the right.
        for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
            gram = x.mT @ x
            poly = b * gram + c * (gram @ gram)
            x = a * x + x @ poly
    else:
        # Wide: the small rows x rows product, multiplied from the left.
        for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
            gram = x @ x.mT
            poly = b * gram + c * (gram @ gram)
            x = a * x + poly @ x
    return x


@torch.compile(dynamic=False, fullgraph=True)
def polar_express_triton(matrix: Tensor, epsilon: float) -> Tensor:
    """Polar Express in bfloat16 with the symmetric products in Triton.

    Args:
        matrix: The (batched) matrix, any float dtype.
        epsilon: Added to the scaled Frobenius norm before normalizing.

    Returns:
        The orthogonalized input in bfloat16.
    """
    tall = matrix.size(-2) > matrix.size(-1)
    x = matrix.to(dtype=torch.bfloat16)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * POLAR_EXPRESS_NORM_SAFETY + epsilon)

    x = x.contiguous()
    gram = torch.empty((*x.shape[:-1], x.size(-2)), device=x.device, dtype=x.dtype)
    poly = torch.empty_like(gram)
    out = torch.empty_like(x)
    line_3 = torch.baddbmm if x.ndim > 2 else torch.addmm
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        ns_line_1(x, out=gram)
        ns_line_2(gram, alpha=c, beta=b, out=poly)
        line_3(x, poly, x, beta=a, out=out)
        x, out = out, x
    return x.mT if tall else x
