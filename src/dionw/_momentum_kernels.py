"""Fused Triton passes for Dion's momentum update (any CUDA architecture).

Each kernel reads a parameter's gradient and momentum once:

- ``accumulate_row_l1``: ``M += G`` and each row's l1 norm (the row scores the
  top-k selection reads), for blocks that update a fraction of their rows;
- ``accumulate_copy_decay``: ``U = M + G`` (the Newton-Schulz input, float32)
  and ``M = momentum * (M + G)``, for blocks that update every row.

Momentum is float32 and contiguous ``[rows, cols]``; the gradient must be
contiguous in the same layout (callers use torch ops otherwise).
"""

from __future__ import annotations

from typing import Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor

__all__ = ["accumulate_copy_decay", "accumulate_row_l1"]

_ROW_BLOCK = 1024
_ELEMENT_BLOCK = 2048


class _Launch(Protocol):
    """Host launch signature, distinct from the decorated kernel's DSL types."""

    def __call__(self, *arguments: Tensor | int | float, BLOCK: int) -> None:
        """Launch on CUDA."""
        ...


@triton.jit
def _accumulate_row_l1_kernel(
    grad_ptr, momentum_ptr, scores_ptr, cols, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    base = row * cols
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, cols, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < cols
        grad = tl.load(grad_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        momentum = tl.load(momentum_ptr + base + offsets, mask=mask, other=0.0)
        momentum = momentum + grad
        tl.store(momentum_ptr + base + offsets, momentum, mask=mask)
        total += tl.abs(momentum)
    tl.store(scores_ptr + row, tl.sum(total, axis=0))


@triton.jit
def _accumulate_copy_decay_kernel(
    grad_ptr, momentum_ptr, out_ptr, decay, numel, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    momentum = tl.load(momentum_ptr + offsets, mask=mask, other=0.0) + grad
    tl.store(out_ptr + offsets, momentum, mask=mask)
    tl.store(momentum_ptr + offsets, momentum * decay, mask=mask)


def _check(grad: Tensor, momentum: Tensor) -> None:
    if not (grad.is_contiguous() and momentum.is_contiguous()):
        raise ValueError("the fused momentum kernels need contiguous tensors")
    if momentum.dtype != torch.float32 or grad.numel() != momentum.numel():
        raise ValueError(
            "the fused momentum kernels need float32 momentum of the grad's size"
        )


def accumulate_row_l1(
    grad: Tensor, momentum: Tensor, scores: Tensor, cols: int
) -> None:
    """``momentum += grad``; ``scores[i]`` = l1 norm of momentum row ``i``.

    Args:
        grad: The gradient (any float dtype), contiguous.
        momentum: float32 momentum, contiguous, ``rows * cols`` elements.
        scores: float32 output with one element per row (contiguous).
        cols: Row length.
    """
    _check(grad, momentum)
    rows = momentum.numel() // cols
    launch = cast("_Launch", _accumulate_row_l1_kernel[(rows,)])
    launch(grad, momentum, scores, cols, BLOCK=_ROW_BLOCK)


def accumulate_copy_decay(
    grad: Tensor, momentum: Tensor, out: Tensor, decay: float
) -> None:
    """``out = momentum + grad``; ``momentum = decay * (momentum + grad)``.

    Args:
        grad: The gradient (any float dtype), contiguous.
        momentum: float32 momentum, contiguous.
        out: float32 output of the momentum's size, contiguous.
        decay: The momentum decay.
    """
    _check(grad, momentum)
    numel = momentum.numel()
    launch = cast(
        "_Launch", _accumulate_copy_decay_kernel[(triton.cdiv(numel, _ELEMENT_BLOCK),)]
    )
    launch(grad, momentum, out, float(decay), numel, BLOCK=_ELEMENT_BLOCK)
