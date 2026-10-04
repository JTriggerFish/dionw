"""Triton kernels for symmetric products: ``A A^T`` and ``c A A^T + b A``.

Copied from microsoft/dion (github.com/microsoft/dion, commit 7692479, MIT
License, Copyright (c) Microsoft Corporation) with the same numerics. Each
kernel computes the blocks on one side of the diagonal and mirrors them, half
the work of a general matrix product. They run on any CUDA architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ns_line_1", "ns_line_2"]

# tl.dot input precisions: exact for float32 inputs, TF32 tensor cores otherwise.
_IEEE: Final[str] = "ieee"
_TF32: Final[str] = "tf32"


class _AutotunedLaunch(Protocol):
    """Host launch signature of an autotuned kernel (keyword arguments)."""

    def __call__(self, **arguments: Tensor | int | float | str) -> None:
        """Launch on CUDA."""
        ...


def _autotune_configs() -> list[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


@triton.jit
def _batch_offset(batch_idx, batch_stride):
    return batch_idx.to(tl.int64) * batch_stride


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Map a program ID to (batch, row, col) of the output matrix."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    return batch_idx, pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N


@triton.autotune(
    configs=_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def _ns_line_1_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """C = A @ A.T for A of shape (M, K); C is (M, M)."""
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks the mirrored store covers.
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += _batch_offset(batch_idx, a_stride_b)
    C_ptr += _batch_offset(batch_idx, c_stride_b)

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    output = accumulator.to(C_ptr.dtype.element_ty)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # The same block mirrored across the diagonal.
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


@triton.autotune(
    configs=_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def _ns_line_2_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """C = alpha * A @ A.T + beta * A for a symmetric (M, M) A."""
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += _batch_offset(batch_idx, a_stride_b)
    C_ptr += _batch_offset(batch_idx, c_stride_b)

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # The block of A added to this block of C.
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += a_add * beta
    output = accumulator.to(C_ptr.dtype.element_ty)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def _symmetric_output(a: Tensor, out: Tensor | None) -> Tensor:
    """The ``[..., M, M]`` output of a symmetric product of ``a`` (``[..., M, K]``).

    Raises:
        ValueError: If ``a`` is not 2D/3D or ``out`` has the wrong shape.
    """
    if a.ndim not in (2, 3):
        raise ValueError(f"Input tensor must be 2D or 3D, but got {a.ndim}D tensor.")
    rows = a.size(-2)
    if out is None:
        return torch.empty((*a.shape[:-1], rows), device=a.device, dtype=a.dtype)
    if out.shape[-2:] != (rows, rows):
        raise ValueError(f"Output must be [..., {rows}, {rows}], got {out.shape}")
    return out


def _symmetric_grid(
    batch_size: int, rows: int
) -> Callable[[dict[str, int]], tuple[int]]:
    """Launch grid over the output blocks of every batch matrix."""

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (
            batch_size
            * triton.cdiv(rows, meta["BLOCK_SIZE_M"])
            * triton.cdiv(rows, meta["BLOCK_SIZE_N"]),
        )

    return grid


def ns_line_1(a: Tensor, *, out: Tensor | None = None) -> Tensor:
    """``a @ a.mT`` for a 2D or batched 3D ``a``, computing half the blocks.

    Raises:
        ValueError: On a non-2D/3D input or a wrongly shaped ``out``.
    """
    out = _symmetric_output(a, out)
    rows, cols = a.shape[-2:]
    launch = cast(
        "_AutotunedLaunch",
        _ns_line_1_kernel[_symmetric_grid(a.size(0) if a.ndim == 3 else 1, rows)],
    )
    launch(
        A_ptr=a,
        C_ptr=out,
        M=rows,
        K=cols,
        a_stride_b=a.stride(0) if a.ndim == 3 else 0,
        a_stride_r=a.stride(-2),
        a_stride_c=a.stride(-1),
        c_stride_b=out.stride(0) if out.ndim == 3 else 0,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        INPUT_PRECISION=_IEEE if a.dtype == torch.float32 else _TF32,
    )
    return out


def ns_line_2(
    a: Tensor, alpha: float, beta: float, *, out: Tensor | None = None
) -> Tensor:
    """``alpha * a @ a.mT + beta * a`` for a symmetric square (batched) ``a``.

    Raises:
        ValueError: On a non-square or non-2D/3D input, or a wrongly shaped ``out``.
    """
    rows, cols = a.shape[-2:]
    if rows != cols:
        raise ValueError(
            f"Input must be symmetric square matrix, but got shape {a.shape}"
        )
    out = _symmetric_output(a, out)
    launch = cast(
        "_AutotunedLaunch",
        _ns_line_2_kernel[_symmetric_grid(a.size(0) if a.ndim == 3 else 1, rows)],
    )
    launch(
        A_ptr=a,
        C_ptr=out,
        M=rows,
        a_stride_b=a.stride(0) if a.ndim == 3 else 0,
        a_stride_r=a.stride(-2),
        a_stride_c=a.stride(-1),
        c_stride_b=out.stride(0) if out.ndim == 3 else 0,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
        INPUT_PRECISION=_IEEE if a.dtype == torch.float32 else _TF32,
    )
    return out
