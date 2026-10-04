"""Dion's fused momentum kernels against plain torch."""

from __future__ import annotations

import torch

from dionw._momentum_kernels import accumulate_copy_decay, accumulate_row_l1


def test_accumulate_row_l1_matches_torch() -> None:
    torch.manual_seed(0)
    rows, cols = 37, 3000  # cols spans several kernel blocks with a remainder
    grad = torch.randn(rows, cols, device="cuda")
    momentum = torch.randn(rows, cols, device="cuda")
    expected = momentum + grad
    scores = torch.empty(rows, device="cuda")
    accumulate_row_l1(grad, momentum, scores, cols)
    torch.testing.assert_close(momentum, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected.abs().sum(-1), rtol=1e-5, atol=1e-3)


def test_accumulate_copy_decay_matches_torch() -> None:
    torch.manual_seed(0)
    grad = torch.randn(5001, device="cuda")
    momentum = torch.randn(5001, device="cuda")
    accumulated = momentum + grad
    out = torch.empty(5001, device="cuda")
    accumulate_copy_decay(grad, momentum, out, 0.95)
    torch.testing.assert_close(out, accumulated, rtol=0, atol=0)
    torch.testing.assert_close(momentum, accumulated * 0.95, rtol=1e-6, atol=1e-7)
