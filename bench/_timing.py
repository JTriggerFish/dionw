"""CUDA-event timing and compile settings shared by the benchmarks."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

WARMUP_CALLS = 5
TIMED_CALLS = 20
# Graphs per compiled function: the benchmarks call Dion's static-shape
# functions directly, outside the optimizer's own raised limits.
RECOMPILE_LIMIT = 64


def raised_recompile_limits() -> AbstractContextManager[None]:
    """Return a context allowing ``RECOMPILE_LIMIT`` graphs per function."""
    return torch._dynamo.config.patch(
        recompile_limit=RECOMPILE_LIMIT, accumulated_recompile_limit=RECOMPILE_LIMIT
    )


def median_ms(
    call: Callable[[], object],
    *,
    warmup: int = WARMUP_CALLS,
    timed: int = TIMED_CALLS,
) -> float:
    """Return the median GPU time of ``call`` in milliseconds.

    Args:
        call: The work to time; compilation and autotuning happen in warm-up.
        warmup: Untimed calls first.
        timed: Calls timed one by one with CUDA events.

    Returns:
        The median of the per-call times.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)
