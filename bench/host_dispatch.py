"""Measure the host time of one ``optimizer.step()`` while the GPU is busy.

The GPU is kept busy before each step, so the host never waits on it and the
measured time is the Python and launch overhead alone. The model is the
benchmark transformer of ``bench.optimizer_step``, with the default routing.

Usage:
    python -m bench.host_dispatch [--width 2048] [--depth 24] [--heads 16]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import dionw
from bench.optimizer_step import BETAS, LR, transformer

WARMUP_STEPS = 5
TIMED_STEPS = 20
# Busy-wait cycles queued before each step: longer than any step's dispatch.
GPU_BUSY_CYCLES = 5_000_000_000


def host_dispatch_ms(width: int, depth: int, heads: int) -> float:
    """Return the median host time of a default ``Dion`` step in ms."""
    torch.manual_seed(0)
    model = transformer(width, depth, heads)
    groups, _ = dionw.param_groups(
        model, fraction=0.25, selection_min_dim=1024, min_matrix_dim=8
    )
    optimizer = dionw.Dion(
        groups,
        lr=LR,
        betas=BETAS,
        weight_decay=0.0,
        newton_schulz=dionw.NewtonSchulz.GRAM,
    )
    for _ in range(WARMUP_STEPS):
        optimizer.step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(TIMED_STEPS):
        torch.cuda._sleep(GPU_BUSY_CYCLES)
        start = time.perf_counter()
        optimizer.step()
        samples.append((time.perf_counter() - start) * 1e3)
        torch.cuda.synchronize()
    return statistics.median(samples)


def main() -> None:
    """Print the median host dispatch time."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    args = parser.parse_args()
    elapsed = host_dispatch_ms(args.width, args.depth, args.heads)
    print(f"{torch.cuda.get_device_name()}: host dispatch {elapsed:.1f} ms per step")


if __name__ == "__main__":
    main()
