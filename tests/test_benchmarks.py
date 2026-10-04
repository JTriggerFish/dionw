"""The README's performance claims, measured (``pytest -m bench``).

Each test times a benchmark from ``bench/`` and checks the claim it supports.
Timings need an otherwise idle GPU; the tests are deselected by default.
"""

from __future__ import annotations

import pytest
import torch

from bench.host_dispatch import host_dispatch_ms
from bench.newton_schulz import WIDE_SHAPES, orthogonalize_ms
from bench.optimizer_step import Variant, step_ms
from bench.static_vs_dynamic import shape_ms
from dionw.newton_schulz import NewtonSchulz

pytestmark = pytest.mark.bench

# The benchmark transformer of the README's performance table (1.34B).
WIDTH, DEPTH, HEADS = 2048, 24, 16
# Compute capabilities of the GPUs Dao-AILab's CuTeDSL Gram kernels support.
CUTEDSL_CAPABILITIES = {(9, 0), (10, 0), (10, 3)}


@pytest.mark.parametrize("shape", WIDE_SHAPES)
def test_gram_is_faster_than_polar_express_on_wide_blocks(
    shape: tuple[int, int, int],
) -> None:
    gram = orthogonalize_ms(NewtonSchulz.GRAM, shape)
    assert gram < orthogonalize_ms(NewtonSchulz.POLAR_EXPRESS_TRITON, shape)
    assert gram < orthogonalize_ms(NewtonSchulz.POLAR_EXPRESS, shape)


def test_static_graphs_are_faster_than_one_dynamic_graph() -> None:
    assert sum(shape_ms(dynamic=False).values()) < sum(shape_ms(dynamic=True).values())


def test_dionw_step_is_faster_than_nordion2_with_the_same_kernels() -> None:
    """Same row selection and Triton Polar Express on both sides."""
    nordion2 = step_ms(Variant.NORDION2, WIDTH, DEPTH, HEADS)
    same_kernels = step_ms(Variant.DIONW_SAME_KERNELS, WIDTH, DEPTH, HEADS)
    gram = step_ms(Variant.DIONW_EVERY_BLOCK, WIDTH, DEPTH, HEADS)
    assert same_kernels < nordion2
    assert gram < same_kernels


@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in CUTEDSL_CAPABILITIES,
    reason="Dao-AILab's CuTeDSL Gram kernels need an H100- or B200-class GPU",
)
def test_dionw_step_is_faster_than_nordion2_with_cutedsl_gram() -> None:
    nordion2 = step_ms(Variant.NORDION2_GRAM, WIDTH, DEPTH, HEADS)
    assert step_ms(Variant.DIONW_EVERY_BLOCK, WIDTH, DEPTH, HEADS) < nordion2


def test_host_dispatch_is_shorter_than_the_gpu_step() -> None:
    """The host enqueues a default step faster than the GPU runs it.

    Without CPU EMA nothing synchronizes, so the GPU never waits on the host.
    """
    host = host_dispatch_ms(WIDTH, DEPTH, HEADS)
    assert host < step_ms(Variant.DIONW_DEFAULT_025, WIDTH, DEPTH, HEADS)
