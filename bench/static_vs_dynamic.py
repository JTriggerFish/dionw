"""Time static against dynamic compilation of Dion's compiled functions.

Static compiles one graph per shape, dynamic one graph for all shapes; both
run Gram Newton-Schulz and the row normalization on DiT-like block shapes.

Usage:
    python -m bench.static_vs_dynamic
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from bench._timing import median_ms, raised_recompile_limits
from dionw import _gram as gram_module
from dionw import _matrix_step as matrix_step

if TYPE_CHECKING:
    from collections.abc import Callable

EPSILON = 1e-8
# (blocks, rows, cols) of DiT-XL-like blocks (28 layers, width 1152): one of the
# Q, K, V projections split into 16 heads of 72 rows, and the row selections
# fraction 0.5 makes on the attention projection and the MLP.
SHAPES: tuple[tuple[int, int, int], ...] = (
    (28 * 16, 72, 1152),
    (28, 576, 1152),
    (28, 2304, 1152),
    (28, 576, 4608),
)


def _compiled(
    dynamic: bool,
) -> tuple[Callable[..., torch.Tensor], Callable[..., object]]:
    """Return Gram Newton-Schulz and the row normalization, compiled afresh."""
    raw_ns = inspect.unwrap(gram_module.gram_polar_express)
    raw_norm = inspect.unwrap(matrix_step._normalize_rows)
    return (
        torch.compile(raw_ns, dynamic=dynamic, fullgraph=True),
        torch.compile(raw_norm, dynamic=dynamic, fullgraph=True),
    )


def shape_ms(dynamic: bool) -> dict[tuple[int, int, int], float]:
    """Return the median time of orthogonalizing and normalizing each shape.

    Args:
        dynamic: One dynamic-shape graph (True) or one static graph per shape.

    Returns:
        Milliseconds per shape, Newton-Schulz and normalization together.
    """
    torch._dynamo.reset()
    gram, normalize = _compiled(dynamic)
    times = {}
    with raised_recompile_limits():
        for shape in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(*shape, device="cuda")
            variance = torch.rand(shape[0], shape[1], 1, device="cuda")
            step = torch.tensor(-1e-3, device="cuda")
            orthogonal = gram(x, EPSILON)
            times[shape] = median_ms(lambda x=x: gram(x, EPSILON)) + median_ms(
                lambda o=orthogonal, v=variance, s=step: normalize(o, v, 0.95, s)
            )
    return times


def main() -> None:
    """Print per-shape and total times for static and dynamic compilation."""
    print(torch.cuda.get_device_name())
    static, dynamic = shape_ms(dynamic=False), shape_ms(dynamic=True)
    print("| Blocks | Static | Dynamic |\n|---|---|---|")
    for shape in SHAPES:
        print(
            f"| {'x'.join(map(str, shape))} | {static[shape]:.2f} ms "
            f"| {dynamic[shape]:.2f} ms |"
        )
    total_static, total_dynamic = sum(static.values()), sum(dynamic.values())
    print(
        f"| Total | {total_static:.2f} ms | {total_dynamic:.2f} ms "
        f"({total_dynamic / total_static:.2f}x) |"
    )


if __name__ == "__main__":
    main()
