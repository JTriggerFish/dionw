"""Time Gram Newton-Schulz against Polar Express on wide blocks.

Each kind orthogonalizes a batch of wide blocks; the shapes are row selections
of DiT-XL-like layers [9] (width 1152) and a 1024 x 8192 block.

Usage:
    python -m bench.newton_schulz
"""

from __future__ import annotations

import torch

from bench._timing import median_ms, raised_recompile_limits
from dionw.newton_schulz import NewtonSchulz, newton_schulz_fn

EPSILON = 1e-8
# (blocks, rows, cols); every block is wide, where Gram iterates on rows x rows.
WIDE_SHAPES: tuple[tuple[int, int, int], ...] = (
    (28, 576, 1152),
    (28, 576, 4608),
    (28, 1152, 4608),
    (16, 1024, 8192),
)


def orthogonalize_ms(kind: NewtonSchulz, shape: tuple[int, int, int]) -> float:
    """Return the median time of one batched orthogonalization of ``kind``.

    Args:
        kind: The Newton-Schulz variant.
        shape: ``(blocks, rows, cols)`` of the float32 input batch.

    Returns:
        Milliseconds per call.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    orthogonalize = newton_schulz_fn(kind)
    with raised_recompile_limits():
        return median_ms(lambda: orthogonalize(x, EPSILON))


def main() -> None:
    """Print one Markdown row per shape, with Gram's speedups."""
    print(torch.cuda.get_device_name())
    print("| Blocks | Gram | Polar Express Triton | Polar Express cuBLAS |")
    print("|---|---|---|---|")
    for shape in WIDE_SHAPES:
        gram = orthogonalize_ms(NewtonSchulz.GRAM, shape)
        triton = orthogonalize_ms(NewtonSchulz.POLAR_EXPRESS_TRITON, shape)
        cublas = orthogonalize_ms(NewtonSchulz.POLAR_EXPRESS, shape)
        print(
            f"| {'x'.join(map(str, shape))} | {gram:.2f} ms "
            f"| {triton:.2f} ms ({triton / gram:.2f}x) "
            f"| {cublas:.2f} ms ({cublas / gram:.2f}x) |"
        )


if __name__ == "__main__":
    main()
