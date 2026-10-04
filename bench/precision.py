"""Rounding error of each Newton-Schulz method on well- and ill-conditioned inputs.

A method's rounding error is its relative Frobenius distance to the same
polynomials evaluated in float64 on the float32 input. Momentum matrices are
often close to low rank, which is where errors in an iterated Gram matrix
compound; the spectra below cover that case.

Usage:
    python -m bench.precision
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
from gram_newton_schulz import GramNewtonSchulz

from bench._timing import raised_recompile_limits
from dionw._gram import GRAM_COEFFICIENTS, gram_polar_express
from dionw._polar_express import (
    POLAR_EXPRESS_COEFFICIENTS,
    POLAR_EXPRESS_NORM_SAFETY,
    polar_express,
    polar_express_triton,
)

if TYPE_CHECKING:
    from torch import Tensor

EPSILON = 1e-7
# Wide, so Gram iterates on the 256 x 256 Gram matrix.
SHAPE = (256, 1024)
LOW_RANK = 8


class Spectrum(Enum):
    """Singular values of a test matrix, largest first."""

    GAUSSIAN = "Gaussian entries"
    GEOMETRIC_2 = "geometric decay over 2 decades"
    GEOMETRIC_4 = "geometric decay over 4 decades"
    GEOMETRIC_6 = "geometric decay over 6 decades"
    LOW_RANK_3 = "rank 8 at 1, the rest at 1e-3"
    LOW_RANK_4 = "rank 8 at 1, the rest at 1e-4"


ILL_CONDITIONED: tuple[Spectrum, ...] = (
    Spectrum.GEOMETRIC_4,
    Spectrum.GEOMETRIC_6,
    Spectrum.LOW_RANK_3,
    Spectrum.LOW_RANK_4,
)


class Method(Enum):
    """The orthogonalizations compared."""

    GRAM_FLOAT32_INPUT = "dionw Gram (float16 iterations), float32 input"
    GRAM_BFLOAT16_INPUT = "dionw Gram (float16 iterations), bfloat16 input"
    POLAR_EXPRESS = "Polar Express (bfloat16), cuBLAS products"
    POLAR_EXPRESS_TRITON = "Polar Express (bfloat16), Triton products"
    REFERENCE_GRAM = "Dao-AILab Gram, restart at iteration 2"
    REFERENCE_GRAM_NO_RESTART = "Dao-AILab Gram, no restart"


def _singular_values(spectrum: Spectrum, count: int) -> Tensor:
    """Return ``count`` float64 singular values of ``spectrum``, largest first."""
    match spectrum:
        case Spectrum.GEOMETRIC_2 | Spectrum.GEOMETRIC_4 | Spectrum.GEOMETRIC_6:
            decades = {
                Spectrum.GEOMETRIC_2: 2,
                Spectrum.GEOMETRIC_4: 4,
                Spectrum.GEOMETRIC_6: 6,
            }[spectrum]
            return torch.logspace(0, -decades, count, device="cuda").double()
        case Spectrum.LOW_RANK_3 | Spectrum.LOW_RANK_4:
            floor = 1e-3 if spectrum is Spectrum.LOW_RANK_3 else 1e-4
            values = torch.full((count,), floor, device="cuda", dtype=torch.float64)
            values[:LOW_RANK] = 1.0
            return values
        case _:
            raise ValueError(f"{spectrum} has no prescribed singular values")


def spectrum_matrix(spectrum: Spectrum) -> Tensor:
    """Return a seeded ``[1, *SHAPE]`` float32 matrix with ``spectrum``."""
    rows, cols = SHAPE
    generator = torch.Generator(device="cuda").manual_seed(0)
    if spectrum is Spectrum.GAUSSIAN:
        return torch.randn(1, rows, cols, device="cuda", generator=generator)
    left, _ = torch.linalg.qr(
        torch.randn(rows, rows, device="cuda", dtype=torch.float64, generator=generator)
    )
    right, _ = torch.linalg.qr(
        torch.randn(cols, rows, device="cuda", dtype=torch.float64, generator=generator)
    )
    matrix = (left * _singular_values(spectrum, rows)) @ right.mT
    return matrix.float()[None]


def polynomial_float64(
    x: Tensor,
    coefficients: tuple[tuple[float, float, float], ...],
    norm_safety: float,
) -> Tensor:
    """Evaluate the Newton-Schulz polynomials of wide ``x`` in float64.

    This is what a method computes without rounding.
    """
    x = x.double()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * norm_safety + EPSILON)
    for a, b, c in coefficients:
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x


def relative_error(value: Tensor, target: Tensor) -> float:
    """Return ``|value - target| / |target|`` in the Frobenius norm."""
    return float((value.double() - target).norm() / target.norm())


def _reference_gram(x: Tensor, restarts: list[int]) -> Tensor:
    """Dao-AILab's Gram Newton-Schulz with its PyTorch backend (any GPU)."""
    reference = GramNewtonSchulz(
        ns_epsilon=EPSILON,
        ns_use_kernels=False,
        use_gram_newton_schulz=True,
        gram_newton_schulz_reset_iterations=restarts,
        compile_kwargs=None,
    )
    return reference(x)


def _orthogonalize(method: Method, x: Tensor) -> Tensor:
    """Return ``method``'s orthogonalization of float32 ``x``."""
    match method:
        case Method.GRAM_FLOAT32_INPUT:
            return gram_polar_express(x, EPSILON)
        case Method.GRAM_BFLOAT16_INPUT:
            return gram_polar_express(x.bfloat16(), EPSILON)
        case Method.POLAR_EXPRESS:
            return polar_express(x, EPSILON)
        case Method.POLAR_EXPRESS_TRITON:
            return polar_express_triton(x, EPSILON)
        case Method.REFERENCE_GRAM:
            return _reference_gram(x, [2])
        case Method.REFERENCE_GRAM_NO_RESTART:
            return _reference_gram(x, [])
        case _ as unreachable:
            raise RuntimeError(f"Unhandled method: {unreachable}")


def rounding_error(method: Method, spectrum: Spectrum) -> float:
    """Return ``method``'s rounding error on the matrix of ``spectrum``."""
    x = spectrum_matrix(spectrum)
    match method:
        case Method.POLAR_EXPRESS | Method.POLAR_EXPRESS_TRITON:
            target = polynomial_float64(
                x, POLAR_EXPRESS_COEFFICIENTS, POLAR_EXPRESS_NORM_SAFETY
            )
        case _:
            target = polynomial_float64(x, GRAM_COEFFICIENTS, 1.0)
    with raised_recompile_limits():
        return relative_error(_orthogonalize(method, x), target)


def main() -> None:
    """Print the rounding error of every method on every spectrum."""
    print(torch.cuda.get_device_name(), f"{SHAPE[0]} x {SHAPE[1]} input")
    print("| Method | " + " | ".join(s.value for s in Spectrum) + " |")
    print("|---" * (len(Spectrum) + 1) + "|")
    for method in Method:
        errors = (f"{rounding_error(method, s):.3f}" for s in Spectrum)
        print(f"| {method.value} | " + " | ".join(errors) + " |")


if __name__ == "__main__":
    main()
