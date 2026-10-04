"""Newton-Schulz kernels against microsoft/dion's and Dao-AILab's.

The copied Polar Express matches microsoft/dion's, and Gram Newton-Schulz
reaches the same polar factor with the rounding error of Dao-AILab's reference,
including on ill-conditioned inputs where the restart matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from dion import newton_schulz_triton as official_kernels
from dion import polar_express as official_polar_express

from bench._timing import raised_recompile_limits
from bench.precision import ILL_CONDITIONED, Method, Spectrum, rounding_error
from dionw._gram import gram_polar_express
from dionw._polar_express import polar_express, polar_express_triton
from dionw._symmetric_kernels import ns_line_1, ns_line_2
from dionw.newton_schulz import NewtonSchulz, newton_schulz_fn

if TYPE_CHECKING:
    from collections.abc import Iterator

SHAPES = [(3, 64, 256), (3, 256, 64), (3, 128, 128), (96, 160)]


@pytest.fixture(autouse=True)
def _raised_recompile_limits() -> Iterator[None]:
    """Allow one graph per shape: these tests call the compiled functions directly.

    ``Dion`` raises the limits around its own calls; a whole-suite run compiles
    more shapes than torch's default of 8 per function.
    """
    with raised_recompile_limits():
        yield


def _exact_polar(x: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(x.double(), full_matrices=False)
    return (u @ vh).float()


@pytest.mark.parametrize("shape", [(3, 64, 256), (3, 256, 64), (3, 128, 128)])
def test_gram_matches_polar_express(shape: tuple[int, int, int]) -> None:
    """Gram lands as close to the exact polar factor as dion's Polar Express.

    Wide (Gram), tall (transposed) and square (standard) batches are checked in
    their own orientation.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    exact = _exact_polar(x)
    gram = gram_polar_express(x, 1e-7).float()
    reference = official_polar_express.polar_express(x, 1e-7).float()
    assert gram.shape == x.shape
    assert gram.dtype == torch.float32
    gram_error = float((gram - exact).norm() / exact.norm())
    reference_error = float((reference - exact).norm() / exact.norm())
    assert gram_error < 0.2
    assert gram_error < reference_error + 0.02
    assert float((gram - reference).norm() / reference.norm()) < 0.15


@pytest.mark.parametrize("spectrum", list(Spectrum))
def test_gram_rounds_like_the_reference_package(spectrum: Spectrum) -> None:
    """Gram's rounding error matches Dao-AILab's, restart at iteration 2.

    Both run the same float16 iteration; dionw adds a bfloat16 output rounding
    (about 3e-3).
    """
    ours = rounding_error(Method.GRAM_FLOAT32_INPUT, spectrum)
    reference = rounding_error(Method.REFERENCE_GRAM, spectrum)
    assert ours < 1.3 * reference + 3e-3


@pytest.mark.parametrize("spectrum", ILL_CONDITIONED)
def test_gram_restart_bounds_the_rounding_error(spectrum: Spectrum) -> None:
    """On ill-conditioned inputs, the restart keeps Gram's rounding error small.

    Without the restart, rounding errors in the iterated Gram matrix compound
    (the reference without restarts shows it on the same input). With it, Gram
    in float16 rounds less than Polar Express in bfloat16.
    """
    ours = rounding_error(Method.GRAM_FLOAT32_INPUT, spectrum)
    assert ours < 0.08
    assert rounding_error(Method.REFERENCE_GRAM_NO_RESTART, spectrum) > 0.1
    assert ours < rounding_error(Method.POLAR_EXPRESS, spectrum)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", SHAPES)
def test_symmetric_kernels_match_the_official_package(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """ns_line_1 (A A^T) and ns_line_2 (c A A + b A) equal dion's kernels."""
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    torch.testing.assert_close(ns_line_1(x), official_kernels.ns_line_1(x))
    gram = ns_line_1(x)
    torch.testing.assert_close(
        ns_line_2(gram, alpha=0.5, beta=-1.5),
        official_kernels.ns_line_2(gram, alpha=0.5, beta=-1.5),
    )


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES)
def test_polar_express_matches_the_official_package(shape: tuple[int, ...]) -> None:
    """Both copied Polar Express variants equal dion's (same bf16 iteration)."""
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    torch.testing.assert_close(
        polar_express(x, 1e-7), official_polar_express.polar_express(x, 1e-7)
    )
    torch.testing.assert_close(
        polar_express_triton(x, 1e-7),
        official_polar_express.polar_express_triton(x, 1e-7),
    )


@pytest.mark.parametrize("kind", list(NewtonSchulz))
def test_every_kind_dispatches_to_an_orthogonalization(kind: NewtonSchulz) -> None:
    """Each kind returns a bf16 batch close to the exact polar factor."""
    torch.manual_seed(0)
    x = torch.randn(2, 64, 96, device="cuda")
    out = newton_schulz_fn(kind)(x, 1e-7)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
    exact = _exact_polar(x)
    assert float((out.float() - exact).norm() / exact.norm()) < 0.2


@pytest.mark.parametrize("spectrum", ILL_CONDITIONED)
def test_float32_input_keeps_gram_precise_on_ill_conditioned_inputs(
    spectrum: Spectrum,
) -> None:
    """A bfloat16 input, as microsoft/dion passes it, costs Gram precision.

    Where singular values spread over decades, it doubles the rounding error or
    more.
    """
    float32_input = rounding_error(Method.GRAM_FLOAT32_INPUT, spectrum)
    assert rounding_error(Method.GRAM_BFLOAT16_INPUT, spectrum) > 1.8 * float32_input
