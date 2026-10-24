"""Plain Muon: parity with microsoft/dion's Muon, Keller's iteration, options."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import pytest
import torch
from dion import Muon
from torch import Tensor, nn

import dionw._buckets
from bench._timing import raised_recompile_limits
from dionw import (
    FRACTION_KEY,
    ROUTE_KEY,
    Dion,
    MatrixUpdate,
    MomentumForm,
    NewtonSchulz,
    RouteKind,
    RowNormalization,
)
from dionw._keller import KELLER_COEFFICIENTS, keller, keller_triton
from dionw._keys import MOMENTUM_FORM_KEY, ROW_NORMALIZATION_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

LR = 1e-3
SHAPES = [(3, 64, 256), (3, 256, 64), (3, 128, 128), (96, 160)]


@pytest.fixture(autouse=True)
def _raised_recompile_limits() -> Iterator[None]:
    """Allow one graph per shape for the directly called compiled functions."""
    with raised_recompile_limits():
        yield


def _muon(
    params: Sequence[Tensor],
    fraction: float,
    *,
    newton_schulz: NewtonSchulz,
    momentum_form: MomentumForm,
    weight_decay: float,
) -> Dion:
    """One matrix group with Muon's options, no EMA."""
    return Dion(
        [
            {
                "params": list(params),
                ROUTE_KEY: RouteKind.MATRIX.value,
                FRACTION_KEY: fraction,
            }
        ],
        lr=LR,
        momentum=0.95,
        muon_beta2=0.95,
        betas=(0.9, 0.99),
        weight_decay=weight_decay,
        eps=1e-8,
        matrix_update=MatrixUpdate(
            newton_schulz=newton_schulz,
            row_normalization=RowNormalization.NONE,
            momentum_form=momentum_form,
        ),
        ema=None,
    )


def _exact_polar(x: Tensor, epsilon: float) -> Tensor:
    """Exact polar factor: removes bf16 Newton-Schulz rounding from parity checks."""
    del epsilon
    u, _, vh = torch.linalg.svd(x.double(), full_matrices=False)
    return (u @ vh).to(torch.bfloat16)


def _exact_polar_of_bfloat16(x: Tensor, epsilon: float) -> Tensor:
    """Exact polar factor of ``x`` rounded to bfloat16, as microsoft/dion passes it."""
    return _exact_polar(x.bfloat16(), epsilon)


def _reference_keller(x: Tensor, epsilon: float) -> Tensor:
    """Keller Jordan's Muon iteration as its reference writes it (per matrix)."""
    a, b, c = KELLER_COEFFICIENTS[0]
    out = []
    for matrix in x.reshape(-1, *x.shape[-2:]):
        y = matrix.bfloat16()
        tall = y.shape[0] > y.shape[1]
        if tall:
            y = y.T
        y = y / (y.norm() + epsilon)
        for _ in KELLER_COEFFICIENTS:
            gram = y @ y.T
            y = a * y + (b * gram + c * gram @ gram) @ y
        out.append(y.T if tall else y)
    return torch.stack(out).reshape(x.shape)


def _reference_muon_step(
    param: Tensor, buffer: Tensor, *, nesterov: bool, weight_decay: float
) -> None:
    """One step of Keller Jordan's Muon, scaled in float32."""
    grad = param.grad
    assert grad is not None
    buffer.mul_(0.95).add_(grad)
    update = grad + 0.95 * buffer if nesterov else buffer
    orthogonal = _exact_polar_of_bfloat16(update, 1e-8)
    scale = 0.2 * math.sqrt(max(param.shape))
    param.data.mul_(1 - LR * weight_decay)
    param.data.add_(orthogonal.float(), alpha=-LR * scale)


def _parity_pair(
    shape: tuple[int, int], momentum_form: MomentumForm, weight_decay: float
) -> tuple[nn.Parameter, nn.Parameter, Dion]:
    """Two equal parameters, the first under dionw's Muon options."""
    torch.manual_seed(0)
    start = torch.randn(*shape, device="cuda") * 0.02
    ours_param = nn.Parameter(start.clone())
    ours = _muon(
        [ours_param],
        1.0,
        newton_schulz=NewtonSchulz.KELLER,
        momentum_form=momentum_form,
        weight_decay=weight_decay,
    )
    return ours_param, nn.Parameter(start.clone()), ours


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("momentum_form", list(MomentumForm))
@pytest.mark.parametrize("shape", [(64, 128), (256, 64), (128, 128)])
def test_matches_the_reference_muon_step(
    shape: tuple[int, int],
    momentum_form: MomentumForm,
    weight_decay: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same steps as Keller Jordan's Muon, to float32 rounding."""
    monkeypatch.setattr(
        dionw._buckets, "newton_schulz_fn", lambda _kind: _exact_polar_of_bfloat16
    )
    ours_param, ref_param, ours = _parity_pair(shape, momentum_form, weight_decay)
    buffer = torch.zeros_like(ref_param)
    nesterov = momentum_form is MomentumForm.NESTEROV
    for _ in range(4):
        grad = torch.randn(*shape, device="cuda")
        ours_param.grad, ref_param.grad = grad.clone(), grad.clone()
        ours.step()
        _reference_muon_step(
            ref_param, buffer, nesterov=nesterov, weight_decay=weight_decay
        )
    torch.testing.assert_close(ours_param, ref_param, rtol=0, atol=1e-7)


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("momentum_form", list(MomentumForm))
@pytest.mark.parametrize("shape", [(64, 128), (256, 64), (128, 128)])
def test_matches_microsoft_dion_muon(
    shape: tuple[int, int],
    momentum_form: MomentumForm,
    weight_decay: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same steps as dion.Muon(adjust_lr="rms_norm"), to its bfloat16 scaling.

    dion.Muon multiplies the bfloat16 polar factor by the learning rate in
    bfloat16; dionw scales it in float32. Each step
    then differs by at most one bfloat16 rounding of the step size (2**-8 of
    ``lr * 0.2 * sqrt(max(rows, cols))``, the entries being at most 1).
    """
    monkeypatch.setattr(
        dionw._buckets, "newton_schulz_fn", lambda _kind: _exact_polar_of_bfloat16
    )
    ours_param, ref_param, ours = _parity_pair(shape, momentum_form, weight_decay)
    reference = Muon(
        [{"params": [ref_param]}],
        lr=LR,
        mu=0.95,
        weight_decay=weight_decay,
        epsilon=1e-8,
        nesterov=momentum_form is MomentumForm.NESTEROV,
        adjust_lr="rms_norm",
        newton_schulz_func=_exact_polar,
    )
    steps = 4
    for _ in range(steps):
        grad = torch.randn(*shape, device="cuda")
        ours_param.grad, ref_param.grad = grad.clone(), grad.clone()
        ours.step()
        reference.step()
    rounding = steps * 2**-8 * LR * 0.2 * math.sqrt(max(shape))
    torch.testing.assert_close(ours_param, ref_param, rtol=0, atol=rounding)


def _float64_keller(x: Tensor) -> Tensor:
    """Keller's iteration in float64: the target without rounding."""
    a, b, c = KELLER_COEFFICIENTS[0]
    out = []
    for matrix in x.reshape(-1, *x.shape[-2:]):
        y = matrix.double()
        tall = y.shape[0] > y.shape[1]
        if tall:
            y = y.T
        y = y / y.norm()
        for _ in KELLER_COEFFICIENTS:
            gram = y @ y.T
            y = a * y + (b * gram + c * (gram @ gram)) @ y
        out.append(y.T if tall else y)
    return torch.stack(out).reshape(x.shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_keller_rounds_no_worse_than_the_reference(shape: tuple[int, ...]) -> None:
    """Both Keller variants round no worse than the per-matrix bf16 reference.

    Rounding is measured against the same iteration in float64. Keller's quintic does not converge (singular values keep moving inside its
    band), so bf16 rounding is not damped: a one-ulp change of the input moves
    the reference's own output by about 1.5%. Implementations therefore agree
    only to that rounding, not elementwise.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    exact = _float64_keller(x)

    def error(out: Tensor) -> float:
        return float((out.double() - exact).norm() / exact.norm())

    reference_error = error(_reference_keller(x, 1e-8))
    for variant in (keller, keller_triton):
        out = variant(x, 1e-8)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape
        assert error(out) < reference_error + 5e-3


def test_keller_lands_singular_values_in_its_band() -> None:
    """Keller's iteration puts singular values in about [0.7, 1.2], not onto 1."""
    torch.manual_seed(0)
    x = torch.randn(2, 96, 160, device="cuda")
    for variant in (keller, keller_triton):
        singular = torch.linalg.svdvals(variant(x, 1e-8).double())
        assert float(singular.min()) > 0.6
        assert float(singular.max()) < 1.25


def test_channels_last_convolution_matches_contiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The torch Nesterov path (channels-last grads) equals the fused kernel's.

    The exact polar factor stands in for Keller's iteration, which would
    amplify the paths' float32 rounding differences (see above).
    """
    monkeypatch.setattr(
        dionw._buckets, "newton_schulz_fn", lambda _kind: _exact_polar_of_bfloat16
    )
    torch.manual_seed(0)
    start = torch.randn(32, 16, 3, 3, device="cuda") * 0.02
    contiguous = nn.Parameter(start.clone())
    channels_last = nn.Parameter(start.clone().to(memory_format=torch.channels_last))
    optimizers = [
        _muon(
            [param],
            1.0,
            newton_schulz=NewtonSchulz.KELLER,
            momentum_form=MomentumForm.NESTEROV,
            weight_decay=0.1,
        )
        for param in (contiguous, channels_last)
    ]
    for _ in range(3):
        grad = torch.randn_like(start)
        contiguous.grad = grad.clone()
        channels_last.grad = grad.clone().to(memory_format=torch.channels_last)
        for optimizer in optimizers:
            optimizer.step()
    torch.testing.assert_close(channels_last, contiguous, rtol=0, atol=1e-7)


def test_nesterov_with_row_selection_raises() -> None:
    """Nesterov momentum needs every row per step."""
    weight = nn.Parameter(torch.zeros(64, 64, device="cuda"))
    with pytest.raises(ValueError, match="Nesterov momentum needs every row"):
        _muon(
            [weight],
            0.5,
            newton_schulz=NewtonSchulz.KELLER,
            momentum_form=MomentumForm.NESTEROV,
            weight_decay=0.0,
        )


def _normuon(weight: nn.Parameter) -> Dion:
    """One full-row NorMuon heavy-ball group (the update before the options)."""
    return Dion(
        [{"params": [weight], ROUTE_KEY: RouteKind.MATRIX.value, FRACTION_KEY: 1.0}],
        lr=LR,
        momentum=0.95,
        muon_beta2=0.95,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        eps=1e-8,
        matrix_update=MatrixUpdate(
            newton_schulz=NewtonSchulz.GRAM,
            row_normalization=RowNormalization.NORMUON,
            momentum_form=MomentumForm.HEAVY_BALL,
        ),
        ema=None,
    )


def test_state_saved_before_the_options_loads_only_as_normuon() -> None:
    """A checkpoint without the option keys is NorMuon with heavy-ball momentum.

    It resumes into that optimizer and is refused by any other.
    """
    weight = nn.Parameter(torch.randn(32, 32, device="cuda"))
    optimizer = _normuon(weight)
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    for group in state["param_groups"]:
        del group[ROW_NORMALIZATION_KEY], group[MOMENTUM_FORM_KEY]
    resumed = _normuon(weight)
    resumed.load_state_dict(copy.deepcopy(state))
    assert resumed.param_groups[0][ROW_NORMALIZATION_KEY] == "normuon"
    resumed.step()
    muon = _muon(
        [weight],
        1.0,
        newton_schulz=NewtonSchulz.GRAM,
        momentum_form=MomentumForm.HEAVY_BALL,
        weight_decay=0.0,
    )
    with pytest.raises(ValueError, match="'row_normalization'"):
        muon.load_state_dict(state)
