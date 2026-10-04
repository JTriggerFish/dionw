"""Dion: parity with microsoft/dion, the AdamW path, resume, groups, layouts."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import pytest
import torch
from dion import NorDion2, NorMuon
from torch import Tensor, nn

import dionw._buckets
from dionw import (
    FRACTION_KEY,
    NEWTON_SCHULZ_KEY,
    NUM_HEADS_KEY,
    ROUTE_KEY,
    Dion,
    EMAConfig,
    EMADevice,
    NewtonSchulz,
    Route,
    RouteKind,
    param_groups,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

LR = 1e-3


def _dion(
    params: Sequence[Tensor],
    fraction: float,
    newton_schulz: NewtonSchulz,
    *,
    weight_decay: float = 0.0,
) -> Dion:
    """One matrix group, no EMA."""
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
        newton_schulz=newton_schulz,
        ema=None,
    )


def _exact_polar(x: Tensor, epsilon: float) -> Tensor:
    """Exact polar factor: removes bf16 Newton-Schulz rounding from parity checks."""
    del epsilon
    u, _, vh = torch.linalg.svd(x.double(), full_matrices=False)
    return (u @ vh).to(torch.bfloat16)


def _exact_polar_of_bfloat16(x: Tensor, epsilon: float) -> Tensor:
    """Exact polar factor of ``x`` rounded to bfloat16, as microsoft/dion passes it.

    dionw hands float32 momentum to Newton-Schulz; microsoft/dion rounds it to
    bfloat16 first. Rounding here too leaves every other difference in view.
    """
    return _exact_polar(x.bfloat16(), epsilon)


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize(
    ("shape", "fraction"),
    [((64, 128), 1.0), ((64, 128), 0.5), ((256, 64), 0.5), ((128, 128), 0.25)],
)
def test_matches_microsoft_dion_nordion2(
    shape: tuple[int, int],
    fraction: float,
    weight_decay: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same steps as dion.NorDion2 given its LR times the Frobenius compensation."""
    monkeypatch.setattr(
        dionw._buckets, "newton_schulz_fn", lambda _kind: _exact_polar_of_bfloat16
    )
    torch.manual_seed(0)
    start = torch.randn(*shape, device="cuda") * 0.02
    ours_param = nn.Parameter(start.clone())
    ref_param = nn.Parameter(start.clone())
    ours = _dion(
        [ours_param], fraction, NewtonSchulz.POLAR_EXPRESS, weight_decay=weight_decay
    )
    rows, cols = shape
    selected = math.ceil(fraction * rows)
    compensation = math.sqrt(min(rows, cols) / min(selected, cols))
    reference = NorDion2(
        [{"params": [ref_param], "lr": LR * compensation}],
        lr=LR,
        fraction=fraction,
        mu=0.95,
        muon_beta2=0.95,
        # NorDion2 decays at its own LR, which carries the compensation.
        weight_decay=weight_decay / compensation,
        epsilon=1e-8,
        adjust_lr="rms_norm",
        newton_schulz_func=_exact_polar,
    )
    for _ in range(4):
        grad = torch.randn(*shape, device="cuda")
        ours_param.grad, ref_param.grad = grad.clone(), grad.clone()
        ours.step()
        reference.step()
    torch.testing.assert_close(ours_param, ref_param, rtol=0, atol=1e-7)


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_flattened_convolution_matches_microsoft_dion_normuon(
    weight_decay: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conv weight is the (out, in * k) matrix NorMuon(flatten=True) uses."""
    monkeypatch.setattr(
        dionw._buckets, "newton_schulz_fn", lambda _kind: _exact_polar_of_bfloat16
    )
    torch.manual_seed(0)
    start = torch.randn(32, 16, 3, device="cuda") * 0.02
    ours_param = nn.Parameter(start.clone())
    ref_param = nn.Parameter(start.clone())
    ours = _dion(
        [ours_param], 1.0, NewtonSchulz.POLAR_EXPRESS, weight_decay=weight_decay
    )
    reference = NorMuon(
        [{"params": [ref_param]}],
        lr=LR,
        mu=0.95,
        muon_beta2=0.95,
        weight_decay=weight_decay,
        epsilon=1e-8,
        adjust_lr="rms_norm",
        flatten=True,
        newton_schulz_func=_exact_polar,
    )
    for _ in range(4):
        grad = torch.randn(*start.shape, device="cuda")
        ours_param.grad, ref_param.grad = grad.clone(), grad.clone()
        ours.step()
        reference.step()
    torch.testing.assert_close(ours_param, ref_param, rtol=0, atol=1e-7)


def test_adamw_route_equals_torch_fused_adamw() -> None:
    """AdamW-routed parameters take exactly torch's AdamW(fused=True) step."""
    torch.manual_seed(0)
    start = torch.randn(64, 8, device="cuda")
    ours = nn.Parameter(start.clone())
    reference = nn.Parameter(start.clone())
    optimizer = Dion(
        [{"params": [ours], ROUTE_KEY: RouteKind.ADAMW.value}],
        lr=1e-2,
        momentum=0.95,
        muon_beta2=0.95,
        betas=(0.9, 0.98),
        weight_decay=0.1,
        eps=1e-8,
        newton_schulz=NewtonSchulz.GRAM,
        ema=None,
    )
    torch_adamw = torch.optim.AdamW(
        [reference], lr=1e-2, betas=(0.9, 0.98), weight_decay=0.1, eps=1e-8, fused=True
    )
    for _ in range(5):
        grad = torch.randn_like(start)
        ours.grad, reference.grad = grad.clone(), grad.clone()
        optimizer.step()
        torch_adamw.step()
    torch.testing.assert_close(ours, reference, rtol=0, atol=0)


class _Net(nn.Module):
    """A small model with every route: matrices, a head split, AdamW vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 64)
        self.qkv = nn.Linear(64, 192)
        self.conv = nn.Conv1d(64, 64, 3, padding=1)
        self.up = nn.Linear(64, 128)
        self.down = nn.Linear(128, 64)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 4)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))

    def dion_routes(self) -> tuple[tuple[Tensor, Route], ...]:
        return (
            (self.qkv.weight, Route(RouteKind.MATRIX, num_heads=12, fraction=None)),
        )

    def forward(self, ids: Tensor) -> Tensor:
        h = self.embed(ids) + self.cls_token
        h = h + self.qkv(h)[..., :64]
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.down(torch.relu(self.up(self.norm(h))))
        return self.head(h)


def _net_optimizer(
    model: nn.Module, *, lr: float, ema: EMAConfig | None, newton_schulz: NewtonSchulz
) -> Dion:
    groups, _ = param_groups(
        model, fraction=0.5, selection_min_dim=32, min_matrix_dim=8
    )
    return Dion(
        groups,
        lr=lr,
        momentum=0.95,
        muon_beta2=0.95,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        eps=1e-8,
        newton_schulz=newton_schulz,
        ema=ema,
    )


def _train(model: nn.Module, optimizer: Dion, steps: int, seed: int) -> list[float]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, 16, (8, 12), device="cuda", generator=generator)
        target = torch.randn(8, 12, 4, device="cuda", generator=generator)
        loss = (model(ids) - target).pow(2).mean()
        losses.append(float(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return losses


def _step_with_random_grads(
    model: nn.Module, optimizer: Dion, steps: int, seed: int
) -> None:
    """Take ``steps`` optimizer steps on seeded random gradients."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(steps):
        for param in model.parameters():
            param.grad = torch.randn(param.shape, device="cuda", generator=generator)
        optimizer.step()


@pytest.mark.parametrize("newton_schulz", list(NewtonSchulz))
def test_training_reduces_the_loss(newton_schulz: NewtonSchulz) -> None:
    torch.manual_seed(0)
    model = _Net().cuda()
    optimizer = _net_optimizer(model, lr=1e-2, ema=None, newton_schulz=newton_schulz)
    generator = torch.Generator(device="cuda").manual_seed(0)
    ids = torch.randint(0, 16, (8, 12), device="cuda", generator=generator)
    target = torch.randn(8, 12, 4, device="cuda", generator=generator)
    losses = []
    for _ in range(30):
        loss = (model(ids) - target).pow(2).mean()
        losses.append(float(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert losses[-1] < 0.75 * losses[0]


def test_zero_learning_rate_freezes_every_parameter() -> None:
    torch.manual_seed(0)
    model = _Net().cuda()
    optimizer = _net_optimizer(
        model, lr=1e-2, ema=None, newton_schulz=NewtonSchulz.GRAM
    )
    _train(model, optimizer, 2, seed=0)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    frozen = copy.deepcopy(model.state_dict())
    _train(model, optimizer, 1, seed=1)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("device", list(EMADevice))
def test_resume_matches_uninterrupted_training(
    device: EMADevice, tmp_path: Path
) -> None:
    """A resumed run equals an uninterrupted one.

    Two steps, save, load into a fresh model and optimizer, two more: equal to
    four uninterrupted steps in parameters, optimizer state and EMA. The steps
    take seeded random gradients: the toy model's backward pass is not
    bit-deterministic on every GPU (its embedding backward accumulates with
    atomics), and the resume path is what this test checks.
    """
    ema = EMAConfig(
        decay=0.9,
        device=device,
        pin_memory=True,
        update_every_n_steps=1,
        start_step=0,
        decay_final=None,
        decay_ramp_steps=None,
    )
    torch.manual_seed(0)
    full = _Net().cuda()
    start = copy.deepcopy(full.state_dict())
    full_optimizer = _net_optimizer(
        full, lr=1e-2, ema=ema, newton_schulz=NewtonSchulz.GRAM
    )
    _step_with_random_grads(full, full_optimizer, 2, seed=0)
    _step_with_random_grads(full, full_optimizer, 2, seed=1)

    first = _Net().cuda()
    first.load_state_dict(start)
    first_optimizer = _net_optimizer(
        first, lr=1e-2, ema=ema, newton_schulz=NewtonSchulz.GRAM
    )
    _step_with_random_grads(first, first_optimizer, 2, seed=0)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": first.state_dict(), "optimizer": first_optimizer.state_dict()}, path
    )

    checkpoint = torch.load(path, weights_only=True)
    resumed = _Net().cuda()
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer = _net_optimizer(
        resumed, lr=1e-2, ema=ema, newton_schulz=NewtonSchulz.GRAM
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    _step_with_random_grads(resumed, resumed_optimizer, 2, seed=1)

    for name, value in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    full_state = full_optimizer.state_dict()
    resumed_state = resumed_optimizer.state_dict()
    assert resumed_state["_ema_step_count"] == full_state["_ema_step_count"] == 4
    for index, entry in full_state["state"].items():
        for key, value in entry.items():
            if isinstance(value, Tensor):
                torch.testing.assert_close(
                    resumed_state["state"][index][key].to(value.device),
                    value,
                    rtol=0,
                    atol=0,
                    msg=f"{index} {key}",
                )


def test_parameters_without_gradients_are_skipped() -> None:
    torch.manual_seed(0)
    layers = [nn.Linear(32, 32, bias=False, device="cuda") for _ in range(3)]
    optimizer = _dion([layer.weight for layer in layers], 0.5, NewtonSchulz.GRAM)
    untouched = layers[1].weight.detach().clone()
    for index in (0, 2):
        layers[index].weight.grad = torch.randn_like(layers[index].weight)
    optimizer.step()
    torch.testing.assert_close(layers[1].weight, untouched, rtol=0, atol=0)
    assert not torch.equal(layers[0].weight, untouched)


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_channels_last_convolution_matches_contiguous(fraction: float) -> None:
    """A channels-last conv weight gets the same step as its contiguous copy."""
    torch.manual_seed(0)
    contiguous = nn.Conv2d(32, 64, 3).cuda()
    channels_last = nn.Conv2d(32, 64, 3).cuda()
    channels_last.weight.data = channels_last.weight.data.contiguous(
        memory_format=torch.channels_last
    )
    channels_last.load_state_dict(contiguous.state_dict())
    assert not channels_last.weight.is_contiguous()
    optimizers = [
        _dion([conv.weight], fraction, NewtonSchulz.GRAM)
        for conv in (contiguous, channels_last)
    ]
    for _ in range(3):
        grad = torch.randn_like(contiguous.weight)
        contiguous.weight.grad = grad.clone()
        channels_last.weight.grad = grad.clone().contiguous(
            memory_format=torch.channels_last
        )
        for optimizer in optimizers:
            optimizer.step()
    torch.testing.assert_close(
        channels_last.weight, contiguous.weight, rtol=0, atol=1e-6
    )


def test_param_group_added_after_a_step_is_stepped() -> None:
    """add_param_group repacks matrix state.

    The new group steps, existing momentum survives, and an invalid group is
    rejected unadded.
    """
    torch.manual_seed(0)
    first = nn.Linear(32, 32, bias=False, device="cuda")
    second = nn.Linear(32, 32, bias=False, device="cuda")
    optimizer = _dion([first.weight], 1.0, NewtonSchulz.GRAM)
    first.weight.grad = torch.randn_like(first.weight)
    optimizer.step()
    momentum = optimizer.state[first.weight]["momentum"].clone()
    with pytest.raises(ValueError, match="fraction"):
        optimizer.add_param_group(
            {
                "params": [second.weight],
                ROUTE_KEY: RouteKind.MATRIX.value,
                FRACTION_KEY: 1.5,
            }
        )
    assert len(optimizer.param_groups) == 1
    optimizer.add_param_group(
        {
            "params": iter([second.weight]),
            ROUTE_KEY: RouteKind.MATRIX.value,
            FRACTION_KEY: 1.0,
        }
    )
    before = second.weight.detach().clone()
    first.weight.grad = None
    second.weight.grad = torch.randn_like(second.weight)
    optimizer.step()
    assert not torch.equal(second.weight, before)
    torch.testing.assert_close(
        optimizer.state[first.weight]["momentum"], momentum, rtol=0, atol=0
    )


def test_invalid_construction_raises() -> None:
    weight = nn.Parameter(torch.zeros(16, 16, device="cuda"))
    with pytest.raises(TypeError, match=r"dionw\.param_groups"):
        Dion([weight], lr=LR)  # ty: ignore[invalid-argument-type] - the error under test
    with pytest.raises(ValueError, match="CUDA"):
        _dion([nn.Parameter(torch.zeros(16, 16))], 1.0, NewtonSchulz.GRAM)
    with pytest.raises(TypeError, match="float32 parameters"):
        _dion([weight.detach().bfloat16().requires_grad_()], 1.0, NewtonSchulz.GRAM)
    with pytest.raises(ValueError, match="momentum"):
        Dion(
            [{"params": [weight], ROUTE_KEY: RouteKind.MATRIX.value}],
            lr=LR,
            momentum=1.0,
        )


def test_state_without_newton_schulz_is_rejected() -> None:
    weight = nn.Parameter(torch.zeros(16, 16, device="cuda"))
    optimizer = _dion([weight], 1.0, NewtonSchulz.GRAM)
    state = optimizer.state_dict()
    del state["param_groups"][0][NEWTON_SCHULZ_KEY]
    with pytest.raises(ValueError, match=NEWTON_SCHULZ_KEY):
        optimizer.load_state_dict(state)


def test_saved_state_keeps_one_storage_per_bucket(tmp_path: Path) -> None:
    """Same-shape momenta are views of one bucket, and torch.save keeps that."""
    torch.manual_seed(0)
    layers = [nn.Linear(32, 32, bias=False, device="cuda") for _ in range(2)]
    optimizer = _dion([layer.weight for layer in layers], 1.0, NewtonSchulz.GRAM)
    for layer in layers:
        layer.weight.grad = torch.randn_like(layer.weight)
    optimizer.step()
    path = tmp_path / "state.pt"
    torch.save(optimizer.state_dict(), path)
    loaded = torch.load(path, weights_only=True)
    first, second = (loaded["state"][i]["momentum"] for i in (0, 1))
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert first.untyped_storage().nbytes() == 2 * 32 * 32 * 4


def test_many_block_shapes_compile_under_the_default_recompile_limit() -> None:
    """Many block shapes compile under torch's default recompile limit.

    24 shapes need more static graphs than the default limit of 8; Dion raises
    the limit around its own compiled calls only.
    """
    layers = [
        nn.Linear(16 + 4 * i, 20 + 4 * i, bias=False, device="cuda") for i in range(24)
    ]
    optimizer = _dion([layer.weight for layer in layers], 0.5, NewtonSchulz.GRAM)
    with torch._dynamo.config.patch(
        recompile_limit=8, fail_on_recompile_limit_hit=True
    ):
        for _ in range(2):
            for layer in layers:
                layer.weight.grad = torch.randn_like(layer.weight)
            optimizer.step()
        assert torch._dynamo.config.recompile_limit == 8
    assert all(math.isfinite(float(layer.weight.abs().max())) for layer in layers)


@pytest.mark.slow
@pytest.mark.parametrize("newton_schulz", list(NewtonSchulz))
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_runs_of_any_length_share_compiled_graphs(
    newton_schulz: NewtonSchulz, fraction: float
) -> None:
    """Runs of every length share compiled graphs.

    Parameters without gradients split a bucket into runs of every length; one
    symbolic graph per function covers them, plus the one-block case.
    """
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    weights = [nn.Parameter(torch.randn(48, 80, device="cuda")) for _ in range(8)]
    optimizer = _dion(list(weights), fraction, newton_schulz)
    for missing in range(len(weights)):
        for index, weight in enumerate(weights):
            weight.grad = None if index == missing else torch.randn_like(weight)
        optimizer.step()
    # Two compiled functions (orthogonalize, normalize), at most 3 graphs each.
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] <= 6
    assert all(math.isfinite(float(weight.abs().max())) for weight in weights)


def _two_groups(decayed: Tensor, exempt: Tensor, fraction: float) -> Dion:
    """A decayed and an exempt matrix group at weight decay 0.1."""
    matrix = RouteKind.MATRIX.value
    return Dion(
        [
            {"params": [decayed], ROUTE_KEY: matrix, FRACTION_KEY: fraction},
            {
                "params": [exempt],
                ROUTE_KEY: matrix,
                FRACTION_KEY: fraction,
                "weight_decay": 0.0,
            },
        ],
        lr=LR,
        momentum=0.95,
        muon_beta2=0.95,
        betas=(0.9, 0.99),
        weight_decay=0.1,
        eps=1e-8,
        newton_schulz=NewtonSchulz.GRAM,
        ema=None,
    )


def test_weight_decay_shrinks_every_row_and_skips_exempt_groups() -> None:
    """Weight decay shrinks every row and skips exempt groups.

    With zero gradients the update is zero: a decayed matrix shrinks by
    ``1 - lr * weight_decay`` on every row, not only the selected ones, and a
    group with ``weight_decay = 0.0`` does not change.
    """
    torch.manual_seed(0)
    decayed = nn.Parameter(torch.randn(64, 96, device="cuda"))
    exempt = nn.Parameter(torch.randn(64, 96, device="cuda"))
    start = [decayed.detach().clone(), exempt.detach().clone()]
    optimizer = _two_groups(decayed, exempt, fraction=0.5)
    decayed.grad, exempt.grad = torch.zeros_like(decayed), torch.zeros_like(exempt)
    optimizer.step()
    torch.testing.assert_close(
        decayed.detach(), start[0] * (1 - LR * 0.1), rtol=0, atol=0
    )
    torch.testing.assert_close(exempt.detach(), start[1], rtol=0, atol=0)


def test_head_split_equals_separate_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``num_heads`` split steps each head as its own matrix.

    The exact polar factor keeps bf16 Newton-Schulz rounding, which differs
    between differently compiled batches on some GPUs, out of the comparison.
    """
    monkeypatch.setattr(dionw._buckets, "newton_schulz_fn", lambda _kind: _exact_polar)
    torch.manual_seed(0)
    fused = nn.Parameter(torch.randn(48, 32, device="cuda") * 0.02)
    heads = [
        nn.Parameter(fused.detach()[16 * i : 16 * (i + 1)].clone()) for i in range(3)
    ]
    split = Dion(
        [
            {
                "params": [fused],
                ROUTE_KEY: RouteKind.MATRIX.value,
                FRACTION_KEY: 1.0,
                NUM_HEADS_KEY: 3,
            }
        ],
        lr=LR,
        momentum=0.95,
        muon_beta2=0.95,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        eps=1e-8,
        newton_schulz=NewtonSchulz.GRAM,
        ema=None,
    )
    separate = _dion(heads, 1.0, NewtonSchulz.GRAM)
    for _ in range(3):
        grad = torch.randn_like(fused)
        fused.grad = grad.clone()
        for i, head in enumerate(heads):
            head.grad = grad[16 * i : 16 * (i + 1)].clone()
        split.step()
        separate.step()
    torch.testing.assert_close(fused.detach(), torch.cat([h.detach() for h in heads]))


@pytest.mark.parametrize("fraction", [0.25, 1.0])
def test_orthogonalization_receives_float32_momentum(
    monkeypatch: pytest.MonkeyPatch, fraction: float
) -> None:
    """Momentum reaches Newton-Schulz unrounded; each method rounds it itself.

    Fraction 0.25 covers the selected rows, 1.0 the fused full-block pass.
    """
    received: list[torch.dtype] = []

    def recording(x: Tensor, epsilon: float) -> Tensor:
        received.append(x.dtype)
        return _exact_polar(x, epsilon)

    monkeypatch.setattr(dionw._buckets, "newton_schulz_fn", lambda _kind: recording)
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(64, 128, device="cuda"))
    weight.grad = torch.randn_like(weight)
    _dion([weight], fraction, NewtonSchulz.GRAM).step()
    assert received == [torch.float32]


def test_runs_split_by_missing_gradients_step_like_lone_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bucket split into runs gives each parameter the step it would get alone.

    The exact polar factor keeps bf16 Newton-Schulz rounding, which differs
    between the partial runs' symbolic graphs and the lone parameters' static
    ones on some GPUs, out of the comparison.
    """
    monkeypatch.setattr(dionw._buckets, "newton_schulz_fn", lambda _kind: _exact_polar)
    torch.manual_seed(0)
    start = [torch.randn(32, 48, device="cuda") * 0.02 for _ in range(4)]
    bucketed = [nn.Parameter(t.clone()) for t in start]
    alone = [nn.Parameter(t.clone()) for t in start]
    together = _dion(bucketed, 0.5, NewtonSchulz.GRAM)
    separately = [_dion([p], 0.5, NewtonSchulz.GRAM) for p in alone]
    for _ in range(3):
        for index in (0, 1, 3):
            grad = torch.randn_like(start[index])
            bucketed[index].grad, alone[index].grad = grad.clone(), grad.clone()
        together.step()
        for index in (0, 1, 3):
            separately[index].step()
    for ours, reference in zip(bucketed, alone, strict=True):
        torch.testing.assert_close(
            ours.detach(), reference.detach(), rtol=1e-5, atol=1e-6
        )
    torch.testing.assert_close(bucketed[2].detach(), start[2], rtol=0, atol=0)


def test_initialize_state_and_late_adamw_groups() -> None:
    """State exists before the first step, also for AdamW groups added later.

    initialize_state creates matrix state; an added AdamW group has its state at
    once and steps.
    """
    torch.manual_seed(0)
    matrix = nn.Parameter(torch.randn(32, 32, device="cuda"))
    vector = nn.Parameter(torch.randn(32, device="cuda"))
    optimizer = _dion([matrix], 1.0, NewtonSchulz.GRAM)
    optimizer.initialize_state(matrix)
    assert not optimizer.state[matrix]["momentum"].any()
    with pytest.raises(ValueError, match="not in the optimizer"):
        optimizer.initialize_state(vector)
    optimizer.add_param_group({"params": [vector], ROUTE_KEY: RouteKind.ADAMW.value})
    assert not optimizer.state[vector]["exp_avg"].any()
    before = vector.detach().clone()
    vector.grad = torch.randn_like(vector)
    optimizer.step()
    assert not torch.equal(vector.detach(), before)


def test_groups_are_validated_when_added() -> None:
    weight = nn.Parameter(torch.zeros(30, 16, device="cuda"))
    matrix = RouteKind.MATRIX.value
    with pytest.raises(ValueError, match="does not divide"):
        _dion([weight], 1.0, NewtonSchulz.GRAM).add_param_group(
            {
                "params": [nn.Parameter(torch.zeros(30, 16, device="cuda"))],
                ROUTE_KEY: matrix,
                NUM_HEADS_KEY: 4,
            }
        )
    with pytest.raises(ValueError, match="route it to AdamW"):
        _dion([nn.Parameter(torch.zeros(16, device="cuda"))], 1.0, NewtonSchulz.GRAM)
    with pytest.raises(ValueError, match="lr and weight_decay"):
        _dion([weight], 1.0, NewtonSchulz.GRAM).add_param_group(
            {
                "params": [nn.Parameter(torch.zeros(8, 8, device="cuda"))],
                ROUTE_KEY: matrix,
                "lr": -1.0,
            }
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fraction", 0.25),
        ("newton_schulz", NewtonSchulz.POLAR_EXPRESS.value),
        ("dion_route", "adamw"),
    ],
)
def test_loading_a_structurally_different_state_raises(key: str, value: object) -> None:
    """A checkpoint with a different route, fraction or kind is refused.

    torch would silently restore the checkpoint's values instead.
    """
    weight = nn.Parameter(torch.randn(32, 32, device="cuda"))
    optimizer = _dion([weight], 0.5, NewtonSchulz.GRAM)
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    state["param_groups"][0][key] = value
    with pytest.raises(ValueError, match=f"'{key}'"):
        optimizer.load_state_dict(state)
