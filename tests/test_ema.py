"""The integrated EMA: schedule, GPU and CPU shadows, save/load, weight swaps."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dionw import (
    FRACTION_KEY,
    ROUTE_KEY,
    Dion,
    EMAConfig,
    EMADevice,
    IntegratedEMAOptimizer,
    NewtonSchulz,
    RouteKind,
)
from dionw.ema import no_profile


def _ema(
    device: EMADevice,
    *,
    decay: float,
    every: int,
    start_step: int,
    decay_final: float | None,
    ramp_steps: int | None,
) -> EMAConfig:
    return EMAConfig(
        decay=decay,
        device=device,
        pin_memory=True,
        update_every_n_steps=every,
        start_step=start_step,
        decay_final=decay_final,
        decay_ramp_steps=ramp_steps,
    )


def _optimizer(params: list[nn.Parameter], ema: EMAConfig | None) -> Dion:
    """A matrix and a vector, one per route."""
    matrix, vector = params
    return Dion(
        [
            {"params": [vector], ROUTE_KEY: RouteKind.ADAMW.value},
            {"params": [matrix], ROUTE_KEY: RouteKind.MATRIX.value, FRACTION_KEY: 0.5},
        ],
        lr=1e-2,
        momentum=0.95,
        muon_beta2=0.95,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        eps=1e-8,
        newton_schulz=NewtonSchulz.GRAM,
        ema=ema,
    )


def _params() -> list[nn.Parameter]:
    torch.manual_seed(0)
    return [
        nn.Parameter(torch.randn(32, 48, device="cuda")),
        nn.Parameter(torch.randn(48, device="cuda")),
    ]


def _step(params: list[nn.Parameter], optimizer: Dion) -> None:
    for param in params:
        param.grad = torch.randn_like(param)
    optimizer.step()


@pytest.mark.parametrize("device", list(EMADevice))
@pytest.mark.parametrize(("every", "start_step"), [(1, 0), (2, 0), (1, 3), (2, 3)])
def test_shadows_follow_the_schedule(
    device: EMADevice, every: int, start_step: int
) -> None:
    """Shadows equal a reference EMA of the live weights.

    They are copied at the first update (or at start_step), then lerped with
    decay ** every on due steps.
    """
    decay = 0.8
    params = _params()
    optimizer = _optimizer(
        params,
        _ema(
            device,
            decay=decay,
            every=every,
            start_step=start_step,
            decay_final=None,
            ramp_steps=None,
        ),
    )
    reference: list[torch.Tensor] | None = None
    for step in range(1, 10):
        _step(params, optimizer)
        live = [p.detach().clone() for p in params]
        if start_step > 0:
            if step < start_step:
                continue
            due = step == start_step or (step - start_step) % every == 0
            initialize = step == start_step
        else:
            due = step % every == 0
            initialize = reference is None
        if not due:
            continue
        if initialize or reference is None:
            reference = live
        else:
            reference = [
                r.lerp(x, 1 - decay**every)
                for r, x in zip(reference, live, strict=True)
            ]
    assert optimizer.ema is not None
    optimizer.ema.flush_pending_updates()
    assert reference is not None
    for param, expected in zip(params, reference, strict=True):
        shadow = optimizer.state[param]["ema_shadow"]
        assert shadow.device.type == ("cuda" if device is EMADevice.GPU else "cpu")
        torch.testing.assert_close(shadow.cuda(), expected, rtol=1e-6, atol=1e-6)


def test_decay_ramp() -> None:
    config = _ema(
        EMADevice.GPU, decay=0.9, every=1, start_step=0, decay_final=0.99, ramp_steps=10
    )
    assert config.decay_at_step(0) == 0.9
    assert config.decay_at_step(5) == pytest.approx(0.945)
    assert config.decay_at_step(50) == pytest.approx(0.99)
    with pytest.raises(ValueError, match="together"):
        _ema(
            EMADevice.GPU,
            decay=0.9,
            every=1,
            start_step=0,
            decay_final=0.99,
            ramp_steps=None,
        )


@pytest.mark.parametrize("device", list(EMADevice))
def test_state_dict_round_trip_restores_shadows_and_count(device: EMADevice) -> None:
    ema = _ema(
        device, decay=0.9, every=1, start_step=0, decay_final=None, ramp_steps=None
    )
    params = _params()
    optimizer = _optimizer(params, ema)
    for _ in range(3):
        _step(params, optimizer)
    state = optimizer.state_dict()
    restored = _optimizer(params, ema)
    restored.load_state_dict(state)
    assert restored.ema is not None
    assert restored.ema.step_count == 3
    for param in params:
        saved = optimizer.state[param]["ema_shadow"]
        loaded = restored.state[param]["ema_shadow"]
        assert loaded.device == saved.device
        torch.testing.assert_close(loaded, saved, rtol=0, atol=0)
    with pytest.raises(ValueError, match="without EMA"):
        _optimizer(params, None).load_state_dict(state)
    with pytest.raises(ValueError, match="non-negative"):
        restored.ema.step_count = -1


@pytest.mark.parametrize("device", list(EMADevice))
def test_swap_and_restore(device: EMADevice) -> None:
    ema = _ema(
        device, decay=0.5, every=1, start_step=0, decay_final=None, ramp_steps=None
    )
    params = _params()
    optimizer = _optimizer(params, ema)
    assert not optimizer.swap_ema_weights()  # no shadows before the first update
    for _ in range(3):
        _step(params, optimizer)
    live = [p.detach().clone() for p in params]
    assert optimizer.swap_ema_weights()
    for param in params:
        torch.testing.assert_close(
            param.detach(), optimizer.state[param]["ema_shadow"].cuda(), rtol=0, atol=0
        )
    assert optimizer.restore_non_ema_weights()
    for param, value in zip(params, live, strict=True):
        torch.testing.assert_close(param.detach(), value, rtol=0, atol=0)


def test_constant_parameter_keeps_a_bit_exact_shadow() -> None:
    """Lerp keeps the shadow of an unchanging parameter exact (no gain drift)."""
    ema = _ema(
        EMADevice.GPU,
        decay=0.999,
        every=1,
        start_step=0,
        decay_final=None,
        ramp_steps=None,
    )
    params = _params()
    optimizer = _optimizer(params, ema)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    for _ in range(20):
        _step(params, optimizer)
    for param in params:
        assert torch.equal(optimizer.state[param]["ema_shadow"], param.detach())


def test_swap_guards_and_parameters_without_gradients() -> None:
    """Swaps work before a parameter's first gradient and guard against misuse.

    AdamW state is created at construction, so every parameter has a shadow; a
    second swap and a step while swapped raise.
    """
    ema = _ema(
        EMADevice.GPU,
        decay=0.5,
        every=1,
        start_step=0,
        decay_final=None,
        ramp_steps=None,
    )
    params = _params()
    optimizer = _optimizer(params, ema)
    matrix, vector = params
    for _ in range(2):
        matrix.grad = torch.randn_like(matrix)
        optimizer.step()
    torch.testing.assert_close(
        optimizer.state[vector]["ema_shadow"], vector.detach(), rtol=0, atol=0
    )
    assert optimizer.swap_ema_weights()
    with pytest.raises(RuntimeError, match="already swapped"):
        optimizer.swap_ema_weights()
    with pytest.raises(RuntimeError, match="swapped in"):
        optimizer.step()
    optimizer.restore_non_ema_weights()
    optimizer.step()


def test_swap_scoped_to_a_model_rejects_foreign_trainable_parameters() -> None:
    ema = _ema(
        EMADevice.GPU,
        decay=0.5,
        every=1,
        start_step=0,
        decay_final=None,
        ramp_steps=None,
    )
    params = _params()
    optimizer = _optimizer(params, ema)
    _step(params, optimizer)
    stranger = nn.Linear(4, 4).cuda()
    with pytest.raises(ValueError, match="no optimizer-managed parameters"):
        optimizer.swap_ema_weights(
            nn.Sequential(nn.Linear(4, 4).cuda().requires_grad_(False))
        )
    with pytest.raises(ValueError, match="outside the optimizer"):
        optimizer.swap_ema_weights(stranger)


def test_cpu_working_buffers_are_not_saved() -> None:
    ema = _ema(
        EMADevice.CPU,
        decay=0.9,
        every=1,
        start_step=0,
        decay_final=None,
        ramp_steps=None,
    )
    params = _params()
    optimizer = _optimizer(params, ema)
    _step(params, optimizer)
    saved = optimizer.state_dict()
    for entry in saved["state"].values():
        assert "ema_shadow" in entry
        assert {"ema_tmp_cpu", "ema_pending", "ema_event"}.isdisjoint(entry)
    assert "ema_tmp_cpu" in optimizer.state[params[0]]


@pytest.mark.parametrize(
    ("saved_on", "loaded_on"),
    [(EMADevice.GPU, EMADevice.CPU), (EMADevice.CPU, EMADevice.GPU)],
)
def test_checkpoints_move_between_gpu_and_cpu_shadows(
    saved_on: EMADevice, loaded_on: EMADevice
) -> None:
    """A checkpoint moves between GPU and CPU shadows.

    Loaded into the other device, it keeps updating like the original.
    """
    configs = {
        device: _ema(
            device, decay=0.8, every=1, start_step=0, decay_final=None, ramp_steps=None
        )
        for device in EMADevice
    }
    params = _params()
    original = _optimizer(params, configs[saved_on])
    for _ in range(2):
        _step(params, original)
    moved = _optimizer(params, configs[loaded_on])
    # A deep copy, as a save and load would make: torch's loader keeps tensors
    # already on the right device, which would share state between the two.
    moved.load_state_dict(copy.deepcopy(original.state_dict()))
    expected_device = "cuda" if loaded_on is EMADevice.GPU else "cpu"
    for param in params:
        assert moved.state[param]["ema_shadow"].device.type == expected_device
    torch.manual_seed(1)
    grads = [torch.randn_like(p) for p in params]
    snapshot = [p.detach().clone() for p in params]
    for optimizer in (original, moved):
        with torch.no_grad():
            for param, value in zip(params, snapshot, strict=True):
                param.copy_(value)
        for param, grad in zip(params, grads, strict=True):
            param.grad = grad.clone()
        optimizer.step()
    assert original.ema is not None
    assert moved.ema is not None
    original.ema.flush_pending_updates()
    moved.ema.flush_pending_updates()
    for param in params:
        torch.testing.assert_close(
            moved.state[param]["ema_shadow"].cuda(),
            original.state[param]["ema_shadow"].cuda(),
            rtol=0,
            atol=0,
        )


def test_decay_ramp_drives_the_shadow_updates() -> None:
    """Each update uses the ramped decay of its step."""
    ema = _ema(
        EMADevice.GPU, decay=0.5, every=1, start_step=0, decay_final=0.9, ramp_steps=4
    )
    params = _params()
    optimizer = _optimizer(params, ema)
    reference: list[torch.Tensor] | None = None
    for step in range(1, 7):
        _step(params, optimizer)
        live = [p.detach().clone() for p in params]
        if reference is None:
            reference = live
            continue
        weight = 1 - ema.decay_at_step(step)
        reference = [r.lerp(x, weight) for r, x in zip(reference, live, strict=True)]
    assert reference is not None
    for param, expected in zip(params, reference, strict=True):
        torch.testing.assert_close(
            optimizer.state[param]["ema_shadow"], expected, rtol=1e-6, atol=1e-6
        )


class _AdamWWithEMA(IntegratedEMAOptimizer, torch.optim.AdamW):
    """The mixin on a stock torch optimizer."""

    def __init__(self, params: list[nn.Parameter], ema: EMAConfig) -> None:
        super().__init__(params, lr=1e-2)
        self._init_integrated_ema(ema, no_profile)


@pytest.mark.parametrize("device", list(EMADevice))
def test_mixin_tracks_bfloat16_parameters_of_a_stock_optimizer(
    device: EMADevice,
) -> None:
    """The mixin tracks bfloat16 weights of torch's AdamW in float32 shadows.

    This exercises the EMA's generic non-float32 paths.
    """
    ema = _ema(
        device, decay=0.7, every=1, start_step=0, decay_final=None, ramp_steps=None
    )
    torch.manual_seed(0)
    params = [nn.Parameter(torch.randn(64, 32, device="cuda", dtype=torch.bfloat16))]
    optimizer = _AdamWWithEMA(params, ema)
    reference: torch.Tensor | None = None
    for _ in range(4):
        params[0].grad = torch.randn_like(params[0])
        optimizer.step()
        live = params[0].detach().float()
        reference = live if reference is None else reference.lerp(live, 1 - 0.7)
    assert optimizer.ema is not None
    optimizer.ema.flush_pending_updates()
    shadow = optimizer.state[params[0]]["ema_shadow"]
    assert shadow.dtype == torch.float32
    assert reference is not None
    torch.testing.assert_close(shadow.cuda(), reference, rtol=1e-6, atol=1e-6)
