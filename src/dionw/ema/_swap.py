"""Temporary swaps of the live weights with their EMA shadows."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from dionw.ema._keys import EMA_SHADOW

__all__ = [
    "EmaSwapEntry",
    "collect_state_ema_swap_entries",
    "optimizer_managed_ema_swap_params",
]


@dataclass(frozen=True)
class EmaSwapEntry:
    """A parameter and its validated shadow, for a temporary weight swap."""

    parameter: torch.nn.Parameter
    shadow: Tensor


def collect_state_ema_swap_entries(
    *,
    params: Sequence[torch.nn.Parameter],
    optimizer_state: Mapping[Tensor, object],
    context: str,
) -> tuple[EmaSwapEntry, ...]:
    """Every parameter's shadow, or nothing before the EMA has started.

    Raises:
        ValueError: If ``params`` is empty.
        TypeError: If a shadow is not a tensor.
        RuntimeError: If only some parameters have shadows (refuses to mix live
            and EMA weights).
    """
    if not params:
        raise ValueError("No parameters available to swap with EMA weights")
    entries: list[EmaSwapEntry] = []
    missing = 0
    for parameter in params:
        raw_state = optimizer_state.get(parameter)
        shadow: object | None = None
        if isinstance(raw_state, MutableMapping):
            shadow = cast("MutableMapping[str, object]", raw_state).get(EMA_SHADOW)
        if shadow is None:
            missing += 1
            continue
        if not isinstance(shadow, Tensor):
            raise TypeError(f"{context} EMA shadow must be a torch.Tensor")
        entries.append(EmaSwapEntry(parameter=parameter, shadow=shadow))
    if not entries:
        return ()
    if missing:
        raise RuntimeError(
            f"{context} EMA swap found {len(entries)} initialized shadows for "
            f"{len(params)} parameters. Refusing to mix live and EMA weights."
        )
    return tuple(entries)


def optimizer_managed_ema_swap_params(
    *,
    model: torch.nn.Module | None,
    optimizer_params: Sequence[torch.nn.Parameter],
    context: str,
) -> tuple[torch.nn.Parameter, ...]:
    """The parameters to swap: the optimizer's, scoped to ``model`` if given.

    Raises:
        ValueError: If the optimizer has no parameters, or ``model`` shares none
            with it, or ``model`` has trainable parameters outside it.
    """
    if not optimizer_params:
        raise ValueError("No parameters available to swap with EMA weights")
    if model is None:
        return tuple(optimizer_params)
    managed = {id(parameter) for parameter in optimizer_params}
    scoped: list[torch.nn.Parameter] = []
    unmanaged = 0
    for parameter in model.parameters():
        if id(parameter) in managed:
            scoped.append(parameter)
        elif parameter.requires_grad:
            unmanaged += 1
    if unmanaged:
        raise ValueError(
            f"{context} EMA swap received a model with {unmanaged} trainable "
            "parameters outside the optimizer."
        )
    if not scoped:
        raise ValueError(
            f"{context} EMA swap received a model with no optimizer-managed parameters."
        )
    return tuple(scoped)
