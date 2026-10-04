"""``IntegratedEMAOptimizer``: an optimizer mixin with an integrated EMA.

The mixin steps, saves and loads an ``OptimizerIntegratedEMA``, and swaps the
live weights with the shadows for evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import torch
from torch import Tensor
from torch.optim import Optimizer

from dionw.ema._keys import (
    EMA_INITIALIZED,
    EMA_KEY_PREFIX,
    EMA_PENDING,
    EMA_SCRATCH_KEYS,
    EMA_SHADOW,
    EMA_STEP_COUNT_KEY,
    EMA_TMP_CPU,
)
from dionw.ema._swap import (
    collect_state_ema_swap_entries,
    optimizer_managed_ema_swap_params,
)
from dionw.ema._update import FP32
from dionw.ema.tracker import OptimizerIntegratedEMA, new_cpu_staging

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dionw.ema.config import EMAConfig, ProfileRegion

__all__ = ["IntegratedEMAOptimizer"]


def _reject_ema_state_without_ema(
    state_dict: Mapping[str, Any], *, optimizer_name: str
) -> None:
    """Refuse EMA shadows in an optimizer built without EMA.

    They would be kept as stale optimizer state and saved again.

    Raises:
        ValueError: If the state has an EMA step counter or EMA entries.
    """
    has_ema_entries = any(
        key.startswith(EMA_KEY_PREFIX)
        for entry in state_dict["state"].values()
        for key in entry
    )
    if EMA_STEP_COUNT_KEY in state_dict or has_ema_entries:
        raise ValueError(
            f"Checkpoint {optimizer_name} state carries EMA shadows but the optimizer "
            "was built without EMA (ema=None). Build it with the saved EMAConfig to "
            "continue the EMA, or load a fresh optimizer state without the EMA."
        )


def _ema_step_count(state_dict: Mapping[str, Any], optimizer_name: str) -> int:
    """Return the saved EMA step count.

    Raises:
        ValueError: If it is missing or not a non-negative int.
    """
    count = state_dict.get(EMA_STEP_COUNT_KEY)
    if type(count) is not int or count < 0:
        raise ValueError(
            f"{optimizer_name} state has no valid EMA step count "
            f"({EMA_STEP_COUNT_KEY}={count!r}): it was not saved with an EMA"
        )
    return count


def _without_ema_entries(
    state: Mapping[Any, dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
    """Return the per-parameter state with every EMA entry removed."""
    return {
        index: {k: v for k, v in entry.items() if not k.startswith(EMA_KEY_PREFIX)}
        for index, entry in state.items()
    }


class IntegratedEMAOptimizer(Optimizer):
    """Optimizer mixin that updates an ``OptimizerIntegratedEMA`` after each step.

    The shadows and the schedule position are saved with the optimizer state;
    CPU EMA's working buffers are not.

    List it before the concrete optimizer and call ``_init_integrated_ema`` at
    the end of ``__init__``. The concrete optimizer must create a parameter's
    state no later than its first step: shadows are created for parameters
    with state.
    """

    _ema_helper: OptimizerIntegratedEMA | None
    _stored_params: dict[int, Tensor]

    def _init_integrated_ema(
        self, ema: EMAConfig | None, profile_region: ProfileRegion
    ) -> None:
        """Attach the EMA (None disables it).

        Raises:
            ValueError: If the optimizer holds no trainable parameter.
        """
        if not self._iter_trainable_params():
            raise ValueError(
                f"{type(self).__name__} requires at least one trainable parameter"
            )
        self._ema_helper = (
            None if ema is None else OptimizerIntegratedEMA(self, ema, profile_region)
        )
        self._stored_params = {}

    def _iter_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return the optimizer's trainable parameters, in group order."""
        return [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]

    @property
    def ema(self) -> OptimizerIntegratedEMA | None:
        """The EMA, or None when disabled."""
        return self._ema_helper

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Run the concrete optimizer's step, then the due EMA update.

        Parameters without a gradient keep their value but their shadow still
        advances on the global cadence.

        Args:
            closure: Optional closure re-evaluating the loss.

        Returns:
            The closure's loss, or None.

        Raises:
            RuntimeError: While EMA weights are swapped in.
        """
        if self._stored_params:
            raise RuntimeError(
                "step() while EMA weights are swapped in: call "
                "restore_non_ema_weights() first"
            )
        helper = self._ema_helper
        due = False
        if helper is not None:
            helper.begin_step()
            due = helper.should_update_this_step()
        loss = super().step(closure)
        if helper is not None and due:
            for param in self._iter_trainable_params():
                helper.update_param(param)
            helper.finalize_updates()
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer state with the shadows and the EMA step count.

        Pending EMA work is finished first; CPU EMA's working buffers are left
        out.
        """
        if self._ema_helper is None:
            return super().state_dict()
        self._ema_helper.flush_pending_updates()
        state = super().state_dict()
        state["state"] = {
            index: {k: v for k, v in entry.items() if k not in EMA_SCRATCH_KEYS}
            for index, entry in state["state"].items()
        }
        state[EMA_STEP_COUNT_KEY] = self._ema_helper.step_count
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state, the shadows and the EMA step count.

        Shadows are restored directly on the configured EMA device (torch's
        loader would cast them to their parameter's device).

        Raises:
            ValueError: If the checkpoint has EMA state but this optimizer has no
                EMA, or has no valid EMA step count while this one has an EMA.
        """
        helper = self._ema_helper
        name = type(self).__name__
        if helper is None:
            _reject_ema_state_without_ema(state_dict, optimizer_name=name)
            super().load_state_dict(state_dict)
            return
        helper.flush_pending_updates()
        count = _ema_step_count(state_dict, name)
        state = {k: v for k, v in state_dict.items() if k != EMA_STEP_COUNT_KEY}
        state["state"] = _without_ema_entries(state_dict["state"])
        super().load_state_dict(state)
        self._restore_shadows(state_dict, helper)
        helper.step_count = count

    def _restore_shadows(
        self, saved: Mapping[str, Any], helper: OptimizerIntegratedEMA
    ) -> None:
        """Restore every saved shadow on the EMA device.

        Raises:
            TypeError: If a saved shadow is not a tensor.
        """
        for source_group, group in zip(
            saved["param_groups"], self.param_groups, strict=True
        ):
            for index, param in zip(
                source_group["params"], group["params"], strict=True
            ):
                source = saved["state"].get(index, {})
                if EMA_SHADOW in source:
                    self._restore_shadow(param, source, helper)

    def _restore_shadow(
        self, param: Tensor, source: Mapping[str, Any], helper: OptimizerIntegratedEMA
    ) -> None:
        """Restore one shadow and, for CPU EMA, a fresh pinned staging buffer.

        Raises:
            TypeError: If the saved shadow is not a tensor.
        """
        shadow = source[EMA_SHADOW]
        if not isinstance(shadow, Tensor):
            raise TypeError(f"{type(self).__name__} EMA shadow must be a tensor")
        on_gpu = helper.uses_gpu_shadow
        pin = helper.config.pin_memory and not on_gpu
        device = param.device if on_gpu else torch.device("cpu")
        state = self.state[param]
        state[EMA_SHADOW] = torch.empty_like(
            shadow, device=device, dtype=FP32, pin_memory=pin
        ).copy_(shadow)
        state[EMA_INITIALIZED] = source[EMA_INITIALIZED]
        state[EMA_PENDING] = False
        if not on_gpu:
            state[EMA_TMP_CPU] = new_cpu_staging(param, pin=pin)

    @torch.no_grad()
    def swap_ema_weights(self, model: torch.nn.Module | None = None) -> bool:
        """Swap the parameters with their shadows, for evaluation.

        Undo with ``restore_non_ema_weights``; ``step`` refuses to run until then.

        Args:
            model: Scope the swap to this module's parameters; None swaps every
                optimizer parameter.

        Returns:
            Whether EMA weights were applied (False without EMA or before the
            first update).

        Raises:
            RuntimeError: If EMA weights are already swapped in.
            ValueError: If no parameters are available to swap, or ``model``
                shares none with the optimizer or has trainable parameters
                outside it.
        """
        if self._ema_helper is None:
            return False
        if self._stored_params:
            raise RuntimeError("EMA weights are already swapped in")
        self._ema_helper.flush_pending_updates()
        context = type(self).__name__
        params = optimizer_managed_ema_swap_params(
            model=model, optimizer_params=self._iter_trainable_params(), context=context
        )
        entries = collect_state_ema_swap_entries(
            params=params, optimizer_state=self.state, context=context
        )
        for entry in entries:
            param = entry.parameter
            self._stored_params[id(param)] = param.data.clone()
            param.data.copy_(entry.shadow.to(device=param.device, dtype=param.dtype))
        return bool(entries)

    @torch.no_grad()
    def restore_non_ema_weights(self) -> bool:
        """Restore the weights saved by ``swap_ema_weights``.

        Returns:
            False without EMA; True otherwise.
        """
        if self._ema_helper is None:
            return False
        for param in self._iter_trainable_params():
            stored = self._stored_params.get(id(param))
            if stored is not None:
                param.data.copy_(stored)
        self._stored_params.clear()
        return True
