"""The AdamW route: torch's fused AdamW kernel (``AdamW(fused=True)``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from dionw._keys import ADAMW_STEP_STATE, EXP_AVG_SQ_STATE, EXP_AVG_STATE

if TYPE_CHECKING:
    from dionw._keys import ParamState

__all__ = ["adamw_step", "init_adamw_state"]


def init_adamw_state(param: Tensor, state: dict[str, Any]) -> None:
    """Create fresh AdamW moments and step counter in ``state``."""
    state[EXP_AVG_STATE] = torch.zeros_like(param)
    state[EXP_AVG_SQ_STATE] = torch.zeros_like(param)
    state[ADAMW_STEP_STATE] = torch.zeros((), dtype=torch.float32, device=param.device)


def adamw_step(group: dict[str, Any], state: ParamState) -> None:
    """Apply one fused AdamW step to the group's parameters with gradients.

    Raises:
        RuntimeError: If a gradient disappears between selection and the step.
    """
    params = [p for p in group["params"] if p.grad is not None]
    if not params:
        return
    entries = [state[p] for p in params]
    grads: list[Tensor] = []
    for param, entry in zip(params, entries, strict=True):
        if EXP_AVG_STATE not in entry:
            init_adamw_state(param, entry)
        if param.grad is None:
            raise RuntimeError("an AdamW update holds a parameter without a gradient")
        grads.append(param.grad)
    steps = [entry[ADAMW_STEP_STATE] for entry in entries]
    torch._foreach_add_(steps, 1)
    beta1, beta2 = group["betas"]
    torch._fused_adamw_(
        params,
        grads,
        [entry[EXP_AVG_STATE] for entry in entries],
        [entry[EXP_AVG_SQ_STATE] for entry in entries],
        [],
        steps,
        amsgrad=False,
        lr=float(group["lr"]),
        beta1=float(beta1),
        beta2=float(beta2),
        weight_decay=float(group["weight_decay"]),
        eps=float(group["eps"]),
        maximize=False,
        grad_scale=None,
        found_inf=None,
    )
