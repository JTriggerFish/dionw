"""EMA shadow arithmetic: ``lerp`` toward the live weights."""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

__all__ = ["FP32", "ema_update_", "foreach_ema_update_"]

FP32: Final[torch.dtype] = torch.float32
# Largest FP32 upcast staged at once for non-FP32 sources on the GPU.
_GPU_EMA_STAGING_CHUNK_NUMEL: Final[int] = 1 << 24


def ema_update_(shadow: Tensor, source: Tensor, *, decay: float) -> None:
    """Move a shadow toward ``source`` in place: ``lerp(shadow, source, 1 - decay)``.

    The difference form keeps a shadow of a constant source bit-exact;
    ``shadow * decay + source * (1 - decay)`` drifts, because the two float32
    weights do not sum to one.

    Args:
        shadow: Tensor updated in place.
        source: Tensor of ``shadow``'s dtype (``lerp`` rejects mixed dtypes).
        decay: Decay in [0, 1].
    """
    shadow.lerp_(source, 1.0 - decay)


@torch.no_grad()
def foreach_ema_update_(
    shadows: list[Tensor], sources: list[Tensor], *, decay: float
) -> None:
    """Apply ``ema_update_`` to lists of shadows and sources.

    FP32 sources update in one ``_foreach_lerp_``; others are upcast into
    transient FP32 copies a bounded chunk at a time.

    Raises:
        TypeError: If a shadow is not float32.
        ValueError: If the lists differ in length.
    """
    if any(shadow.dtype != FP32 for shadow in shadows):
        raise TypeError("EMA shadows must be float32")
    weight = 1.0 - decay
    direct: tuple[list[Tensor], list[Tensor]] = ([], [])
    upcast: tuple[list[Tensor], list[Tensor]] = ([], [])
    for shadow, source in zip(shadows, sources, strict=True):
        target = direct if source.dtype == FP32 else upcast
        target[0].append(shadow)
        target[1].append(source)
    if direct[0]:
        torch._foreach_lerp_(direct[0], direct[1], weight)
    start = 0
    while start < len(upcast[1]):
        stop = start + 1
        numel = upcast[1][start].numel()
        while (
            stop < len(upcast[1])
            and numel + upcast[1][stop].numel() <= _GPU_EMA_STAGING_CHUNK_NUMEL
        ):
            numel += upcast[1][stop].numel()
            stop += 1
        chunk = upcast[1][start:stop]
        staged = [torch.empty_like(source, dtype=FP32) for source in chunk]
        torch._foreach_copy_(staged, chunk)
        torch._foreach_lerp_(upcast[0][start:stop], staged, weight)
        start = stop
