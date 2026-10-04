"""Bucketed matrix state: same-shape blocks of a group stored contiguously.

Each bucket owns one momentum ``[N, rows, cols]`` and one variance
``[N, rows, 1]`` buffer for its ``N`` blocks; ``optimizer.state[p]`` holds views
into them, so a step never stacks parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor, nn

from dionw._keys import (
    FRACTION_KEY,
    MOMENTUM_STATE,
    NEWTON_SCHULZ_KEY,
    NUM_HEADS_KEY,
    ROUTE_KEY,
    VARIANCE_STATE,
    ParamState,
)
from dionw.newton_schulz import NewtonSchulz, Orthogonalize, newton_schulz_fn
from dionw.routing import RouteKind

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

__all__ = ["Bucket", "block_shape", "build_buckets", "raised_recompile_limits"]

# Graphs one bucket shape can compile per function: its full static shape, a
# symbolic block count for runs that skip parameters, and the one-block case.
_GRAPHS_PER_BUCKET_SHAPE: Final[int] = 3
# (blocks, orthogonalized rows, cols) of every bucket built in this process: the
# compiled functions are shared by all Dion instances.
_BUCKET_SHAPES: set[tuple[int, int, int]] = set()


@dataclass
class Bucket:
    """Same-shape blocks of one matrix group.

    Attributes:
        group_index: Index of the owning param group.
        orthogonalize: The group's Newton-Schulz function.
        params: The bucket's parameters, in group order.
        slices: Each parameter's range of blocks.
        rows: Rows of one block.
        cols: Columns of one block.
        selected_rows: Rows updated per step, ``ceil(fraction * rows)``.
        momentum: ``[N, rows, cols]`` float32 momentum of every block.
        variance: ``[N, rows, 1]`` float32 NorMuon row variance.
        step_scale: Signed step size, refilled in place each step (building a
            device tensor from a float would synchronize with the GPU).
    """

    group_index: int
    orthogonalize: Orthogonalize
    params: list[nn.Parameter]
    slices: list[slice]
    rows: int
    cols: int
    selected_rows: int
    momentum: Tensor
    variance: Tensor
    step_scale: Tensor

    def run_blocks(self, positions: Sequence[int]) -> slice:
        """Return the blocks of consecutive parameters ``positions``."""
        return slice(self.slices[positions[0]].start, self.slices[positions[-1]].stop)


def block_shape(param: Tensor, num_heads: int | None) -> tuple[int, int, int]:
    """Return ``(blocks, rows, cols)`` of a matrix parameter's blocks.

    Trailing dimensions flatten into columns; a head split divides the rows.

    Raises:
        ValueError: If ``num_heads`` does not divide the rows.
    """
    rows, cols = int(param.shape[0]), math.prod(int(d) for d in param.shape[1:])
    if num_heads is None:
        return 1, rows, cols
    if rows % num_heads != 0:
        raise ValueError(f"num_heads={num_heads} does not divide {rows} rows")
    return num_heads, rows // num_heads, cols


def raised_recompile_limits() -> AbstractContextManager[None]:
    """Return a context raising dynamo's recompile limits for Dion's compiled calls.

    The compiled functions use one static graph per block shape (1.35x to 2.3x
    faster than a dynamic-shape graph on DiT-XL shapes, RTX 4090 to GH200),
    which can exceed torch's
    default limits (8 per guard set, 256 per function) and fail a ``fullgraph``
    compile. Both are raised only around these calls, to the global value or
    enough for every bucket shape built so far, whichever is higher.
    """
    needed = _GRAPHS_PER_BUCKET_SHAPE * len(_BUCKET_SHAPES)
    config = torch._dynamo.config
    return config.patch(
        recompile_limit=max(int(config.recompile_limit), needed),
        accumulated_recompile_limit=max(
            int(config.accumulated_recompile_limit), needed
        ),
    )


def build_buckets(
    param_groups: list[dict[str, Any]], state: ParamState
) -> list[Bucket]:
    """Pack every matrix group's state into per-shape buckets.

    Existing momentum and variance entries (a loaded checkpoint, or the buckets
    of a previous layout) are copied in; ``state[p]`` then holds views.

    Args:
        param_groups: The optimizer's groups.
        state: The optimizer's per-parameter state.

    Returns:
        The buckets, in group order.
    """
    buckets: list[Bucket] = []
    for index, group in enumerate(param_groups):
        if RouteKind(group[ROUTE_KEY]) is not RouteKind.MATRIX:
            continue
        orthogonalize = newton_schulz_fn(NewtonSchulz(group[NEWTON_SCHULZ_KEY]))
        for params in _same_shape_params(group).values():
            bucket = _new_bucket(index, group, params, orthogonalize)
            _attach_state(bucket, state)
            _BUCKET_SHAPES.add(
                (len(bucket.momentum), bucket.selected_rows, bucket.cols)
            )
            buckets.append(bucket)
    return buckets


def _same_shape_params(
    group: dict[str, Any],
) -> dict[tuple[int, int, torch.device], list[nn.Parameter]]:
    """Return the group's parameters keyed by ``(rows, cols, device)`` of their blocks."""
    by_shape: dict[tuple[int, int, torch.device], list[nn.Parameter]] = {}
    for param in group["params"]:
        _, rows, cols = block_shape(param, group[NUM_HEADS_KEY])
        by_shape.setdefault((rows, cols, param.device), []).append(param)
    return by_shape


def _new_bucket(
    index: int,
    group: dict[str, Any],
    params: list[nn.Parameter],
    orthogonalize: Orthogonalize,
) -> Bucket:
    """Return a zero-state bucket for same-shape parameters of group ``index``."""
    slices: list[slice] = []
    for param in params:
        start = slices[-1].stop if slices else 0
        slices.append(slice(start, start + block_shape(param, group[NUM_HEADS_KEY])[0]))
    _, rows, cols = block_shape(params[0], group[NUM_HEADS_KEY])
    blocks, device = slices[-1].stop, params[0].device
    return Bucket(
        group_index=index,
        orthogonalize=orthogonalize,
        params=params,
        slices=slices,
        rows=rows,
        cols=cols,
        selected_rows=min(rows, math.ceil(float(group[FRACTION_KEY]) * rows)),
        momentum=torch.zeros(blocks, rows, cols, device=device),
        variance=torch.zeros(blocks, rows, 1, device=device),
        step_scale=torch.zeros((), device=device),
    )


def _attach_state(bucket: Bucket, state: ParamState) -> None:
    """Point each parameter's state at its views of the bucket, copying old state."""
    for param, blocks in zip(bucket.params, bucket.slices, strict=True):
        entry = state[param]
        momentum = bucket.momentum[blocks].view(param.shape)
        variance = bucket.variance[blocks].view(param.shape[0], 1)
        if MOMENTUM_STATE in entry:
            momentum.copy_(entry[MOMENTUM_STATE])
            variance.copy_(entry[VARIANCE_STATE])
        entry[MOMENTUM_STATE] = momentum
        entry[VARIANCE_STATE] = variance
