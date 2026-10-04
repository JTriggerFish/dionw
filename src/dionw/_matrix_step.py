"""One Dion step for a run of consecutive bucket parameters with gradients.

The stages, per block (the ``Dion`` docstring has the math):

1. ``accumulate``: ``M += G``, and for row-selecting blocks each row's l1 norm;
2. ``select``: top-``k`` rows, decayed in ``M`` (error feedback);
3. ``orthogonalize``: Polar Express on the selected (or all) rows;
4. ``normalize_rows``: NorMuon row normalization, times the signed step;
5. ``apply``: decoupled weight decay, then the update added to the weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor

from dionw._buckets import Bucket, raised_recompile_limits
from dionw._keys import (
    MOMENTUM_KEY,
    MOMENTUM_STATE,
    MUON_BETA2_KEY,
    ParamState,
)
from dionw._momentum_kernels import accumulate_copy_decay, accumulate_row_l1

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["step_run"]

# Update RMS of AdamW that matrix steps are matched to (Moonshot's 0.2).
ADAMW_UPDATE_RMS: Final[float] = 0.2
NORMALIZATION_EPS: Final[float] = 1e-8
# Added to the Frobenius norm before Newton-Schulz; independent of AdamW's eps.
NEWTON_SCHULZ_EPS: Final[float] = 1e-8


@dataclass(frozen=True)
class _Run:
    """Consecutive bucket parameters stepped together.

    Attributes:
        bucket: The owning bucket.
        positions: The parameters' positions in the bucket.
        params: The parameters.
        grads: Their gradients.
        momenta: Their momentum views.
        momentum: ``[blocks, rows, cols]`` momentum of the run.
        variance: ``[blocks, rows, 1]`` variance of the run.
        fused: Whether every gradient is contiguous (the Triton passes apply).
        partial: Whether the run is shorter than its bucket.
    """

    bucket: Bucket
    positions: Sequence[int]
    params: list[Tensor]
    grads: list[Tensor]
    momenta: list[Tensor]
    momentum: Tensor
    variance: Tensor
    fused: bool
    partial: bool

    def row_slices(self) -> list[slice]:
        """Return each parameter's rows within the run's ``blocks * rows`` rows."""
        slices = self.bucket.slices
        start, rows = slices[self.positions[0]].start, self.bucket.rows
        return [
            slice((slices[i].start - start) * rows, (slices[i].stop - start) * rows)
            for i in self.positions
        ]

    def own_blocks(self) -> list[slice]:
        """Return each parameter's blocks within the run."""
        slices = self.bucket.slices
        start = slices[self.positions[0]].start
        return [
            slice(slices[i].start - start, slices[i].stop - start)
            for i in self.positions
        ]


def step_run(
    bucket: Bucket, positions: Sequence[int], group: dict[str, Any], state: ParamState
) -> None:
    """Apply one Dion step to bucket parameters ``positions``.

    The positions are consecutive and every parameter has a gradient.

    Raises:
        RuntimeError: If a parameter has no gradient.
    """
    run = _make_run(bucket, positions, state)
    step = bucket.step_scale.fill_(-_step_size(bucket, float(group["lr"])))
    with raised_recompile_limits():
        if bucket.selected_rows < bucket.rows:
            update, row_index = _selected_update(run, group, step)
        else:
            update, row_index = _full_update(run, group, step), None
    _apply(run, update, row_index, group)


def _make_run(bucket: Bucket, positions: Sequence[int], state: ParamState) -> _Run:
    """Collect a run's tensors and views."""
    params: list[Tensor] = [bucket.params[i] for i in positions]
    grads: list[Tensor] = []
    for param in params:
        if param.grad is None:
            raise RuntimeError("a Dion run holds a parameter without a gradient")
        grads.append(param.grad)
    blocks = bucket.run_blocks(positions)
    return _Run(
        bucket=bucket,
        positions=positions,
        params=params,
        grads=grads,
        momenta=[state[p][MOMENTUM_STATE] for p in params],
        momentum=bucket.momentum[blocks],
        variance=bucket.variance[blocks],
        # The fused kernels need row-major gradients (not channels-last ones).
        fused=all(g.is_contiguous() for g in grads),
        partial=len(positions) != len(bucket.params),
    )


def _step_size(bucket: Bucket, lr: float) -> float:
    """Return the step size of a bucket at learning rate ``lr``.

    It is ``lr`` times AdamW's update RMS times the Frobenius compensation of a
    partial-row step.
    """
    rows, cols, k = bucket.rows, bucket.cols, bucket.selected_rows
    return (
        lr
        * ADAMW_UPDATE_RMS
        * math.sqrt(max(rows, cols))
        * math.sqrt(min(rows, cols) / min(k, cols))
    )


def _selected_update(
    run: _Run, group: dict[str, Any], step: Tensor
) -> tuple[Tensor, Tensor]:
    """Accumulate, select the top-k rows, orthogonalize and normalize them.

    Returns:
        The ``[blocks, k, cols]`` float32 update and the ``[blocks, k, 1]`` row
        index.
    """
    bucket, cols = run.bucket, run.bucket.cols
    row_index = torch.topk(
        _accumulate_with_scores(run), bucket.selected_rows, dim=-1, sorted=False
    )
    index = row_index.indices.unsqueeze(-1)
    selected = torch.gather(run.momentum, 1, index.expand(-1, -1, cols))
    run.momentum.scatter_(
        1, index.expand(-1, -1, cols), selected * float(group[MOMENTUM_KEY])
    )
    selected_variance = torch.gather(run.variance, 1, index)
    orthogonal = _orthogonalize(run, selected, selected_variance)
    update, selected_variance = _normalize_rows(
        orthogonal, selected_variance, float(group[MUON_BETA2_KEY]), step
    )
    run.variance.scatter_(1, index, selected_variance)
    return update, index


def _accumulate_with_scores(run: _Run) -> Tensor:
    """``M += G`` and return each row's momentum l1 norm, ``[blocks, rows]``."""
    if not run.fused:
        torch._foreach_add_(run.momenta, run.grads)
        return torch.linalg.vector_norm(run.momentum, ord=1, dim=-1)
    scores = torch.empty(run.momentum.shape[:2], device=run.momentum.device)
    flat = scores.view(-1)
    for grad, momentum, rows in zip(
        run.grads, run.momenta, run.row_slices(), strict=True
    ):
        accumulate_row_l1(grad, momentum, flat[rows], run.bucket.cols)
    return scores


def _full_update(run: _Run, group: dict[str, Any], step: Tensor) -> Tensor:
    """Accumulate, decay, orthogonalize and normalize every row.

    Returns:
        The ``[blocks, rows, cols]`` float32 update.
    """
    current = _accumulate_copy_decay(run, float(group[MOMENTUM_KEY]))
    orthogonal = _orthogonalize(run, current, run.variance)
    update, variance = _normalize_rows(
        orthogonal, run.variance, float(group[MUON_BETA2_KEY]), step
    )
    run.variance.copy_(variance)
    return update


def _accumulate_copy_decay(run: _Run, decay: float) -> Tensor:
    """Return ``M + G`` (float32) and set ``M = decay * (M + G)``."""
    if not run.fused:
        torch._foreach_add_(run.momenta, run.grads)
        current = run.momentum.clone()
        run.momentum.mul_(decay)
        return current
    current = torch.empty_like(run.momentum)
    flat, cols = current.view(-1), run.bucket.cols
    for grad, momentum, rows in zip(
        run.grads, run.momenta, run.row_slices(), strict=True
    ):
        accumulate_copy_decay(
            grad, momentum, flat[rows.start * cols : rows.stop * cols], decay
        )
    return current


def _orthogonalize(run: _Run, matrices: Tensor, variance: Tensor) -> Tensor:
    """Orthogonalize a ``[blocks, k, cols]`` batch.

    A partial run compiles with a symbolic block count, so one graph covers
    every run length.
    """
    if run.partial:
        _mark_block_dim_dynamic(matrices)
    orthogonal = run.bucket.orthogonalize(matrices, NEWTON_SCHULZ_EPS)
    if run.partial:
        _mark_block_dim_dynamic(orthogonal, variance)
    return orthogonal


def _mark_block_dim_dynamic(*tensors: Tensor) -> None:
    """Compile the block dimension of ``[N, ...]`` inputs symbolically."""
    for tensor in tensors:
        torch._dynamo.maybe_mark_dynamic(tensor, 0)


@torch.compile(dynamic=False, fullgraph=True)
def _normalize_rows(
    update: Tensor, variance: Tensor, beta2: float, step: Tensor
) -> tuple[Tensor, Tensor]:
    """NorMuon row normalization.

    Divides each row by the root of an EMA of its mean square, restores the
    block's Frobenius norm and multiplies by ``step`` (a 0-d device tensor, so a
    scheduled LR does not recompile).

    Returns:
        The float32 step and the new variance.
    """
    update = update.float()
    norm = torch.linalg.vector_norm(update, dim=(-2, -1), keepdim=True)
    variance = torch.lerp(variance, update.square().mean(-1, keepdim=True), 1 - beta2)
    normalized = update / (variance.sqrt() + NORMALIZATION_EPS)
    new_norm = torch.linalg.vector_norm(normalized, dim=(-2, -1), keepdim=True)
    restore = norm / new_norm.clamp_min(NORMALIZATION_EPS)
    return normalized * (restore * step), variance


def _apply(
    run: _Run, update: Tensor, row_index: Tensor | None, group: dict[str, Any]
) -> None:
    """Decay every row of the run's weights, then add their block updates."""
    weight_decay = float(group["weight_decay"])
    if weight_decay > 0.0:
        torch._foreach_mul_(run.params, 1.0 - float(group["lr"]) * weight_decay)
    for param, own in zip(run.params, run.own_blocks(), strict=True):
        index = None if row_index is None else row_index[own]
        _add_blocks_(param, update[own], index, run.bucket.rows, run.bucket.cols)


def _add_blocks_(
    target: Tensor, delta: Tensor, index: Tensor | None, rows: int, cols: int
) -> None:
    """``target += delta`` in block layout.

    Args:
        target: A float32 parameter of any layout.
        delta: ``[blocks, k, cols]`` update rows.
        index: ``[blocks, k, 1]`` rows they go to, or None for every row.
        rows: Rows of one block.
        cols: Columns of one block.
    """
    if target.is_contiguous():
        view = target.view(-1, rows, cols)
        if index is None:
            view.add_(delta)
        else:
            view.scatter_add_(1, index.expand(-1, -1, cols), delta)
        return
    # A non-contiguous weight (a channels-last convolution) has no
    # (blocks, rows, cols) view: add through its logical shape.
    if index is not None:
        delta = torch.zeros(delta.size(0), rows, cols, device=delta.device).scatter_(
            1, index.expand(-1, -1, cols), delta
        )
    target.add_(delta.view(target.shape))
