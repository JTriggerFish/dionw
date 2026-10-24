"""One Dion step for a run of consecutive bucket parameters with gradients.

The stages, per block (the ``Dion`` docstring has the math):

1. ``accumulate``: ``M += G``, and for row-selecting blocks each row's l1 norm;
2. ``select``: top-``k`` rows, decayed in ``M`` (error feedback); blocks that
   update every row decay all of ``M`` and orthogonalize its heavy-ball or
   Nesterov form (``MomentumForm``);
3. ``orthogonalize``: the group's Newton-Schulz on the selected (or all) rows;
4. ``finish``: NorMuon row normalization or none (``RowNormalization``), times
   the signed step;
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
    MOMENTUM_FORM_KEY,
    MOMENTUM_KEY,
    MOMENTUM_STATE,
    ParamState,
)
from dionw._momentum_kernels import accumulate_copy_decay, accumulate_row_l1
from dionw._row_normalization import finish
from dionw.matrix_update import MomentumForm

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["MAX_RUN_ELEMENTS", "run_elements", "step_run"]

# Update RMS of AdamW that matrix steps are matched to (Moonshot's 0.2).
ADAMW_UPDATE_RMS: Final[float] = 0.2
# Added to the Frobenius norm before Newton-Schulz; independent of AdamW's eps.
NEWTON_SCHULZ_EPS: Final[float] = 1e-8
# Largest update (blocks x selected rows x cols) one run orthogonalizes at once:
# a step holds a few float32 and bfloat16 copies of it, so a bigger bucket is
# stepped in several runs (1 GiB of float32 per copy). A 5B DiT on GH200 (MOSS
# shapes): full-row Muon's step peak drops from 28.9 GiB above the state to
# 3.45 GiB at the same step time; Dion at fraction 0.25 from 5.85 to 2.79 GiB
# for +7 ms (116 ms).
MAX_RUN_ELEMENTS: Final[int] = 2**28


def run_elements(bucket: Bucket, position: int) -> int:
    """Return the update elements of the parameter at ``position`` of ``bucket``."""
    blocks = bucket.slices[position]
    return (blocks.stop - blocks.start) * bucket.selected_rows * bucket.cols


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
    update, selected_variance = finish(orthogonal, selected_variance, group, step)
    if selected_variance is not None:
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
    nesterov = MomentumForm(group[MOMENTUM_FORM_KEY]) is MomentumForm.NESTEROV
    current = _accumulate_copy_decay(run, float(group[MOMENTUM_KEY]), nesterov)
    orthogonal = _orthogonalize(run, current, run.variance)
    update, variance = finish(orthogonal, run.variance, group, step)
    if variance is not None:
        run.variance.copy_(variance)
    return update


def _accumulate_copy_decay(run: _Run, decay: float, nesterov: bool) -> Tensor:
    """Set ``M = decay * (M + G)`` and return the float32 Newton-Schulz input.

    The input is ``M + G`` before the decay (heavy ball), or ``G + M`` after it
    (Nesterov).
    """
    if not run.fused:
        return _torch_copy_decay(run, decay, nesterov)
    current = torch.empty_like(run.momentum)
    flat, cols = current.view(-1), run.bucket.cols
    for grad, momentum, rows in zip(
        run.grads, run.momenta, run.row_slices(), strict=True
    ):
        accumulate_copy_decay(
            grad,
            momentum,
            flat[rows.start * cols : rows.stop * cols],
            decay,
            nesterov=nesterov,
        )
    return current


def _torch_copy_decay(run: _Run, decay: float, nesterov: bool) -> Tensor:
    """``_accumulate_copy_decay`` in torch ops, for non-contiguous gradients."""
    torch._foreach_add_(run.momenta, run.grads)
    if not nesterov:
        current = run.momentum.clone()
        run.momentum.mul_(decay)
        return current
    run.momentum.mul_(decay)
    current = run.momentum.clone()
    for grad, own in zip(run.grads, run.own_blocks(), strict=True):
        current[own].view(grad.shape).add_(grad)
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
