"""``Dion``: row-selected NorMuon on matrices, AdamW on the rest, one LR.

The matrix update is microsoft/dion's NorDion2 ("Dion3"; ``fraction = 1`` is
NorMuon), for one device. Per matrix block:

1. ``M += G`` (momentum accumulates every row every step);
2. keep the top ``k = ceil(fraction * rows)`` rows by momentum l1 norm, then
   decay those rows of ``M`` by ``momentum`` (error feedback: the other rows keep
   accumulating until selected);
3. orthogonalize the selected rows (Polar Express, 5 steps);
4. NorMuon: divide each row by the root of an EMA of its mean square
   (``muon_beta2``), then restore the block's Frobenius norm;
5. ``W[rows] -= lr * 0.2 * sqrt(max(rows, cols)) * c * O`` with
   ``c = sqrt(min(rows, cols) / min(k, cols))``.

The ``0.2 * sqrt(max(rows, cols))`` factor matches AdamW's update RMS
(Moonshot's rule), so ``lr`` is an AdamW-equivalent learning rate for every
parameter, and ``c`` gives a block's step the Frobenius norm of its
``fraction = 1`` step (``1 / sqrt(fraction)`` for square or wide blocks).
Weight decay is decoupled, ``W *= 1 - lr * weight_decay``, on every row.
Parameters routed to AdamW get torch's fused AdamW kernel.

A block is a 2D weight, one head of a ``num_heads`` split (rows), or a
higher-dimensional weight flattened to ``(shape[0], prod(shape[1:]))`` (a
convolution's output channels are its rows). Same-shape blocks of a group share
one momentum and one variance buffer and ``state[p]`` holds views into them
(``dionw._buckets``). ``torch.save`` keeps that sharing; saving one parameter's
state alone saves its whole bucket, and formats that reject shared storage
(safetensors) need the views cloned first.

Parameters must be float32 (mixed precision keeps float32 master weights and
computes in bfloat16): an AdamW-sized update is below bfloat16's resolution at
ordinary learning rates, so bfloat16 weights would drop most of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import torch
from torch import Tensor
from torch.optim import Optimizer

from dionw._adamw import adamw_step, init_adamw_state
from dionw._buckets import Bucket, block_shape, build_buckets
from dionw._keys import (
    EXP_AVG_STATE,
    FRACTION_KEY,
    MOMENTUM_KEY,
    MUON_BETA2_KEY,
    NEWTON_SCHULZ_KEY,
    NUM_HEADS_KEY,
    ROUTE_KEY,
    STRUCTURAL_KEYS,
)
from dionw._matrix_step import step_run
from dionw.ema import EMAConfig, IntegratedEMAOptimizer, ProfileRegion, no_profile
from dionw.newton_schulz import NewtonSchulz
from dionw.routing import RouteKind

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["NEWTON_SCHULZ_KEY", "Dion"]


def _check_hyperparameters(
    *, lr: float, weight_decay: float, eps: float, coefficients: dict[str, float]
) -> None:
    """Check the scalar hyperparameters.

    Raises:
        ValueError: Unless ``lr, weight_decay >= 0``, ``eps > 0`` and every
            coefficient (momenta, betas) is in [0, 1).
    """
    if lr < 0.0 or weight_decay < 0.0 or eps <= 0.0:
        raise ValueError("lr and weight_decay must be >= 0 and eps > 0")
    for name, value in coefficients.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}")


def _check_group(group: dict[str, Any]) -> None:
    """Check a group's route, Newton-Schulz kind, lr, weight decay and blocks.

    Raises:
        TypeError: Without a route (not built by ``param_groups``).
        ValueError: On an unknown route or kind, a negative lr or weight decay,
            or an invalid matrix group (see ``_check_matrix_group``).
    """
    if ROUTE_KEY not in group:
        raise TypeError(
            "Dion param groups need a route: build them with "
            "dionw.param_groups(model) instead of passing parameters directly"
        )
    NewtonSchulz(group[NEWTON_SCHULZ_KEY])
    if float(group["lr"]) < 0.0 or float(group["weight_decay"]) < 0.0:
        raise ValueError("a group's lr and weight_decay must be >= 0")
    match RouteKind(group[ROUTE_KEY]):
        case RouteKind.MATRIX:
            _check_matrix_group(group)
        case RouteKind.ADAMW:
            pass
        case _ as unreachable:
            raise RuntimeError(f"Unhandled RouteKind: {unreachable}")


def _check_matrix_group(group: dict[str, Any]) -> None:
    """Check a matrix group's fraction and its parameters' blocks.

    Raises:
        ValueError: On a fraction outside (0, 1], a parameter with fewer than
            two dimensions, or a head split that does not divide the rows.
    """
    if not 0.0 < float(group[FRACTION_KEY]) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {group[FRACTION_KEY]}")
    for param in group["params"]:
        if param.ndim < 2:
            raise ValueError(
                f"a matrix group holds a {param.ndim}-D parameter; route it to AdamW"
            )
        block_shape(param, group[NUM_HEADS_KEY])


def _check_structure(
    saved: list[dict[str, Any]], current: list[dict[str, Any]]
) -> None:
    """Check that saved groups match the current ones on ``STRUCTURAL_KEYS``.

    torch's ``load_state_dict`` restores every group setting from the
    checkpoint, so a changed route, fraction, head split or Newton-Schulz kind
    would otherwise be reverted silently.

    Raises:
        ValueError: On a different group count, a missing key, or a mismatch.
    """
    if len(saved) != len(current):
        raise ValueError(
            f"Dion state has {len(saved)} param groups, the optimizer {len(current)}"
        )
    for index, (old, new) in enumerate(zip(saved, current, strict=True)):
        for key in STRUCTURAL_KEYS:
            if key not in old:
                raise ValueError(
                    f"Dion state group {index} has no {key!r}: it was not saved by "
                    "dionw.Dion"
                )
            if old[key] != new[key]:
                raise ValueError(
                    f"Dion state group {index} {key!r}: checkpoint={old[key]!r}, "
                    f"optimizer={new[key]!r}. Build the groups as the checkpoint "
                    "was built to resume it."
                )


def _check_param(param: Tensor) -> None:
    """Raise unless ``param`` is a float32 CUDA tensor.

    Raises:
        ValueError: Off CUDA.
        TypeError: Not float32.
    """
    if not param.is_cuda:
        raise ValueError(
            f"Dion runs on CUDA devices, got a parameter on {param.device}"
        )
    if param.dtype != torch.float32:
        raise TypeError(
            f"Dion needs float32 parameters, got {param.dtype}: keep float32 master "
            "weights and run the forward and backward in bfloat16 (autocast)"
        )


class _DionCore(Optimizer):
    """The Dion and AdamW updates over routed param groups (no EMA)."""

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        lr: float,
        momentum: float,
        muon_beta2: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float,
        newton_schulz: NewtonSchulz,
    ) -> None:
        """Validate hyperparameters and add the groups (``Dion`` documents them)."""
        _check_hyperparameters(
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            coefficients={
                MOMENTUM_KEY: momentum,
                MUON_BETA2_KEY: muon_beta2,
                "beta1": betas[0],
                "beta2": betas[1],
            },
        )
        defaults = {
            "lr": lr,
            MOMENTUM_KEY: momentum,
            MUON_BETA2_KEY: muon_beta2,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
            FRACTION_KEY: 1.0,
            NUM_HEADS_KEY: None,
            NEWTON_SCHULZ_KEY: newton_schulz.value,
        }
        self._buckets: list[Bucket] | None = None
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Validate and add a group; matrix state is repacked on the next step.

        Raises:
            TypeError: Without a route, for unordered params (a set), or for a
                parameter that is not float32.
            ValueError: On an invalid route, Newton-Schulz kind or fraction, or a
                parameter off CUDA.
        """
        params = param_group["params"]
        if isinstance(params, set):
            raise TypeError("Dion param group params must be ordered, not a set")
        group = {
            **param_group,
            "params": [params] if isinstance(params, Tensor) else list(params),
        }
        for param in group["params"]:
            _check_param(param)
        _check_group({**self.defaults, **group})
        super().add_param_group(group)
        self._buckets = None
        # AdamW state exists from the start, so an integrated EMA tracks every
        # parameter, including ones that have not had a gradient yet. Matrix
        # state exists from the first step (the buckets).
        if RouteKind(group[ROUTE_KEY]) is RouteKind.ADAMW:
            for param in group["params"]:
                init_adamw_state(param, self.state[param])

    def initialize_state(self, param: Tensor) -> None:
        """Create a parameter's fresh state before its first step, without an update.

        An integrated EMA then tracks it even if it never receives a gradient.

        Raises:
            ValueError: If ``param`` is not in the optimizer.
        """
        group = next(
            (g for g in self.param_groups if any(p is param for p in g["params"])), None
        )
        if group is None:
            raise ValueError("parameter is not in the optimizer")
        match RouteKind(group[ROUTE_KEY]):
            case RouteKind.MATRIX:
                self._ensure_buckets()
            case RouteKind.ADAMW:
                if EXP_AVG_STATE not in self.state[param]:
                    init_adamw_state(param, self.state[param])
            case _ as unreachable:
                raise RuntimeError(f"Unhandled RouteKind: {unreachable}")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load, then repack matrix state into buckets on the next step.

        Raises:
            ValueError: If the saved groups differ from the optimizer's in count,
                route, fraction, head split or Newton-Schulz kind.
        """
        _check_structure(state_dict["param_groups"], self.param_groups)
        super().load_state_dict(state_dict)
        self._buckets = None

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Update every parameter with a gradient.

        Args:
            closure: Optional closure re-evaluating the loss.

        Returns:
            The closure's loss, or None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for bucket in self._ensure_buckets():
            self._step_bucket(bucket)
        for group in self.param_groups:
            if RouteKind(group[ROUTE_KEY]) is RouteKind.ADAMW:
                adamw_step(group, self.state)
        return loss

    def _ensure_buckets(self) -> list[Bucket]:
        """Return the matrix buckets, building them after a layout change."""
        if self._buckets is None:
            self._buckets = build_buckets(self.param_groups, self.state)
        return self._buckets

    def _step_bucket(self, bucket: Bucket) -> None:
        """Step the bucket's parameters that have gradients, in contiguous runs."""
        group = self.param_groups[bucket.group_index]
        run: list[int] = []
        for position, param in enumerate(bucket.params):
            if param.grad is not None:
                run.append(position)
            elif run:
                step_run(bucket, run, group, self.state)
                run = []
        if run:
            step_run(bucket, run, group, self.state)


class Dion(IntegratedEMAOptimizer, _DionCore):
    """Dion on matrices, AdamW on the rest, with an optional integrated EMA.

    Build the groups with ``dionw.param_groups(model)``. ``lr`` is an
    AdamW-equivalent learning rate for every parameter: an existing AdamW
    learning rate, warmup and schedule carry over unchanged.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        *,
        lr: float,
        momentum: float = 0.95,
        muon_beta2: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        newton_schulz: NewtonSchulz = NewtonSchulz.GRAM,
        ema: EMAConfig | None = None,
        profile_region: ProfileRegion = no_profile,
    ) -> None:
        """Create the optimizer.

        Args:
            params: Groups from ``dionw.param_groups``.
            lr: Learning rate, AdamW-equivalent for every parameter.
            momentum: Matrix momentum (error-feedback decay of selected rows).
            muon_beta2: EMA decay of each row's squared update (NorMuon).
            betas: AdamW betas of AdamW-routed parameters.
            weight_decay: Decoupled weight decay of groups without their own.
            eps: AdamW epsilon.
            newton_schulz: Orthogonalization of matrix updates.
            ema: EMA schedule and placement, or None for no EMA.
            profile_region: Named profiling region around EMA work.

        Raises:
            TypeError: For groups without a route or non-float32 parameters.
            ValueError: On out-of-range hyperparameters or parameters off CUDA.
        """
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            muon_beta2=muon_beta2,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            newton_schulz=newton_schulz,
        )
        self._init_integrated_ema(ema, profile_region)
