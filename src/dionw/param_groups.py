"""Build ``Dion`` param groups: route every parameter, split weight decay.

Routes. A declared route wins (a ``RouteProvider`` module's ``dion_routes``, or
the ``routes`` argument; conflicting declarations raise). Every other parameter
follows one default rule, first match wins (``RouteReason`` names each outcome):

1. ``nn.Embedding`` / ``nn.EmbeddingBag`` weights: AdamW;
2. grouped (including depthwise) convolution weights: AdamW;
3. learned token sets, a parameter whose own name ends in ``token`` or
   ``tokens`` (``cls_token``, ``register_tokens``): AdamW at any count;
4. a parameter whose own name contains an embedding-like substring
   (``pos_embed``, ``position_embedding``, ``relative_position_bias``): AdamW;
5. vectors and scalars (biases, norm scales): AdamW;
6. tensors whose smaller side, flattened to ``(shape[0], prod(shape[1:]))``, is
   below ``min_matrix_dim``: AdamW;
7. everything else: MATRIX (linear and convolution weights, bare matrix
   parameters, ``nn.MultiheadAttention.in_proj_weight``, transposed
   convolutions).

Input and output layers (a patch embedding reading pixels, a head writing
logits or pixels) are ordinary matrices to the rule; declare them AdamW if
their geometry should not be orthogonalized.

Row fraction. A matrix block (a weight, or one head of a ``num_heads`` split)
updates ``fraction`` of its rows per step when its smaller side is at least
``selection_min_dim``, every row otherwise: narrow blocks are cheap to
orthogonalize whole, and selecting rows of a low-rank factor or an attention
head drops rank directions. A route's ``fraction`` overrides the rule.

Weight decay. Groups of decayed parameters carry no ``weight_decay`` key, so
they take the optimizer's ``weight_decay``; groups of exempt parameters carry
``weight_decay = 0.0``. By default vectors, learned token sets and
embedding-like names are exempt; ``no_weight_decay`` replaces that rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from dionw._buckets import block_shape
from dionw._default_rule import (
    Owner,
    default_route,
    is_exempt_from_decay,
    owners,
)
from dionw._keys import FRACTION_KEY, NUM_HEADS_KEY, ROUTE_KEY
from dionw.report import RoutedParameter, RouteReason, RoutingReport
from dionw.routing import Route, RouteKind, RouteProvider

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

    from torch import Tensor, nn

__all__ = ["ROUTE_KEY", "ParamGroups", "param_groups"]

_LOGGER = logging.getLogger("dionw")


class ParamGroups(NamedTuple):
    """Param groups for ``Dion`` and the routing that produced them.

    It unpacks as ``groups, report``.
    """

    groups: list[dict[str, Any]]
    report: RoutingReport


@dataclass(frozen=True)
class _RowRule:
    """The row-fraction rule: ``fraction`` from ``selection_min_dim`` up."""

    fraction: float
    selection_min_dim: int


@dataclass(frozen=True)
class _GroupKey:
    kind: RouteKind
    num_heads: int | None
    fraction: float
    decayed: bool


def param_groups(
    model: nn.Module,
    *,
    params: Iterable[Tensor] | None = None,
    fraction: float = 0.25,
    selection_min_dim: int = 1024,
    min_matrix_dim: int = 8,
    routes: Mapping[Tensor, Route] | None = None,
    no_weight_decay: Collection[Tensor] | None = None,
) -> ParamGroups:
    """Route each trainable parameter and build ``Dion`` param groups.

    Args:
        model: Module owning the parameters; its ``RouteProvider`` submodules
            declare routes.
        params: Parameters to optimize, all registered on ``model``; None takes
            every parameter of ``model``. Parameters with ``requires_grad=False``
            are skipped.
        fraction: Row fraction of matrix blocks at least ``selection_min_dim``
            on their smaller side, in (0, 1].
        selection_min_dim: Smaller-side size from which blocks select rows.
        min_matrix_dim: Smaller-side size below which tensors take AdamW.
        routes: Extra per-parameter routes, merged with module declarations.
        no_weight_decay: Exactly the parameters exempt from weight decay; None
            applies the default rule (vectors, learned tokens, embedding-like
            names).

    Returns:
        ``(groups, report)``: AdamW groups first, then matrix groups, each in
        parameter order; the report is also logged at INFO on the ``dionw``
        logger.

    Raises:
        ValueError: On out-of-range settings, invalid or conflicting routes, a
            parameter not registered on ``model``, or no trainable parameters.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if selection_min_dim <= 0 or min_matrix_dim <= 0:
        raise ValueError("selection_min_dim and min_matrix_dim must be positive")
    found = owners(model)
    trainable = _trainable(model, params, found)
    exempt = (
        {i for i, p in trainable.items() if is_exempt_from_decay(found[i], p)}
        if no_weight_decay is None
        else {id(p) for p in no_weight_decay}
    )
    declared = _declared_routes(model, routes)
    rule = _RowRule(fraction, selection_min_dim)
    routed = [
        _route(
            found[i], p, declared.get(i), rule, min_matrix_dim, decayed=i not in exempt
        )
        for i, p in trainable.items()
    ]
    groups, report = _build_groups(list(trainable.values()), routed)
    _LOGGER.info("\n".join(report.lines()))
    return ParamGroups(groups=groups, report=report)


def _trainable(
    model: nn.Module, params: Iterable[Tensor] | None, found: dict[int, Owner]
) -> dict[int, Tensor]:
    """Return the distinct trainable parameters by id.

    Raises:
        ValueError: If none require gradients or one is not on ``model``.
    """
    trainable: dict[int, Tensor] = {}
    for param in model.parameters() if params is None else params:
        if param.requires_grad:
            trainable.setdefault(id(param), param)
    if not trainable:
        raise ValueError("No parameters require gradients")
    outside = [tuple(p.shape) for i, p in trainable.items() if i not in found]
    if outside:
        raise ValueError(f"Parameters not registered on the model: {outside[:8]}")
    return trainable


def _declared_routes(
    model: nn.Module, routes: Mapping[Tensor, Route] | None
) -> dict[int, Route]:
    """Return the routes declared by ``RouteProvider`` modules and ``routes``.

    Raises:
        ValueError: If a declared parameter is not registered on ``model`` or
            two declarations disagree.
    """
    pairs = [("the routes argument", p, r) for p, r in (routes or {}).items()]
    for module in model.modules():
        if isinstance(module, RouteProvider):
            source = type(module).__name__
            pairs.extend((source, p, r) for p, r in module.dion_routes())
    registered = {id(p) for p in model.parameters()}
    declared: dict[int, Route] = {}
    for source, param, route in pairs:
        if id(param) not in registered:
            raise ValueError(
                f"{source} declares a route for a parameter (shape "
                f"{tuple(param.shape)}) that is not registered on the model"
            )
        previous = declared.setdefault(id(param), route)
        if previous != route:
            raise ValueError(
                f"Conflicting routes for a parameter of shape {tuple(param.shape)}: "
                f"{previous} vs {route} ({source})"
            )
    return declared


def _route(
    owner: Owner,
    param: Tensor,
    declared: Route | None,
    rule: _RowRule,
    min_matrix_dim: int,
    *,
    decayed: bool,
) -> RoutedParameter:
    """Resolve one parameter's route, row fraction and decay."""
    if declared is None:
        route, reason = default_route(owner, param, min_matrix_dim)
    else:
        route, reason = declared, RouteReason.DECLARED
    match route.kind:
        case RouteKind.MATRIX:
            row_fraction = _matrix_fraction(owner.name, param, route, rule)
        case RouteKind.ADAMW:
            row_fraction = 1.0
        case _ as unreachable:
            raise RuntimeError(f"Unhandled RouteKind: {unreachable}")
    return RoutedParameter(
        name=owner.name,
        shape=tuple(int(d) for d in param.shape),
        kind=route.kind,
        fraction=row_fraction,
        num_heads=route.num_heads,
        decayed=decayed,
        reason=reason,
    )


def _matrix_fraction(name: str, param: Tensor, route: Route, rule: _RowRule) -> float:
    """Return the row fraction of a matrix parameter's blocks.

    Raises:
        ValueError: For a MATRIX route on a vector or an indivisible head split.
    """
    if param.ndim < 2:
        raise ValueError(
            f"MATRIX route on {name} needs >= 2 dims, got {tuple(param.shape)}"
        )
    _, rows, cols = block_shape(param, route.num_heads)
    if route.fraction is not None:
        return float(route.fraction)
    return rule.fraction if min(rows, cols) >= rule.selection_min_dim else 1.0


def _build_groups(
    params: list[Tensor], routed: list[RoutedParameter]
) -> tuple[list[dict[str, Any]], RoutingReport]:
    """Group parameters by route, AdamW first, and order the report alike."""
    grouped: dict[_GroupKey, list[tuple[Tensor, RoutedParameter]]] = {}
    for param, entry in zip(params, routed, strict=True):
        key = _GroupKey(entry.kind, entry.num_heads, entry.fraction, entry.decayed)
        grouped.setdefault(key, []).append((param, entry))
    ordered = sorted(grouped.items(), key=lambda item: item[0].kind is RouteKind.MATRIX)
    groups = [_group(key, [p for p, _ in members]) for key, members in ordered]
    report = RoutingReport(tuple(e for _, members in ordered for _, e in members))
    return groups, report


def _group(key: _GroupKey, params: list[Tensor]) -> dict[str, Any]:
    """Return one optimizer group dict."""
    group: dict[str, Any] = {"params": params, ROUTE_KEY: key.kind.value}
    if key.kind is RouteKind.MATRIX:
        group[FRACTION_KEY] = key.fraction
        group[NUM_HEADS_KEY] = key.num_heads
    if not key.decayed:
        group["weight_decay"] = 0.0
    return group
