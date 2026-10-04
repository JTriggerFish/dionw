"""The default route and weight-decay rules of ``param_groups``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from torch import Tensor, nn

from dionw.report import RouteReason
from dionw.routing import Route, RouteKind

__all__ = [
    "EMBEDDING_NAME_SUBSTRINGS",
    "LEARNED_TOKEN_NAME_SUFFIXES",
    "Owner",
    "default_route",
    "is_exempt_from_decay",
    "matrix_dims",
    "owners",
]

# Own-name suffixes of learned token sets.
LEARNED_TOKEN_NAME_SUFFIXES: Final[tuple[str, ...]] = ("token", "tokens")
# Own-name substrings of position embeddings and bias tables.
EMBEDDING_NAME_SUBSTRINGS: Final[tuple[str, ...]] = (
    "pos_embed",
    "position_embedding",
    "relative_position_bias",
)
_WEIGHT: Final[str] = "weight"
_MATRIX: Final[Route] = Route(RouteKind.MATRIX)
_ADAMW: Final[Route] = Route(RouteKind.ADAMW)


@dataclass(frozen=True)
class Owner:
    """Where a parameter is registered.

    Attributes:
        name: Qualified name on the model.
        module: The module that registers it.
        local_name: Its name on that module.
    """

    name: str
    module: nn.Module
    local_name: str


def owners(model: nn.Module) -> dict[int, Owner]:
    """Return each parameter's owner, by parameter id (first registration wins)."""
    found: dict[int, Owner] = {}
    for module_name, module in model.named_modules():
        for local_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{local_name}" if module_name else local_name
            found.setdefault(id(param), Owner(name, module, local_name))
    return found


def matrix_dims(param: Tensor) -> tuple[int, int]:
    """Return ``(rows, cols)`` of a tensor as a matrix (trailing dims flatten)."""
    return int(param.shape[0]), math.prod(int(d) for d in param.shape[1:])


def default_route(
    owner: Owner, param: Tensor, min_matrix_dim: int
) -> tuple[Route, RouteReason]:
    """Route a parameter nothing declared; first matching rule wins.

    Args:
        owner: Where the parameter is registered.
        param: The parameter.
        min_matrix_dim: Smaller-side size below which tensors take AdamW.

    Returns:
        The route and the rule that chose it.
    """
    module_reason = _module_reason(owner)
    if module_reason is not None:
        return _ADAMW, module_reason
    if owner.local_name.endswith(LEARNED_TOKEN_NAME_SUFFIXES):
        return _ADAMW, RouteReason.LEARNED_TOKENS
    if _has_embedding_name(owner.local_name):
        return _ADAMW, RouteReason.EMBEDDING_NAME
    if param.ndim < 2:
        return _ADAMW, RouteReason.VECTOR
    if min(matrix_dims(param)) < min_matrix_dim:
        return _ADAMW, RouteReason.NARROW
    return _MATRIX, RouteReason.MATRIX


def _module_reason(owner: Owner) -> RouteReason | None:
    """Return the AdamW reason of an embedding or grouped-convolution weight."""
    if owner.local_name != _WEIGHT:
        return None
    match owner.module:
        case nn.Embedding() | nn.EmbeddingBag():
            return RouteReason.EMBEDDING
        case (
            nn.Conv1d()
            | nn.Conv2d()
            | nn.Conv3d()
            | nn.ConvTranspose1d()
            | nn.ConvTranspose2d()
            | nn.ConvTranspose3d()
        ) as conv if int(conv.groups) > 1:
            return RouteReason.GROUPED_CONVOLUTION
        case _:
            return None


def _has_embedding_name(local_name: str) -> bool:
    return any(substring in local_name for substring in EMBEDDING_NAME_SUBSTRINGS)


def is_exempt_from_decay(owner: Owner, param: Tensor) -> bool:
    """Return whether the default rule exempts a parameter from weight decay.

    Vectors, learned token sets and embedding-like names are exempt.
    """
    return (
        param.ndim < 2
        or owner.local_name.endswith(LEARNED_TOKEN_NAME_SUFFIXES)
        or _has_embedding_name(owner.local_name)
    )
