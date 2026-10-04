"""Per-parameter routes: which update a parameter receives under ``Dion``.

Most models need no declarations: ``dionw.param_groups`` routes every parameter
by a default rule. A module overrides it for its own parameters by defining
``dion_routes`` (``RouteProvider``), or the caller passes a ``routes`` mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

__all__ = ["Route", "RouteKind", "RouteProvider"]


class RouteKind(Enum):
    """Which update a parameter receives."""

    # Element-wise AdamW: vectors, embeddings, learned tokens, and layers that
    # read or write data directly (patch embeddings, output heads) when declared.
    ADAMW = "adamw"
    # The orthogonalized matrix update (row-selected NorMuon).
    MATRIX = "matrix"


@dataclass(frozen=True)
class Route:
    """A parameter's route.

    Attributes:
        kind: The update the parameter receives.
        num_heads: For MATRIX, orthogonalize this many equal row blocks
            independently (a fused QKV projection: 3 x attention heads); None
            keeps the whole matrix. Must be None for ADAMW.
        fraction: For MATRIX, the share of each block's rows updated per step,
            overriding the size rule of ``param_groups``; None applies the rule.
            Must be None for ADAMW.

    Raises:
        ValueError: If ``num_heads`` or ``fraction`` is invalid for ``kind``.
    """

    kind: RouteKind
    num_heads: int | None = None
    fraction: float | None = None

    def __post_init__(self) -> None:
        """Reject matrix options on an AdamW route and out-of-range values."""
        match self.kind:
            case RouteKind.MATRIX:
                if self.num_heads is not None and self.num_heads <= 0:
                    raise ValueError(
                        f"Route num_heads must be positive, got {self.num_heads}"
                    )
                if self.fraction is not None and not 0.0 < self.fraction <= 1.0:
                    raise ValueError(
                        f"Route fraction must be in (0, 1], got {self.fraction}"
                    )
            case RouteKind.ADAMW:
                if self.num_heads is not None or self.fraction is not None:
                    raise ValueError(
                        "Route num_heads and fraction apply only to MATRIX routes"
                    )
            case _ as unreachable:
                raise RuntimeError(f"Unhandled RouteKind: {unreachable}")


@runtime_checkable
class RouteProvider(Protocol):
    """A module that declares routes for some of its own parameters.

    Any ``nn.Module`` with this method qualifies; no base class is needed.
    """

    def dion_routes(self) -> Iterable[tuple[Tensor, Route]]:
        """Return (parameter, route) pairs overriding the default rule."""
        ...
