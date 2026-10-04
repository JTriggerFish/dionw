"""The routing report: where each parameter went and why."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from dionw.routing import RouteKind

__all__ = ["RouteReason", "RoutedParameter", "RoutingReport"]


class RouteReason(Enum):
    """Why a parameter took its route."""

    DECLARED = "declared"
    EMBEDDING = "embedding"
    GROUPED_CONVOLUTION = "grouped convolution"
    LEARNED_TOKENS = "learned tokens"
    EMBEDDING_NAME = "embedding-like name"
    VECTOR = "vector"
    NARROW = "narrow"
    MATRIX = "matrix"


@dataclass(frozen=True)
class RoutedParameter:
    """One parameter's resolved route.

    Attributes:
        name: Qualified parameter name on the model.
        shape: Parameter shape.
        kind: The update it receives.
        fraction: Row fraction of its blocks (1.0 for AdamW).
        num_heads: Head split of a MATRIX route, or None.
        decayed: Whether it receives the optimizer's weight decay.
        reason: Why it took this route.
    """

    name: str
    shape: tuple[int, ...]
    kind: RouteKind
    fraction: float
    num_heads: int | None
    decayed: bool
    reason: RouteReason


@dataclass(frozen=True)
class RoutingReport:
    """Every routed parameter, in group order."""

    parameters: tuple[RoutedParameter, ...]

    def lines(self) -> list[str]:
        """Return a summary of the routing.

        One line per group, then each >=2D parameter that went to AdamW, with
        its reason.
        """
        lines = ["dionw parameter routing:"]
        lines.extend(self._group_lines())
        lines.extend(
            f"    adamw ({p.reason.value}): {p.name} {p.shape}"
            for p in self.parameters
            if p.kind is RouteKind.ADAMW and len(p.shape) >= 2
        )
        return lines

    def _group_lines(self) -> list[str]:
        """Return one line per group: route, fraction, heads, decay, size."""
        sizes: dict[tuple[RouteKind, float, int | None, bool], list[int]] = {}
        for p in self.parameters:
            key = (p.kind, p.fraction, p.num_heads, p.decayed)
            sizes.setdefault(key, []).append(math.prod(p.shape))
        return [
            f"  {kind.value} fraction={fraction} heads={heads} "
            f"decay={'yes' if decayed else 'no'} tensors={len(numels)} "
            f"params={sum(numels) / 1e6:.2f}M"
            for (kind, fraction, heads, decayed), numels in sizes.items()
        ]
