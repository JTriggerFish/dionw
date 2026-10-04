"""param_groups: the default rule, declarations, row fractions, weight decay."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor, nn

from dionw import ROUTE_KEY, Route, RouteKind, RouteReason, param_groups

if TYPE_CHECKING:
    from collections.abc import Iterable

MATRIX, ADAMW = RouteKind.MATRIX, RouteKind.ADAMW


class _Shapes(nn.Module):
    """One parameter per default-rule outcome."""

    def __init__(self) -> None:
        super().__init__()
        self.bare = nn.Parameter(torch.zeros(32, 48))
        self.mha = nn.MultiheadAttention(32, 4, batch_first=True)
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv = nn.Conv2d(32, 16, 3)
        self.grouped = nn.Conv2d(32, 32, 3, groups=4)
        self.embed = nn.Embedding(64, 32)
        self.pos_embed = nn.Parameter(torch.zeros(16, 32))
        self.register_tokens = nn.Parameter(torch.zeros(16, 32))
        self.narrow = nn.Linear(32, 4)
        self.norm = nn.LayerNorm(32)
        # Name rules read a parameter's own name, not its module path.
        self.pos_embedder = nn.Linear(32, 32, bias=False)


def test_default_rule_outcomes() -> None:
    model = _Shapes().cuda()
    _, report = param_groups(
        model, fraction=0.5, selection_min_dim=1024, min_matrix_dim=8
    )
    assert {p.name: (p.kind, p.reason) for p in report.parameters} == {
        "bare": (MATRIX, RouteReason.MATRIX),
        "mha.in_proj_weight": (MATRIX, RouteReason.MATRIX),
        "mha.in_proj_bias": (ADAMW, RouteReason.VECTOR),
        "mha.out_proj.weight": (MATRIX, RouteReason.MATRIX),
        "mha.out_proj.bias": (ADAMW, RouteReason.VECTOR),
        "up.weight": (MATRIX, RouteReason.MATRIX),
        "up.bias": (ADAMW, RouteReason.VECTOR),
        "conv.weight": (MATRIX, RouteReason.MATRIX),
        "conv.bias": (ADAMW, RouteReason.VECTOR),
        "grouped.weight": (ADAMW, RouteReason.GROUPED_CONVOLUTION),
        "grouped.bias": (ADAMW, RouteReason.VECTOR),
        "embed.weight": (ADAMW, RouteReason.EMBEDDING),
        "pos_embed": (ADAMW, RouteReason.EMBEDDING_NAME),
        "register_tokens": (ADAMW, RouteReason.LEARNED_TOKENS),
        "narrow.weight": (ADAMW, RouteReason.NARROW),
        "narrow.bias": (ADAMW, RouteReason.VECTOR),
        "norm.weight": (ADAMW, RouteReason.VECTOR),
        "norm.bias": (ADAMW, RouteReason.VECTOR),
        "pos_embedder.weight": (MATRIX, RouteReason.MATRIX),
    }


class _Attention(nn.Module):
    """Declares routes without inheriting anything (RouteProvider is a Protocol)."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(64, 192, bias=False)
        self.patch = nn.Linear(48, 64, bias=False)

    def dion_routes(self) -> Iterable[tuple[Tensor, Route]]:
        return (
            (self.qkv.weight, Route(RouteKind.MATRIX, num_heads=6, fraction=None)),
            (self.patch.weight, Route(RouteKind.ADAMW, num_heads=None, fraction=None)),
        )


def test_declarations_route_and_split_heads() -> None:
    model = _Attention().cuda()
    groups, report = param_groups(
        model, fraction=0.5, selection_min_dim=1024, min_matrix_dim=8
    )
    by_name = {p.name: p for p in report.parameters}
    assert by_name["qkv.weight"].num_heads == 6
    assert by_name["qkv.weight"].reason is RouteReason.DECLARED
    assert by_name["patch.weight"].kind is ADAMW
    matrix = next(g for g in groups if g[ROUTE_KEY] == RouteKind.MATRIX.value)
    assert matrix["num_heads"] == 6
    assert matrix["params"] == [model.qkv.weight]


def test_routes_argument_merges_and_conflicts_raise() -> None:
    model = _Attention().cuda()
    same = {model.qkv.weight: Route(RouteKind.MATRIX, num_heads=6, fraction=None)}
    assert param_groups(
        model, fraction=0.5, selection_min_dim=1024, min_matrix_dim=8, routes=same
    ).report.parameters
    clash = {model.qkv.weight: Route(RouteKind.ADAMW, num_heads=None, fraction=None)}
    with pytest.raises(ValueError, match="Conflicting routes"):
        param_groups(
            model, fraction=0.5, selection_min_dim=1024, min_matrix_dim=8, routes=clash
        )
    stranger = nn.Parameter(torch.zeros(8, 8, device="cuda"))
    with pytest.raises(ValueError, match="not registered"):
        param_groups(
            model,
            fraction=0.5,
            selection_min_dim=1024,
            min_matrix_dim=8,
            routes={stranger: Route(RouteKind.ADAMW, num_heads=None, fraction=None)},
        )


class _Fractions(nn.Module):
    """A wide, a narrow and a declared matrix."""

    def __init__(self) -> None:
        super().__init__()
        self.wide = nn.Linear(64, 96, bias=False)
        self.narrow = nn.Linear(16, 96, bias=False)
        self.declared = nn.Linear(64, 64, bias=False)


def test_row_fraction_applies_from_selection_min_dim() -> None:
    """Row selection applies from selection_min_dim; a declared fraction wins."""
    model = _Fractions().cuda()
    routes = {
        model.declared.weight: Route(RouteKind.MATRIX, num_heads=None, fraction=1.0)
    }
    _, report = param_groups(
        model, fraction=0.25, selection_min_dim=32, min_matrix_dim=8, routes=routes
    )
    fractions = {p.name: p.fraction for p in report.parameters}
    assert fractions == {
        "wide.weight": 0.25,
        "narrow.weight": 1.0,
        "declared.weight": 1.0,
    }


def test_weight_decay_groups_inherit_the_optimizer_value() -> None:
    """Decayed groups inherit the optimizer's weight decay.

    They carry no weight_decay key; exempt groups carry 0.0. no_weight_decay
    replaces the default rule.
    """
    model = _Shapes().cuda()
    groups, report = param_groups(
        model, fraction=0.5, selection_min_dim=1024, min_matrix_dim=8
    )
    decayed = {p.name for p in report.parameters if p.decayed}
    assert {"bare", "embed.weight", "conv.weight"} <= decayed
    assert {"norm.weight", "conv.bias", "pos_embed", "register_tokens"}.isdisjoint(
        decayed
    )
    names = {id(p): n for n, p in model.named_parameters()}
    for group in groups:
        group_decayed = {names[id(p)] in decayed for p in group["params"]}
        assert len(group_decayed) == 1
        if group_decayed == {True}:
            assert "weight_decay" not in group
        else:
            assert group["weight_decay"] == 0.0
    _, explicit = param_groups(
        model,
        fraction=0.5,
        selection_min_dim=1024,
        min_matrix_dim=8,
        no_weight_decay=[model.bare],
    )
    assert {p.name for p in explicit.parameters if not p.decayed} == {"bare"}


def test_groups_order_adamw_first_and_skip_frozen_parameters() -> None:
    model = _Shapes().cuda()
    model.conv.requires_grad_(False)
    groups, report = param_groups(
        model, fraction=0.5, selection_min_dim=1024, min_matrix_dim=8
    )
    kinds = [RouteKind(g[ROUTE_KEY]) for g in groups]
    first_matrix = kinds.index(RouteKind.MATRIX)
    assert set(kinds[:first_matrix]) == {RouteKind.ADAMW}
    assert set(kinds[first_matrix:]) == {RouteKind.MATRIX}
    assert not any(p.name.startswith("conv.") for p in report.parameters)
    lines = report.lines()
    assert lines[0] == "dionw parameter routing:"
    assert "    adamw (learned tokens): register_tokens (16, 32)" in lines


def test_invalid_settings_raise() -> None:
    model = _Shapes().cuda()
    with pytest.raises(ValueError, match="fraction"):
        param_groups(model, fraction=0.0, selection_min_dim=1024, min_matrix_dim=8)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="No parameters require gradients"):
        param_groups(model, fraction=0.5, selection_min_dim=1024, min_matrix_dim=8)
    with pytest.raises(ValueError, match="MATRIX routes"):
        Route(RouteKind.ADAMW, num_heads=2, fraction=None)
