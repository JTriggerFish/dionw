"""Time one optimizer step: dionw against fused AdamW and microsoft/dion's NorDion2.

The parameters are those of a generic pre-norm transformer (token embedding,
fused QKV, attention projection, 4x MLP, LayerNorms, output head) with random
gradients; only ``optimizer.step()`` is timed (median of CUDA-event timings
after warm-up, compilation and autotuning included in the warm-up).

Variants:
- fused AdamW on every parameter (``torch.optim.AdamW(fused=True)``);
- microsoft/dion NorDion2 at fraction 0.25, every matrix selecting rows, QKV
  split per head, the rest on its AdamW: with its Triton Polar Express, and,
  with ``--with-cutedsl-gram``, with Dao-AILab's CuTeDSL Gram Newton-Schulz
  (needs the ``gram-newton-schulz`` package and an H100- or B200-class GPU);
- dionw at fraction 0.25 with the same row selection (every block selects
  rows), with NorDion2's Polar Express Triton kernels and with Gram;
- dionw at fraction 0.25 and 0.5 with the default rule (row selection from 1024).

Usage:
    python -m bench.optimizer_step [--width 2048] [--depth 24] [--heads 16]
        [--with-cutedsl-gram]
"""

from __future__ import annotations

import argparse
import gc
from enum import Enum
from typing import TYPE_CHECKING

import torch
from dion import NorDion2
from torch import Tensor, nn

import dionw
from bench._timing import median_ms

if TYPE_CHECKING:
    from collections.abc import Iterable

LR = 1e-4
BETAS = (0.9, 0.95)
VOCAB = 32000


class Variant(Enum):
    """The optimizers compared."""

    ADAMW = "fused AdamW, all parameters"
    NORDION2 = "microsoft/dion NorDion2, f=0.25, every block, Polar Express Triton"
    NORDION2_GRAM = "microsoft/dion NorDion2, f=0.25, every block, Gram (CuTeDSL)"
    DIONW_SAME_KERNELS = "dionw, f=0.25, every block, Polar Express Triton"
    DIONW_EVERY_BLOCK = "dionw, f=0.25, every block, Gram"
    DIONW_DEFAULT_025 = "dionw, f=0.25, default rule"
    DIONW_DEFAULT_050 = "dionw, f=0.5, default rule"


class _Block(nn.Module):
    """One pre-norm transformer block's parameters."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def dion_routes(self) -> tuple[tuple[Tensor, dionw.Route], ...]:
        """Orthogonalize each head of the fused QKV projection on its own."""
        return (
            (
                self.qkv.weight,
                dionw.Route(dionw.RouteKind.MATRIX, num_heads=3 * self.heads),
            ),
        )


class _Transformer(nn.Module):
    """Token embedding, blocks, final norm and output head."""

    def __init__(self, width: int, depth: int, heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, width)
        self.blocks = nn.ModuleList(_Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)


def transformer(width: int, depth: int, heads: int) -> _Transformer:
    """Return the model on the GPU with a random gradient on every parameter."""
    model = _Transformer(width, depth, heads).cuda()
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 1e-3
    return model


def _blocks(model: _Transformer) -> list[_Block]:
    """Return the model's blocks, typed."""
    return [block for block in model.blocks if isinstance(block, _Block)]


def _nordion2(model: _Transformer, heads: int, *, gram: bool) -> torch.optim.Optimizer:
    """NorDion2 on every linear weight (QKV per head); AdamW on the rest.

    ``gram`` selects its CuTeDSL Gram Newton-Schulz over Triton Polar Express.
    """
    blocks = _blocks(model)
    qkv = [block.qkv.weight for block in blocks]
    matrices = [
        m.weight for block in blocks for m in (block.proj, block.fc1, block.fc2)
    ] + [model.head.weight]
    matrix_ids = {id(p) for p in qkv + matrices}
    rest = [p for p in model.parameters() if id(p) not in matrix_ids]
    return NorDion2(
        [
            {"params": matrices},
            {"params": qkv, "num_heads": 3 * heads},
            {"params": rest, "algorithm": "adamw"},
        ],
        lr=LR,
        fraction=0.25,
        betas=BETAS,
        weight_decay=0.0,
        adjust_lr="rms_norm",
        use_triton=True,
        use_gram_newton_schulz=gram,
    )


def _dionw(
    model: _Transformer,
    fraction: float,
    selection_min_dim: int,
    newton_schulz: dionw.NewtonSchulz,
) -> dionw.Dion:
    """Return Dion built from the default routing."""
    groups, _ = dionw.param_groups(
        model,
        fraction=fraction,
        selection_min_dim=selection_min_dim,
        min_matrix_dim=8,
    )
    return dionw.Dion(
        groups,
        lr=LR,
        betas=BETAS,
        weight_decay=0.0,
        newton_schulz=newton_schulz,
    )


def build(variant: Variant, model: _Transformer, heads: int) -> torch.optim.Optimizer:
    """Return the optimizer of ``variant`` for ``model``."""
    match variant:
        case Variant.ADAMW:
            return torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, fused=True)
        case Variant.NORDION2:
            return _nordion2(model, heads, gram=False)
        case Variant.NORDION2_GRAM:
            return _nordion2(model, heads, gram=True)
        case Variant.DIONW_SAME_KERNELS:
            return _dionw(model, 0.25, 1, dionw.NewtonSchulz.POLAR_EXPRESS_TRITON)
        case Variant.DIONW_EVERY_BLOCK:
            return _dionw(model, 0.25, 1, dionw.NewtonSchulz.GRAM)
        case Variant.DIONW_DEFAULT_025:
            return _dionw(model, 0.25, 1024, dionw.NewtonSchulz.GRAM)
        case Variant.DIONW_DEFAULT_050:
            return _dionw(model, 0.5, 1024, dionw.NewtonSchulz.GRAM)
        case _ as unreachable:
            raise RuntimeError(f"Unhandled variant: {unreachable}")


def step_ms(variant: Variant, width: int, depth: int, heads: int) -> float:
    """Return the median ``optimizer.step()`` time of ``variant`` in ms.

    The model and optimizer are freed before returning.
    """
    torch.manual_seed(0)
    model = transformer(width, depth, heads)
    optimizer = build(variant, model, heads)
    elapsed = median_ms(optimizer.step)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed


def _param_count(params: Iterable[Tensor]) -> int:
    return sum(p.numel() for p in params)


def main() -> None:
    """Print one Markdown table row per variant."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument(
        "--with-cutedsl-gram",
        action="store_true",
        help="also time NorDion2 with Dao-AILab's CuTeDSL Gram Newton-Schulz",
    )
    args = parser.parse_args()
    name = torch.cuda.get_device_name()
    print(
        f"{name}, transformer width {args.width}, depth {args.depth}, heads {args.heads}"
    )
    model = transformer(args.width, args.depth, args.heads)
    print(f"parameters: {_param_count(model.parameters()) / 1e9:.2f}B")
    del model
    print("| Variant | ms |\n|---|---|")
    for variant in Variant:
        if variant is Variant.NORDION2_GRAM and not args.with_cutedsl_gram:
            continue
        elapsed = step_ms(variant, args.width, args.depth, args.heads)
        print(f"| {variant.value} | {elapsed:.1f} |", flush=True)


if __name__ == "__main__":
    main()
