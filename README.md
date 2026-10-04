# dionw

dionw is a single-GPU PyTorch optimizer that applies Dion to matrix parameters
and AdamW to all other parameters, under a single learning rate. It is intended
as a drop-in replacement for AdamW in existing training scripts.

**Contents:** [Install](#install) · [Quick start](#quick-start) ·
[Defaults](#defaults) · [The update](#the-update) · [Precision](#precision) ·
[Routing](#routing) · [EMA](#ema) ·
[Checkpoints and resume](#checkpoints-and-resume) · [Scope](#scope) ·
[Differences from microsoft/dion](#differences-from-microsoftdion-nordion2-commit-7692479) ·
[Performance](#performance) · [Open questions](#open-questions) ·
[Development](#development) · [References](#references) · [License](#license)

- **Matrices** (linear and convolution weights) get microsoft/dion's NorDion2
  update, also called Dion3 [[1, 2]](#references): row-selected, orthogonalized,
  NorMuon-normalized [[3]](#references) momentum. With `fraction=1` it is
  NorMuon.
- **Everything else** (biases, norms, embeddings, learned tokens) gets torch's
  fused AdamW kernel.
- **Fast Triton kernels on any recent NVIDIA GPU.** Orthogonalization defaults
  to Gram Newton–Schulz, built on Triton symmetric-product kernels and up to
  2.7 times faster than Polar Express on wide blocks; the momentum update runs
  as fused Triton passes. Nothing is tied to the Hopper and Blackwell datacenter
  GPUs that the reference Gram kernels require. With the same row selection, a
  step is 1.7 to 2.3 times faster than microsoft/dion's NorDion2 with its Triton
  Polar Express, and 2.1 times faster than NorDion2 with its CuTeDSL Gram
  kernels on a GH200; with the same kernels on both sides, the implementation
  alone is 1.4 to 2.1 times faster ([Performance](#performance)).
- **One AdamW-equivalent learning rate.** Matrix steps are scaled to AdamW's
  update RMS, so an existing AdamW learning rate, warmup and schedule carry over
  unchanged.
- **An optional EMA of the weights,** kept on the GPU or on the CPU, saved with
  the optimizer state.
- **Automatic routing.** `dionw.param_groups(model)` assigns each parameter to
  one of the two updates, accepts per-module overrides, and reports the reason
  for each assignment.

The algorithm, Polar Express and its symmetric-product kernels come from
[microsoft/dion](https://github.com/microsoft/dion) [[8]](#references); Gram
Newton–Schulz is adapted from Dao-AILab's
[gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz)
[[7]](#references). The test suite checks this implementation against
microsoft/dion step for step. What dionw adds is the single-device
implementation around them: Gram Newton–Schulz on Triton, bucketed state, fused
momentum passes, routing, the EMA and resume checks.

**AI disclosure.** The initial version (0.1.0), including the code, tests and
documentation, was written entirely by Claude Opus 5.5, an AI model from
Anthropic, under the direction and review of the maintainer.

## Install

```bash
pip install "dionw @ git+https://github.com/JTriggerFish/dionw"
```

Requirements: PyTorch 2.13 or newer, a CUDA GPU, and Triton (included in
PyTorch's Linux CUDA wheels). Tested on Blackwell (RTX 5090, RTX PRO 6000), Ada
Lovelace (RTX 4090) and Hopper (GH200, aarch64) GPUs.

## Quick start

```python
import dionw

groups, report = dionw.param_groups(model)  # route every parameter
print("\n".join(report.lines()))  # optional: print the routing
optimizer = dionw.Dion(groups, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)

for batch in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

Parameters must be float32. The usual mixed-precision setup satisfies this:
float32 master weights, with the forward and backward passes under bfloat16
autocast ([Precision](#precision)).

## Defaults

`dionw.Dion(groups, lr=..., ...)`:

| Setting | Default | microsoft/dion NorDion2 | Meaning |
|---|---|---|---|
| `lr` | required | 0.01 | AdamW-equivalent learning rate for every parameter |
| `momentum` | 0.95 | `mu` 0.95 | Matrix momentum; decay of the selected rows (error feedback) |
| `muon_beta2` | 0.95 | 0.95 | EMA decay of each row's squared update (NorMuon) |
| `betas` | (0.9, 0.95) | (0.9, 0.95) | AdamW betas of AdamW-routed parameters |
| `weight_decay` | 0.0 | 0.01 | Decoupled weight decay of decayed groups |
| `eps` | 1e-8 | `epsilon` 1e-8 | AdamW epsilon (Newton–Schulz uses a fixed 1e-8) |
| `newton_schulz` | `NewtonSchulz.GRAM` | Polar Express | Orthogonalization of matrix updates ([Precision](#precision)) |
| `ema` | `None` | — | EMA schedule and placement (`EMAConfig`) |
| learning-rate scaling | `0.2 * sqrt(max(rows, cols))`, fixed | `adjust_lr="spectral_norm"` | See [The update](#the-update) |

`dionw.param_groups(model, ...)`:

| Setting | Default | microsoft/dion NorDion2 | Meaning |
|---|---|---|---|
| `fraction` | 0.25 | 0.25 | Share of rows a block updates per step |
| `selection_min_dim` | 1024 | — (every matrix selects rows) | Smaller side from which a block selects rows |
| `min_matrix_dim` | 8 | — (groups built by hand) | Smaller side below which a tensor takes AdamW |
| `no_weight_decay` | `None` | — | `None`: vectors, learned tokens and embedding-like names skip weight decay |

`dionw.EMAConfig(...)`:

| Setting | Default | Meaning |
|---|---|---|
| `decay` | 0.9999 | Per-step decay |
| `device` | `EMADevice.GPU` | Where the shadows live |
| `pin_memory` | `True` | Pinned CPU buffers (CPU EMA) |
| `update_every_n_steps` | 1 | Update every n steps, with decay `decay ** n` |
| `start_step` | 0 | Step that first copies the live weights into the shadows; 0 copies at the first update |
| `decay_final`, `decay_ramp_steps` | `None` | Optional linear ramp of the decay |

## The update

For each matrix block:

1. `M += G`: momentum accumulates on every row, every step.
2. The top `k = ceil(fraction * rows)` rows by momentum l1 norm are selected,
   and only those rows of `M` are decayed by `momentum`. This is error
   feedback: the other rows keep accumulating until they are selected.
3. The selected rows are orthogonalized with five Polar Express
   [[4]](#references) steps.
4. NorMuon: each row is divided by the root of an EMA (`muon_beta2`) of its
   mean square, and the block's Frobenius norm is restored.
5. `W[rows] -= lr * 0.2 * sqrt(max(rows, cols)) * c * O`, with
   `c = sqrt(min(rows, cols) / min(k, cols))`.

**The `0.2 * sqrt(max(rows, cols))` factor.** An orthogonalized update `O` of
shape `(rows, cols)` has `min(rows, cols)` singular values equal to one, so its
Frobenius norm is `sqrt(min(rows, cols))` and its per-element RMS is
`1/sqrt(max(rows, cols))`. Multiplying by `sqrt(max(rows, cols))` gives every
block an update RMS of one, whatever its shape; the factor 0.2 then matches the
update RMS that AdamW typically reaches in practice (0.2 to 0.4), as proposed
in *Muon is Scalable for LLM Training* [[5]](#references). The same learning
rate, warmup and schedule therefore serve the matrix and the AdamW routes.
microsoft/dion offers this rule as `adjust_lr="rms_norm"`; its default,
`spectral_norm` (`lr * sqrt(fan_out / fan_in)`), targets learning-rate transfer
across model widths instead.

**The compensation `c`.** Updating `k` of a block's rows shrinks the step. The
Dion3 paper [[2]](#references) (Section 8.1) matches the Frobenius norm of a
partial-row step to a full one by dividing the learning rate by `sqrt(f)`, and
microsoft/dion leaves that to the user's learning rate. dionw applies it inside
the optimizer, per block: `c = sqrt(min(rows, cols) / min(k, cols))` is
`1/sqrt(fraction)` for square or wide blocks, and 1 for tall blocks whose
selected rows already span every column.

Weight decay is decoupled and applies to every row:
`W *= 1 - lr * weight_decay`.

A block is a 2D weight, one head of a `num_heads` split, or a
higher-dimensional weight flattened to `(shape[0], rest)`. For a convolution
that is `(out, in * kh * kw)`; for a transposed convolution the rows are its
input channels.

## Precision

- **Parameters and state are float32.** `Dion` rejects other parameter dtypes:
  an AdamW-sized update is below bfloat16's resolution at ordinary learning
  rates, so bfloat16 weights would silently lose most of it. Stochastic rounding
  of bfloat16 parameters could be implemented, but in the authors' runs it still
  trailed float32 parameters over long training, so it is not supported. Momentum,
  the NorMuon row variance, the AdamW moments and the EMA shadows are float32.
- **Updates are applied in float32.** The orthogonalized block comes back in
  bfloat16; the NorMuon normalization, the learning-rate scaling and the weight
  decay run in float32. There is no stochastic rounding.
- **Orthogonalization input.** The momentum rows reach orthogonalization in
  float32, and each method rounds them itself. microsoft/dion rounds them to
  bfloat16 first; on ill-conditioned inputs that costs Gram most of float16's
  advantage (table below).
- **Gram Newton–Schulz (the default)** normalizes its input in float32, then
  iterates in float16, as Dao-AILab's reference does. float16 carries three
  more mantissa bits than bfloat16, and the normalization keeps every value
  within its range. Iterating on the Gram matrix `X Xᵀ` instead of `X` lets
  rounding errors compound, since the iterate is no longer recomputed from `X`;
  the iteration therefore restarts from `X` at its third step, as in the
  reference.
- **Polar Express** (`POLAR_EXPRESS`, `POLAR_EXPRESS_TRITON`) iterates in
  bfloat16, as in microsoft/dion.

Rounding error of each method: the relative Frobenius distance between its
output and the same polynomials evaluated in float64 on the float32 input, for
a 256 × 1024 block (RTX 5090, `python -m bench.precision`; the RTX 4090 and
GH200 give the same figures within 0.01). Momentum matrices are often close to
low rank, which the last columns model.

| Method | Gaussian | Singular values over 4 decades | Over 6 decades | Rank 8, rest at 1e-3 | Rank 8, rest at 1e-4 |
|---|---|---|---|---|---|
| **Gram, float32 input (dionw, the default)** | **0.003** | **0.037** | **0.039** | **0.030** | **0.052** |
| Gram, bfloat16 input (as microsoft/dion passes it) | 0.003 | 0.076 | 0.103 | 0.237 | 0.408 |
| Polar Express, cuBLAS products | 0.042 | 0.061 | 0.086 | 0.163 | 0.321 |
| Polar Express, Triton products | 0.010 | 0.106 | 0.149 | 0.304 | 0.599 |
| Dao-AILab's Gram, restart at the third step | 0.005 | 0.037 | 0.038 | 0.030 | 0.051 |
| Dao-AILab's Gram, no restart | 0.002 | 0.320 | 0.414 | 0.619 | 0.242 |

`tests/test_newton_schulz.py` checks that Gram rounds like Dao-AILab's
reference on every input, and that on the ill-conditioned ones it stays below
0.08 and below Polar Express, where the reference without a restart exceeds
0.1. Rounding error is separate from how far five steps are from the exact
polar factor: by design, singular values far below the largest are not lifted
all the way to one, by any of these methods.

## Routing

`param_groups` sends every trainable parameter to one of the two updates. A
declared route wins. Otherwise the first matching rule applies:

| Parameter | Update |
|---|---|
| `nn.Embedding` / `nn.EmbeddingBag` weights | AdamW |
| Grouped and depthwise convolution weights | AdamW |
| Learned token sets: the parameter's own name ends in `token` or `tokens` (`cls_token`, `register_tokens`), at any count | AdamW |
| The parameter's own name contains `pos_embed`, `position_embedding` or `relative_position_bias` | AdamW |
| Vectors and scalars | AdamW |
| Tensors whose smaller side, flattened to `(shape[0], rest)`, is below `min_matrix_dim` (8) | AdamW |
| Everything else: linear and convolution weights, bare matrix parameters, `nn.MultiheadAttention.in_proj_weight`, transposed convolutions | Dion |

**Narrow tensors.** `min_matrix_dim` compares the smaller side of a tensor's
matrix view, `(shape[0], prod(shape[1:]))`, taken before any `num_heads` split,
against an absolute size, not a ratio. It is there for two cases: layers that
read or write a handful of channels (RGB or small latent inputs and outputs),
which Muon's guidance [[6]](#references) keeps on AdamW, and near-rank-1
tensors, whose polar factor would discard all magnitude information. Larger
first and last layers are not caught by it; declare them AdamW. The rule
follows timm's Muon, whose floor is `min_dim_size = 4`; dionw's default is 8 and it
has no aspect-ratio limit.

**Row fraction.** A block selects `fraction` (default 0.25, as in
microsoft/dion) of its rows per step only when its smaller side is at least
`selection_min_dim` (default 1024). Narrower blocks update every row. They are
cheap to orthogonalize whole, and selecting rows of a low-rank factor or an
attention head would drop rank directions. Narrow blocks include every
attention head, every low-rank factor, and tall matrices whose smaller side is
under `selection_min_dim`. A tall block with both sides at least that size
still selects rows; while its `k` selected rows span every column, the
compensation `c` is 1.

**Declaring routes.** Any module can override the rule for its own parameters
by defining `dion_routes`. No base class is needed:

```python
from dionw import Route, RouteKind


class Attention(nn.Module):
    def dion_routes(self):
        return (
            # Orthogonalize each head of a fused QKV projection on its own.
            (self.qkv.weight, Route(RouteKind.MATRIX, num_heads=3 * self.heads)),
        )


class PatchEmbed(nn.Module):
    def dion_routes(self):
        # Reads pixels: routed to AdamW.
        return ((self.proj.weight, Route(RouteKind.ADAMW)),)
```

Routes can also be passed to `param_groups` as `routes={param: Route(...)}`.
Conflicting declarations raise an error.

**Weight decay.** By default vectors, learned tokens and embedding-like names
are exempt. `no_weight_decay=[...]` replaces that rule with an exact list.
Exempt groups carry `weight_decay=0.0`. Decayed groups inherit the optimizer's
`weight_decay`, so the value is set in one place.

**The report.** It lists every group and every 2D-or-larger parameter that went
to AdamW, with the reason: declared, embedding, grouped convolution, learned
tokens, embedding-like name or narrow. It is also logged at INFO on the `dionw`
logger.

## EMA

```python
optimizer = dionw.Dion(
    groups,
    lr=3e-4,
    ema=dionw.EMAConfig(decay=0.9999, device=dionw.EMADevice.CPU),
)
...
optimizer.swap_ema_weights(model)  # evaluate with the EMA weights
evaluate(model)
optimizer.restore_non_ema_weights()
```

- **Storage.** Shadows are float32 and live in `optimizer.state[p]`, so they are
  saved, loaded and moved with the optimizer state.
- **GPU shadows** update with tensor-list kernels right after the step.
- **CPU shadows** save GPU memory. Each due step copies the weights into pinned
  buffers on a side stream, and a background thread updates the shadows while
  training continues.
- **Swaps are paired.** A second swap, or `step()` while EMA weights are
  swapped in, raises an error.
- **Every parameter has a shadow,** including ones that have not received a
  gradient yet.
- **Schedule options:** `update_every_n_steps` applies `decay ** n` every n
  steps, `start_step` delays the start, and `decay_final` with
  `decay_ramp_steps` ramps the decay linearly.

## Checkpoints and resume

`optimizer.state_dict()` holds the momentum, the variance, the AdamW moments,
the EMA shadows and the EMA step count. CPU EMA's staging buffers are not
saved.

Each group records its route, row fraction, head split and Newton–Schulz kind.
`load_state_dict` raises if any of them differs from the optimizer it loads
into: torch would otherwise silently restore the saved values. The other
hyperparameters (`lr`, `betas`, `momentum`, `muon_beta2`, `weight_decay`,
`eps`) are restored from the checkpoint, as in any torch optimizer; changing
them requires setting them on the groups after loading.

Same-shape blocks of a group share one momentum buffer, and `state[p]` holds
views into it. `torch.save` keeps that sharing, so checkpoints are not
inflated. Formats that reject shared storage, such as safetensors, need the
views cloned first.

`add_param_group` works at any point. Matrix state is repacked on the next
step. `initialize_state(param)` creates a parameter's state before its first
step without updating it, for example before seeding its EMA shadow.

## Scope

- **Tested against AdamW** in VAE and diffusion-model training so far.
- **One process, one CUDA device.**
- **DDP is untested.** Every rank would run the same update on the same
  averaged gradients, but Triton autotuning picks kernel configurations per
  process by timing. Ranks can therefore round differently in bfloat16 and
  drift apart. Nothing re-synchronizes them.
- **FSDP and DTensor are out of scope.**
  [microsoft/dion](https://github.com/microsoft/dion) supports them.
- **float32 parameters only.**

## Differences from microsoft/dion (NorDion2, commit `7692479`)

| | microsoft/dion | dionw |
|---|---|---|
| Scope | FSDP2 / DDP | One GPU |
| LR scaling | `spectral_norm` by default, `lr·sqrt(fan_out/fan_in)`; `rms_norm` optional | Always AdamW-equivalent: `0.2·sqrt(max(rows, cols))` |
| Row-fraction compensation | None | `c = sqrt(min(rows, cols) / min(k, cols))` |
| Which blocks select rows | Every matrix | Blocks whose smaller side is at least `selection_min_dim` |
| Convolutions | Rejected with `flatten=True` | Flattened to `(shape[0], rest)`, rows selected |
| Non-matrix parameters | User-built groups, with `algorithm="adamw"` or `"lion"` | Routed automatically; torch's fused AdamW |
| Newton–Schulz input | Rounded to bfloat16 | float32; each method rounds it ([Precision](#precision)) |
| Gram Newton–Schulz | Optional, through Dao-AILab's `gram-newton-schulz` (CuTeDSL kernels; H100 or B200/B300 only) | The default, on portable Triton kernels: any CUDA GPU Triton supports |
| EMA | None | GPU or CPU, integrated |

**Why microsoft/dion rejects convolutions in NorDion2.** Its row selection
works on a tensor's last two dimensions, taken before any flattening. For a
convolution weight `(out, in, kh, kw)` those are the kernel axes, so the top-k
would rank kernel slices instead of output channels and the error-feedback
decay would land on the wrong axes; its FSDP communication path derives its
dimensions the same way. It therefore raises an error for flattened 3D+
parameters and points convolutions to Muon or NorMuon. dionw selects rows after
flattening the weight to `(out, in * kh * kw)`, so the selected rows are output
channels, and as a single-device optimizer it has no sharded layout to keep
consistent. At `fraction=1` a flattened convolution step equals
`NorMuon(flatten=True)` to 1e-7 (`tests/test_optimizer.py`).

With an exact polar factor in place of Newton–Schulz, applied to the same
bfloat16-rounded input, a dionw step equals a NorDion2 or `NorMuon(flatten=True)`
step to 1e-7, with and without weight decay. The test passes NorDion2 the learning rate times `c`, and the weight
decay divided by `c`, since NorDion2 decays at its own learning rate. The copied
Polar Express functions and Triton kernels match the package's within float
tolerance.

## Performance

Optimizer step only, median over 20 steps after warm-up, measured with
`python -m bench.optimizer_step`: the parameters of a 1.34B-parameter transformer
(width 2048, depth 24, 16 heads, 32k vocabulary) with random gradients. Every
variant steps the same float32 parameters and float32 gradients (the only
parameter dtype dionw accepts); each orthogonalization rounds internally as
described in [Precision](#precision). PyTorch 2.13.0, Triton 3.7.1.

| Variant (ms) | RTX 4090 | RTX 5090 | GH200 |
|---|---|---|---|
| Fused AdamW, all parameters | 41.3 | 25.0 | 11.5 |
| microsoft/dion NorDion2, f=0.25, every block (Polar Express Triton) | 141.5 | 87.9 | 58.5 |
| microsoft/dion NorDion2, f=0.25, every block (Gram, CuTeDSL) | — | — | 55.1 |
| dionw, f=0.25, every block, Polar Express Triton (same kernels) | 89.2 | 61.7 | 27.3 |
| dionw, f=0.25, every block, Gram | 79.0 | 52.8 | 26.0 |
| **dionw, f=0.25, default rule (the default)** | **92.5** | **61.0** | **27.2** |
| dionw, f=0.5, default rule | 156.6 | 102.2 | 45.2 |

NorDion2's Gram Newton–Schulz uses Dao-AILab's CuTeDSL kernels, which need an
H100- or B200-class GPU, so that row is measured on the GH200 only
(`--with-cutedsl-gram`).

Host dispatch of a default step on the benchmark model takes 11 ms on the
RTX 4090 machine, 7 ms on the RTX 5090 machine and 17 ms on the GH200, each
less than the step's GPU time (`python -m bench.host_dispatch`). Without CPU
EMA, no step synchronizes with the GPU. With CPU EMA, the step after a due step
waits until that step's parameter copies have finished, and a due step waits
until the previous CPU update has finished.

- **`NewtonSchulz.GRAM`, the default,** iterates on the small Gram matrix of
  wide blocks, and is faster there than Polar Express on every GPU measured
  (`python -m bench.newton_schulz`; float32 input batches, speedup range over
  four wide block shapes, below). Its rounding error is in [Precision](#precision). It runs on Triton
  kernels, so the speedup is not limited to the Hopper and Blackwell datacenter
  GPUs that the reference implementation's CuTeDSL kernels require.
- **`NewtonSchulz.POLAR_EXPRESS_TRITON`** is microsoft/dion's Polar Express
  with Triton symmetric products.
- **`NewtonSchulz.POLAR_EXPRESS`** uses cuBLAS products.

| Gram's speedup on wide blocks | RTX 4090 | RTX 5090 | GH200 |
|---|---|---|---|
| Over Polar Express, Triton products | 1.2–1.9× | 1.3–2.7× | 1.05–1.8× |
| Over Polar Express, cuBLAS products | 1.5–2.5× | 1.9–3.5× | 1.1–2.2× |

The compiled functions use one static graph per block shape: on DiT-XL-like
[[9]](#references) blocks, orthogonalization and row normalization run 1.35
(RTX 4090) to 2.3 (GH200) times faster than with one dynamic-shape graph
(`python -m bench.static_vs_dynamic`). A model with many shapes would exceed torch's
default recompile limit of 8, so Dion raises the limit around its own compiled
calls only; the global setting is left unchanged.

## Open questions

The following design choices would benefit from external review:

- **The compensation factor `c`** (defined in [The update](#the-update)). The
  Dion3 paper's transfer rule is a blanket `1/sqrt(fraction)` on the learning
  rate. dionw makes the factor
  shape-dependent, so tall blocks whose `k` is at least `cols` get none. Does
  that hold up in training as well as in the norm argument?
- **Row selection only on blocks at least 1024 wide.** Is that threshold right
  outside the VAE and diffusion models it has been tested on?
- **Stale variance.** Rows that go unselected for many steps keep an old
  NorMuon variance. microsoft/dion behaves the same way.

## Development

```bash
pip install -e . --group test --group dev   # pip >= 25.1, or uv pip
pre-commit install
pytest                          # skips tests marked slow or bench
pytest -m slow                  # compile-heavy tests
pytest -m bench                 # the performance claims; needs an idle GPU
pre-commit run --all-files      # hygiene, ruff check and format, ty
```

Every test needs a CUDA GPU. The precision claims are ordinary tests. The
performance claims are the `bench` tests, which time the scripts in `bench/`;
each script also prints its table (`python -m bench.<name>`, from the
repository root).

The `test` dependency group installs two references: microsoft/dion at commit
`7692479` for the parity tests, and Dao-AILab's `gram-newton-schulz` (its
PyTorch backend, which runs on any GPU) for the Gram tests.

## References

1. K. Ahn, B. Xu, N. Abreu, Y. Fan et al. *Dion: Distributed Orthonormalized
   Updates.* 2025. [arXiv:2504.05295](https://arxiv.org/abs/2504.05295)
2. N. Amsel, J. Zhang, K. Ahn, A. Naeimi et al. *Dion3: Full-Stack Orthogonal
   Updates.* 2026. [arXiv:2608.11612](https://arxiv.org/abs/2608.11612)
3. Z. Li, L. Liu, C. Liang, W. Chen et al. *NorMuon: Making Muon more efficient
   and scalable.* 2025. [arXiv:2510.05491](https://arxiv.org/abs/2510.05491)
4. N. Amsel, D. Persson, C. Musco, R. M. Gower. *The Polar Express: Optimal
   Matrix Sign Methods and Their Application to the Muon Algorithm.* 2025.
   [arXiv:2505.16932](https://arxiv.org/abs/2505.16932)
5. J. Liu, J. Su, X. Yao, Z. Jiang et al. *Muon is Scalable for LLM Training.*
   2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)
6. K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein.
   *Muon: An optimizer for hidden layers in neural networks.* 2024.
   [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
7. J. Zhang, N. Amsel, B. Chen, T. Dao. *Gram Newton-Schulz.*
   [github.com/Dao-AILab/gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz)
8. Microsoft. *Dion optimizer implementations.*
   [github.com/microsoft/dion](https://github.com/microsoft/dion)
9. W. Peebles, S. Xie. *Scalable Diffusion Models with Transformers.* 2022.
   [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) (the DiT-XL shapes in
   `bench/static_vs_dynamic.py` and `bench/newton_schulz.py`)

## License

MIT. `src/dionw/_polar_express.py` and `src/dionw/_symmetric_kernels.py`
contain code from microsoft/dion (MIT); `src/dionw/_gram.py` adapts Dao-AILab's
Gram Newton–Schulz, declared MIT. Details are in `LICENSE`.
