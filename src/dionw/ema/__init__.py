"""Optimizer-integrated EMA of the parameters, on the GPU or the CPU.

Shadows live in ``optimizer.state[p]`` (``ema_shadow``, float32), so they are
saved, loaded and moved with the optimizer state; the schedule position is the
top-level ``_ema_step_count`` entry. The EMA updates after each optimizer step,
every ``update_every_n_steps`` steps, with decay ``decay ** n`` so the half-life
in steps does not depend on ``n``.

- **GPU shadows** update with tensor-list kernels after the step.
- **CPU shadows** save GPU memory: each due step copies the parameters into
  pinned CPU buffers on a side stream, and a background thread applies the
  update while training continues. At most one CPU update is in flight; the
  next due step, ``state_dict``, a weight swap or ``flush_pending_updates``
  waits for it.

Every shadow update uses ``ema_update_`` (``lerp``): a constant parameter keeps a
bit-exact shadow.

Use ``IntegratedEMAOptimizer`` as a mixin before a concrete optimizer
(``class X(IntegratedEMAOptimizer, torch.optim.AdamW)``) and call
``_init_integrated_ema`` at the end of ``__init__``; ``dionw.Dion`` does this.
"""

from dionw.ema._keys import EMA_STEP_COUNT_KEY
from dionw.ema.config import EMAConfig, EMADevice, ProfileRegion, no_profile
from dionw.ema.mixin import IntegratedEMAOptimizer
from dionw.ema.tracker import OptimizerIntegratedEMA

__all__ = [
    "EMA_STEP_COUNT_KEY",
    "EMAConfig",
    "EMADevice",
    "IntegratedEMAOptimizer",
    "OptimizerIntegratedEMA",
    "ProfileRegion",
    "no_profile",
]
