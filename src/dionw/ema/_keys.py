"""Names of the EMA's entries in the optimizer state."""

from __future__ import annotations

from typing import Final

__all__ = [
    "EMA_EVENT",
    "EMA_INITIALIZED",
    "EMA_KEY_PREFIX",
    "EMA_PENDING",
    "EMA_SCRATCH_KEYS",
    "EMA_SHADOW",
    "EMA_STEP_COUNT_KEY",
    "EMA_TMP_CPU",
]

# Top-level state-dict entry: the EMA schedule position.
EMA_STEP_COUNT_KEY: Final[str] = "_ema_step_count"
# Every per-parameter EMA entry starts with this prefix.
EMA_KEY_PREFIX: Final[str] = "ema_"
# The float32 shadow.
EMA_SHADOW: Final[str] = "ema_shadow"
# Whether the shadow has been copied from the live weights yet.
EMA_INITIALIZED: Final[str] = "ema_initialized"
# Whether a CPU update of the shadow is queued.
EMA_PENDING: Final[str] = "ema_pending"
# The pinned CPU staging buffer of CPU EMA.
EMA_TMP_CPU: Final[str] = "ema_tmp_cpu"
# The CUDA event of the staged copy.
EMA_EVENT: Final[str] = "ema_event"
# Working entries rebuilt on load and never saved.
EMA_SCRATCH_KEYS: Final[tuple[str, ...]] = (EMA_PENDING, EMA_TMP_CPU, EMA_EVENT)
