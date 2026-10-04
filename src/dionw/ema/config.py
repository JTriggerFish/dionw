"""EMA schedule and placement (``EMAConfig``) and the profiling hook type."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import Enum

__all__ = ["EMAConfig", "EMADevice", "ProfileRegion", "no_profile"]

# A named profiling region around EMA work, e.g. a profiler's ``region``.
ProfileRegion = Callable[[str], AbstractContextManager[object]]


def no_profile(name: str) -> AbstractContextManager[object]:
    """The default ``ProfileRegion``: no profiling."""
    del name
    return nullcontext()


class EMADevice(Enum):
    """Where EMA shadows live."""

    GPU = "gpu"
    CPU = "cpu"


@dataclass(frozen=True)
class EMAConfig:
    """EMA schedule and placement.

    Attributes:
        decay: Per-step decay in [0, 1).
        device: Where shadows live.
        pin_memory: Pin CPU shadows and staging buffers (CPU EMA).
        update_every_n_steps: Update every n optimizer steps (>= 1); the decay is
            raised to n.
        start_step: First optimizer step that copies the live weights into the
            shadows; earlier steps leave the EMA untouched. 0 initializes at the
            first update.
        decay_final: End of a linear decay ramp, or None for a constant decay.
        decay_ramp_steps: Optimizer steps to reach ``decay_final``; set with it.

    Raises:
        ValueError: On out-of-range values or a half-specified ramp.
    """

    decay: float = 0.9999
    device: EMADevice = EMADevice.GPU
    pin_memory: bool = True
    update_every_n_steps: int = 1
    start_step: int = 0
    decay_final: float | None = None
    decay_ramp_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate the schedule."""
        if self.update_every_n_steps <= 0:
            raise ValueError("EMAConfig.update_every_n_steps must be positive")
        if self.start_step < 0:
            raise ValueError("EMAConfig.start_step must be non-negative")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMAConfig.decay must be in [0, 1)")
        if (self.decay_final is None) != (self.decay_ramp_steps is None):
            raise ValueError(
                "EMAConfig.decay_final and decay_ramp_steps must be set together"
            )
        if self.decay_final is not None and not 0.0 <= self.decay_final < 1.0:
            raise ValueError("EMAConfig.decay_final must be in [0, 1)")
        if self.decay_ramp_steps is not None and self.decay_ramp_steps <= 0:
            raise ValueError("EMAConfig.decay_ramp_steps must be positive")

    def decay_at_step(self, step_count: int) -> float:
        """Return the per-step decay at an EMA step count.

        The decay ramps linearly from ``decay`` to ``decay_final`` over
        ``decay_ramp_steps``, then holds.

        Raises:
            ValueError: If ``step_count`` is negative.
        """
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        if self.decay_final is None or self.decay_ramp_steps is None:
            return self.decay
        progress = min(1.0, step_count / self.decay_ramp_steps)
        return self.decay + (self.decay_final - self.decay) * progress
