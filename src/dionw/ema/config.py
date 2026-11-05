"""EMA schedule and placement (``EMAConfig``) and the profiling hook type."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import Enum

__all__ = ["EMAConfig", "EMADevice", "EMASchedule", "ProfileRegion", "no_profile"]

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


class EMASchedule(Enum):
    """How the per-step decay evolves (``EMAConfig.schedule``)."""

    # ``decay`` at every step.
    CONSTANT = "constant"
    # ``decay`` ramped linearly to ``decay_final`` over ``decay_ramp_steps``.
    LINEAR_RAMP = "linear_ramp"
    # Karras et al.'s power function (EDM2), ``(1 - 1/n) ** (power_gamma + 1)``
    # at EMA step n, capped at ``decay``: the average covers a fixed fraction of
    # the steps so far (relative width ``sigma_rel``; gamma 6.94 is 0.10).
    POWER = "power"


@dataclass(frozen=True)
class EMAConfig:
    """EMA schedule and placement.

    Attributes:
        decay: Per-step decay in [0, 1); with ``power_gamma``, the power
            schedule's cap.
        device: Where shadows live.
        pin_memory: Pin CPU shadows and staging buffers (CPU EMA).
        update_every_n_steps: Update every n optimizer steps (>= 1); the decay is
            raised to n.
        start_step: First optimizer step that copies the live weights into the
            shadows; earlier steps leave the EMA untouched. 0 initializes at the
            first update.
        decay_final: End of a linear decay ramp, or None for a constant decay.
        decay_ramp_steps: Optimizer steps to reach ``decay_final``; set with it.
        power_gamma: Exponent gamma (> 0) of the power schedule, or None.
            Exclusive with the ramp.

    Raises:
        ValueError: On out-of-range values, a half-specified ramp, or a ramp
            combined with ``power_gamma``.
    """

    decay: float = 0.9999
    device: EMADevice = EMADevice.GPU
    pin_memory: bool = True
    update_every_n_steps: int = 1
    start_step: int = 0
    decay_final: float | None = None
    decay_ramp_steps: int | None = None
    power_gamma: float | None = None

    def __post_init__(self) -> None:
        """Validate the schedule."""
        if self.update_every_n_steps <= 0:
            raise ValueError("EMAConfig.update_every_n_steps must be positive")
        if self.start_step < 0:
            raise ValueError("EMAConfig.start_step must be non-negative")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("EMAConfig.decay must be in [0, 1)")
        self._check_ramp()
        self._check_power()

    def _check_ramp(self) -> None:
        """The linear ramp is set whole and in range.

        Raises:
            ValueError: Otherwise.
        """
        if (self.decay_final is None) != (self.decay_ramp_steps is None):
            raise ValueError(
                "EMAConfig.decay_final and decay_ramp_steps must be set together"
            )
        if self.decay_final is not None and not 0.0 <= self.decay_final < 1.0:
            raise ValueError("EMAConfig.decay_final must be in [0, 1)")
        if self.decay_ramp_steps is not None and self.decay_ramp_steps <= 0:
            raise ValueError("EMAConfig.decay_ramp_steps must be positive")

    def _check_power(self) -> None:
        """``power_gamma`` is positive and not combined with the ramp.

        Raises:
            ValueError: Otherwise.
        """
        if self.power_gamma is None:
            return
        if self.decay_final is not None:
            raise ValueError(
                "EMAConfig.power_gamma and the decay_final ramp are exclusive"
            )
        if not self.power_gamma > 0.0:
            raise ValueError("EMAConfig.power_gamma must be positive")

    @property
    def schedule(self) -> EMASchedule:
        """The decay schedule the set fields select."""
        if self.power_gamma is not None:
            return EMASchedule.POWER
        if self.decay_final is not None:
            return EMASchedule.LINEAR_RAMP
        return EMASchedule.CONSTANT

    def decay_at_step(self, step_count: int) -> float:
        """Return the per-step decay at an EMA step count (``schedule``).

        Raises:
            ValueError: If ``step_count`` is negative.
        """
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        match (self.power_gamma, self.decay_final, self.decay_ramp_steps):
            case (None, None, None):
                return self.decay
            case (None, float() | int() as final, int() as ramp_steps):
                progress = min(1.0, step_count / ramp_steps)
                return self.decay + (final - self.decay) * progress
            case (float() | int() as gamma, None, None):
                if step_count == 0:
                    return 0.0
                return min(self.decay, (1.0 - 1.0 / step_count) ** (gamma + 1.0))
            case unsupported:
                raise ValueError(f"Unsupported EMA schedule fields {unsupported}")
