"""``OptimizerIntegratedEMA``: shadows in ``optimizer.state[p]``, GPU or CPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import torch
from torch import Tensor

from dionw.ema._cpu_job import AsyncCpuEmaItem, AsyncCpuEmaJob
from dionw.ema._keys import (
    EMA_EVENT,
    EMA_INITIALIZED,
    EMA_PENDING,
    EMA_SHADOW,
    EMA_TMP_CPU,
)
from dionw.ema._update import FP32, foreach_ema_update_
from dionw.ema.config import EMAConfig, EMADevice, ProfileRegion

if TYPE_CHECKING:
    from torch.optim import Optimizer

__all__ = ["OptimizerIntegratedEMA", "new_cpu_staging"]

# Profiling region names.
_STAGE_REGION: Final[str] = "ema_stage_copy"
_FINALIZE_REGION: Final[str] = "ema_finalize"


def new_cpu_staging(param: Tensor, *, pin: bool) -> Tensor:
    """Return a zero CPU buffer that receives a parameter's snapshot for CPU EMA.

    float32 and bfloat16 parameters are staged as is; other dtypes in float32.
    """
    dtype = param.dtype if param.dtype in (FP32, torch.bfloat16) else FP32
    return torch.zeros_like(param, dtype=dtype, device="cpu", pin_memory=pin)


class OptimizerIntegratedEMA:
    """EMA shadows stored in ``optimizer.state[p]``.

    A shadow is created on the first update after the optimizer has created
    ``p``'s state.
    """

    def __init__(
        self, optimizer: Optimizer, config: EMAConfig, profile_region: ProfileRegion
    ) -> None:
        """Attach to ``optimizer``.

        Args:
            optimizer: The optimizer whose state holds the shadows.
            config: Schedule and placement.
            profile_region: Named profiling region around EMA work.
        """
        self.optimizer = optimizer
        self.config = config
        self._profile_region = profile_region
        match config.device:
            case EMADevice.GPU:
                self._use_gpu_shadow = True
            case EMADevice.CPU:
                self._use_gpu_shadow = False
            case _ as unreachable:
                raise RuntimeError(f"Unhandled EMADevice: {unreachable}")
        self._step_count = 0
        self._do_update = True
        self._initialize_from_live = False
        self._eff_decay: float | None = None
        self._cpu_ema_job: AsyncCpuEmaJob | None = None
        # Reused FP32 upcast buffer for non-FP32 CPU snapshots.
        self._cpu_fp32_staging: Tensor | None = None
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}
        # GPU work queued until finalize_updates, then issued as list kernels.
        self._gpu_copy_sources: list[Tensor] = []
        self._gpu_copy_shadows: list[Tensor] = []
        self._gpu_update_sources: list[Tensor] = []
        self._gpu_update_shadows: list[Tensor] = []

    @property
    def step_count(self) -> int:
        """Optimizer steps seen by the EMA schedule (saved with the state)."""
        return self._step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        """Set the schedule position, e.g. 0 to restart it over seeded shadows.

        Raises:
            ValueError: If ``value`` is negative.
        """
        if value < 0:
            raise ValueError(f"EMA step_count must be non-negative, got {value}")
        self._step_count = value

    @property
    def uses_gpu_shadow(self) -> bool:
        """Whether shadows live on the parameters' GPU (else pinned CPU)."""
        return self._use_gpu_shadow

    def _copy_stream_for(self, device: torch.device) -> torch.cuda.Stream:
        """The dedicated GPU-to-CPU copy stream of a CUDA device."""
        stream = self._copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._copy_streams[device] = stream
        return stream

    def _wait_for_previous_snapshot_copy(self) -> None:
        """Wait until staged copies no longer read the live parameters."""
        job = self._cpu_ema_job
        if job is None:
            return
        job.wait_for_copy()
        if job.done():
            job.wait_for_update()
            self._cpu_ema_job = None

    def _wait_for_previous_cpu_update(self) -> None:
        """Wait for the in-flight CPU update (one at most)."""
        job = self._cpu_ema_job
        if job is None:
            return
        job.wait_for_update()
        self._cpu_ema_job = None

    def flush_pending_updates(self) -> None:
        """Finish queued EMA work before the shadows are read.

        Raises:
            RuntimeError: If the CPU worker failed.
        """
        if self._use_gpu_shadow:
            self._finalize_gpu_updates()
        else:
            self._wait_for_previous_cpu_update()

    @torch.no_grad()
    def _finalize_gpu_updates(self) -> None:
        """Apply queued GPU copies and updates as list kernels (no host sync)."""
        if self._gpu_copy_shadows:
            torch._foreach_copy_(self._gpu_copy_shadows, self._gpu_copy_sources)
            self._gpu_copy_shadows.clear()
            self._gpu_copy_sources.clear()
        if self._gpu_update_shadows:
            decay = self.config.decay if self._eff_decay is None else self._eff_decay
            foreach_ema_update_(
                self._gpu_update_shadows, self._gpu_update_sources, decay=decay
            )
            self._gpu_update_shadows.clear()
            self._gpu_update_sources.clear()

    def begin_step(self) -> None:
        """Advance the schedule at the start of an optimizer step."""
        if not self._use_gpu_shadow:
            self._wait_for_previous_snapshot_copy()
        self._step_count += 1
        n = self.config.update_every_n_steps
        start_step = self.config.start_step
        self._initialize_from_live = start_step > 0 and self._step_count == start_step
        if start_step > 0 and self._step_count < start_step:
            self._do_update = False
        elif self._initialize_from_live:
            self._do_update = True
        else:
            cadence_step = self._step_count - start_step
            self._do_update = cadence_step > 0 and cadence_step % n == 0
        step_decay = self.config.decay_at_step(self._step_count)
        self._eff_decay = (
            None if not self._do_update or self._initialize_from_live else step_decay**n
        )
        if self._do_update and not self._use_gpu_shadow:
            self._wait_for_previous_cpu_update()

    def should_update_this_step(self) -> bool:
        """Whether this step updates the shadows."""
        return self._do_update

    def _create_shadow(self, param: Tensor, state: dict[str, Any]) -> None:
        """A zero FP32 shadow (and a CPU staging buffer for CPU EMA).

        Raises:
            ValueError: If GPU EMA is configured for a CPU parameter.
        """
        if self._use_gpu_shadow and not param.is_cuda:
            raise ValueError("EMA configured for the GPU but a parameter is on the CPU")
        pin = self.config.pin_memory and not self._use_gpu_shadow
        state[EMA_SHADOW] = torch.zeros_like(
            param,
            dtype=FP32,
            device=param.device if self._use_gpu_shadow else torch.device("cpu"),
            pin_memory=pin,
        )
        state[EMA_INITIALIZED] = False
        if not self._use_gpu_shadow:
            state[EMA_TMP_CPU] = new_cpu_staging(param, pin=pin)

    @torch.no_grad()
    def update_param(self, param: Tensor) -> None:
        """Stage one parameter's update for this step.

        ``finalize_updates`` applies it; parameters without optimizer state are
        skipped.
        """
        with self._profile_region(_STAGE_REGION):
            if not self._do_update or param not in self.optimizer.state:
                return
            state = self.optimizer.state[param]
            if EMA_SHADOW not in state:
                self._create_shadow(param, state)
            if self._use_gpu_shadow:
                self._queue_gpu_update(param, state)
            else:
                self._stage_cpu_snapshot(param, state)

    def _queue_gpu_update(self, param: Tensor, state: dict[str, Any]) -> None:
        """Queue a GPU shadow for a copy (first update) or a lerp."""
        shadow = state[EMA_SHADOW]
        if self._initialize_from_live or not state[EMA_INITIALIZED]:
            self._gpu_copy_shadows.append(shadow)
            self._gpu_copy_sources.append(param)
            state[EMA_INITIALIZED] = True
        else:
            self._gpu_update_shadows.append(shadow)
            self._gpu_update_sources.append(param)
        state[EMA_PENDING] = False

    def _stage_cpu_snapshot(self, param: Tensor, state: dict[str, Any]) -> None:
        """Copy the parameter into its pinned CPU buffer on a side stream.

        The CUDA event of the copy is stored for the worker to wait on.
        """
        if self._initialize_from_live:
            state[EMA_INITIALIZED] = False
        tmp_cpu = state[EMA_TMP_CPU]
        event: torch.cuda.Event | None = None
        if param.is_cuda and tmp_cpu.is_pinned():
            copy_stream = self._copy_stream_for(param.device)
            copy_stream.wait_stream(torch.cuda.current_stream(param.device))
            with torch.cuda.stream(copy_stream):
                tmp_cpu.copy_(param, non_blocking=True)
                event = torch.cuda.Event(blocking=False)
                event.record(copy_stream)
        else:
            tmp_cpu.copy_(param, non_blocking=False)
        state[EMA_EVENT] = event
        state[EMA_PENDING] = True

    @torch.no_grad()
    def finalize_updates(self) -> None:
        """Apply this step's GPU updates, or start the CPU worker on them."""
        with self._profile_region(_FINALIZE_REGION):
            if self._use_gpu_shadow:
                self._finalize_gpu_updates()
                return
            if not self._do_update:
                return
            decay = self.config.decay if self._eff_decay is None else self._eff_decay
            items = [
                AsyncCpuEmaItem(
                    state=state,
                    event=state[EMA_EVENT],
                    shadow=state[EMA_SHADOW],
                    tmp_cpu=state[EMA_TMP_CPU],
                )
                for group in self.optimizer.param_groups
                for param in group["params"]
                if (state := self.optimizer.state.get(param))
                and state.get(EMA_PENDING, False)
            ]
            if not items:
                return
            if self._cpu_ema_job is not None:
                raise RuntimeError(
                    "CPU EMA job already pending after the pre-step wait"
                )
            job = AsyncCpuEmaJob(
                items=items, decay=decay, fp32_staging=self._cpu_fp32_staging_for(items)
            )
            self._cpu_ema_job = job
            job.start()

    def _cpu_fp32_staging_for(self, items: list[AsyncCpuEmaItem]) -> Tensor | None:
        """Return the reused FP32 upcast buffer, or None when all are FP32.

        It grows to fit every non-FP32 snapshot; one CPU job at a time uses it.
        """
        need = max(
            (item.tmp_cpu.numel() for item in items if item.tmp_cpu.dtype != FP32),
            default=0,
        )
        if need == 0:
            return None
        staging = self._cpu_fp32_staging
        if staging is None or staging.numel() < need:
            staging = torch.empty(need, dtype=FP32, device="cpu")
            self._cpu_fp32_staging = staging
        return staging
