"""The background worker of CPU EMA: one queued update at a time."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

from dionw.ema._update import FP32, ema_update_

if TYPE_CHECKING:
    import torch
    from torch import Tensor

from dionw.ema._keys import EMA_EVENT, EMA_INITIALIZED, EMA_PENDING

__all__ = ["AsyncCpuEmaItem", "AsyncCpuEmaJob"]


@dataclass
class AsyncCpuEmaItem:
    """One staged CPU update.

    Attributes:
        state: The parameter's optimizer state entry.
        event: The copy's CUDA event, or None for a synchronous copy.
        shadow: The FP32 CPU shadow.
        tmp_cpu: The staged snapshot of the parameter.
    """

    state: dict[str, Any]
    event: torch.cuda.Event | None
    shadow: Tensor
    tmp_cpu: Tensor

    def __post_init__(self) -> None:
        """Check that both tensors live on the CPU.

        Raises:
            ValueError: If the shadow or the staging buffer is on another device.
        """
        if self.shadow.device.type != "cpu" or self.tmp_cpu.device.type != "cpu":
            raise ValueError("CPU EMA shadows and staging buffers must be on the CPU")


class AsyncCpuEmaJob:
    """One queued CPU update with copy-ready and update-done barriers."""

    def __init__(
        self,
        *,
        items: list[AsyncCpuEmaItem],
        decay: float,
        fp32_staging: Tensor | None,
    ) -> None:
        """Create the job.

        Args:
            items: Staged snapshots, applied in order.
            decay: Effective decay of this update.
            fp32_staging: Flat FP32 CPU buffer that fits the largest non-FP32
                snapshot, or None when all are FP32; only this job uses it.

        Raises:
            ValueError: If there are no items or a snapshot does not fit.
        """
        if not items:
            raise ValueError("AsyncCpuEmaJob requires at least one item")
        staged = [item.tmp_cpu.numel() for item in items if item.tmp_cpu.dtype != FP32]
        capacity = 0 if fp32_staging is None else fp32_staging.numel()
        if max(staged, default=0) > capacity:
            raise ValueError("CPU EMA FP32 staging buffer is smaller than a snapshot")
        self.items = items
        self.decay = decay
        self.fp32_staging = fp32_staging
        self.copy_ready = Event()
        self.update_done = Event()
        self.exception: BaseException | None = None
        self._thread = Thread(target=self._run, name="dionw_cpu_ema", daemon=True)

    def start(self) -> None:
        """Start the worker thread."""
        self._thread.start()

    def wait_for_copy(self) -> None:
        """Wait until every staged GPU-to-CPU copy has completed.

        Raises:
            RuntimeError: If the worker failed.
        """
        self.copy_ready.wait()
        self._raise_if_failed()

    def wait_for_update(self) -> None:
        """Wait until the update has been applied.

        Raises:
            RuntimeError: If the worker failed.
        """
        self.update_done.wait()
        self._thread.join()
        self._raise_if_failed()

    def done(self) -> bool:
        """Whether the update has been applied."""
        return self.update_done.is_set()

    def _raise_if_failed(self) -> None:
        if self.exception is not None:
            raise RuntimeError("Asynchronous CPU EMA update failed") from self.exception

    def _run(self) -> None:
        """Wait for the copies, then update every shadow."""
        try:
            for item in self.items:
                if item.event is not None:
                    item.event.synchronize()
            self.copy_ready.set()
            for item in self.items:
                if not item.state[EMA_INITIALIZED]:
                    item.shadow.copy_(item.tmp_cpu)
                    item.state[EMA_INITIALIZED] = True
                else:
                    source = item.tmp_cpu
                    if source.dtype != FP32:
                        if self.fp32_staging is None:
                            raise RuntimeError("CPU EMA FP32 staging buffer missing")
                        staged = self.fp32_staging[: source.numel()].view(source.shape)
                        source = staged.copy_(source)
                    ema_update_(item.shadow, source, decay=self.decay)
                item.state.pop(EMA_EVENT, None)
                item.state[EMA_PENDING] = False
        except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread
            self.exception = exc
            self.copy_ready.set()
        finally:
            self.update_done.set()
