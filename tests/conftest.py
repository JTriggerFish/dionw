"""dionw runs on CUDA only: the suite fails without a GPU instead of skipping."""

from __future__ import annotations

import pytest
import torch


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail fast without CUDA."""
    del session
    if not torch.cuda.is_available():
        raise pytest.UsageError("dionw's tests need a CUDA GPU")
