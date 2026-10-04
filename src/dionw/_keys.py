"""Names of Dion's param-group and optimizer-state entries.

torch's own group keys (``params``, ``lr``, ``betas``, ``eps``,
``weight_decay``) are used as torch spells them.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Final

from torch import Tensor

__all__ = [
    "ADAMW_STEP_STATE",
    "EXP_AVG_SQ_STATE",
    "EXP_AVG_STATE",
    "FRACTION_KEY",
    "MOMENTUM_KEY",
    "MOMENTUM_STATE",
    "MUON_BETA2_KEY",
    "NEWTON_SCHULZ_KEY",
    "NUM_HEADS_KEY",
    "ROUTE_KEY",
    "STRUCTURAL_KEYS",
    "VARIANCE_STATE",
    "ParamState",
]

# An optimizer's per-parameter state.
ParamState = MutableMapping[Tensor, dict[str, Any]]

# --- param-group keys ---------------------------------------------------------
# The resolved route (a RouteKind value).
ROUTE_KEY: Final[str] = "dion_route"
# The orthogonalization (a NewtonSchulz value).
NEWTON_SCHULZ_KEY: Final[str] = "newton_schulz"
# Share of each block's rows updated per step.
FRACTION_KEY: Final[str] = "fraction"
# Head split of a matrix group's blocks, or None.
NUM_HEADS_KEY: Final[str] = "num_heads"
# Matrix momentum (error-feedback decay of selected rows).
MOMENTUM_KEY: Final[str] = "momentum"
# EMA decay of each row's squared update (NorMuon).
MUON_BETA2_KEY: Final[str] = "muon_beta2"
# Keys fixed when the groups are built; load_state_dict refuses a change.
STRUCTURAL_KEYS: Final[tuple[str, ...]] = (
    ROUTE_KEY,
    FRACTION_KEY,
    NUM_HEADS_KEY,
    NEWTON_SCHULZ_KEY,
)

# --- per-parameter state keys -------------------------------------------------
MOMENTUM_STATE: Final[str] = "momentum"
VARIANCE_STATE: Final[str] = "variance"
EXP_AVG_STATE: Final[str] = "exp_avg"
EXP_AVG_SQ_STATE: Final[str] = "exp_avg_sq"
# Not "step": torch's load_state_dict would move a "step" entry to the CPU.
ADAMW_STEP_STATE: Final[str] = "adamw_step"
