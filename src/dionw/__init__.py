"""dionw: single-GPU Dion on matrices, AdamW on the rest, one learning rate.

Dion here is row-selected NorMuon; the learning rate is AdamW-equivalent for
every parameter, and an EMA of the weights is integrated in the optimizer.

Usage::

    import dionw

    groups, report = dionw.param_groups(model)
    optimizer = dionw.Dion(groups, lr=3e-4, weight_decay=0.05)
"""

from importlib.metadata import version

from dionw._keys import (
    FRACTION_KEY,
    MOMENTUM_FORM_KEY,
    NEWTON_SCHULZ_KEY,
    NUM_HEADS_KEY,
    ROUTE_KEY,
    ROW_NORMALIZATION_KEY,
)
from dionw.ema import EMAConfig, EMADevice, IntegratedEMAOptimizer
from dionw.matrix_update import (
    DION3_UPDATE,
    MUON_UPDATE,
    MatrixUpdate,
    MomentumForm,
    RowNormalization,
)
from dionw.newton_schulz import NewtonSchulz
from dionw.optimizer import Dion
from dionw.param_groups import ParamGroups, param_groups
from dionw.report import RoutedParameter, RouteReason, RoutingReport
from dionw.routing import Route, RouteKind, RouteProvider

__all__ = [
    "DION3_UPDATE",
    "FRACTION_KEY",
    "MOMENTUM_FORM_KEY",
    "MUON_UPDATE",
    "NEWTON_SCHULZ_KEY",
    "NUM_HEADS_KEY",
    "ROUTE_KEY",
    "ROW_NORMALIZATION_KEY",
    "Dion",
    "EMAConfig",
    "EMADevice",
    "IntegratedEMAOptimizer",
    "MatrixUpdate",
    "MomentumForm",
    "NewtonSchulz",
    "ParamGroups",
    "Route",
    "RouteKind",
    "RouteProvider",
    "RouteReason",
    "RoutedParameter",
    "RoutingReport",
    "RowNormalization",
    "__version__",
    "param_groups",
]

__version__ = version("dionw")
