"""Stochastic PDE solvers and utilities for exponax."""

from importlib.metadata import version as _distribution_version

from . import nonlin_fun, stepper
from ._diagnostics import (
    richardson_weak_extrapolation,
    stochastic_structure_factor,
    structure_factor,
)
from ._rollout import stochastic_ensemble_rollout, stochastic_rollout
from ._splitting import strang_split_step
from ._warnings import NoiseRegularityWarning, TamingOrderWarning
from .nonlin_fun import TamedPolynomialNonlinearFun
from .stepper import StochasticAllenCahn

__version__ = _distribution_version("exponax-spde")

__all__ = [
    "StochasticAllenCahn",
    "TamedPolynomialNonlinearFun",
    "NoiseRegularityWarning",
    "TamingOrderWarning",
    "stochastic_rollout",
    "stochastic_ensemble_rollout",
    "structure_factor",
    "stochastic_structure_factor",
    "richardson_weak_extrapolation",
    "strang_split_step",
    "stepper",
    "nonlin_fun",
    "__version__",
]

# Migration messages for the old deterministic surface. Remove in v0.4.0.
_REMOVED = {
    "BaseStepper",
    "ForcedStepper",
    "RepeatedStepper",
    "derivative",
    "fft",
    "ifft",
    "get_spectrum",
    "make_grid",
    "rollout",
    "repeat",
    "stack_sub_trajectories",
    "build_ic_set",
    "wrap_bc",
    "metrics",
    "etdrk",
    "ic",
    "viz",
    "spectral",
    "FourierInterpolator",
    "map_between_resolutions",
}


def __getattr__(name: str) -> None:
    if name in _REMOVED:
        raise AttributeError(f"{__name__}.{name} has moved. Use exponax.{name}.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
