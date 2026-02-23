"""
exponax-spde
============
Stochastic PDE solvers built on top of `exponax`.

Public API
----------
StochasticAllenCahn
    Exponential Euler-Maruyama (EEM) integrator for the stochastic
    Allen-Cahn SPDE with additive or multiplicative Q-Wiener noise,
    in 1-D, 2-D, and 3-D on periodic domains.

TamedPolynomialNonlinearFun
    Drop-in replacement for ``exponax.nonlin_fun.PolynomialNonlinearFun``
    with optional Hutzenthaler-Jentzen taming; useful for any SPDE whose
    nonlinear term grows super-linearly (polynomial order ≥ 3).

Stochastic utilities (also available as ``spde.<fn>``):
    stochastic_rollout
    stochastic_ensemble_rollout
    structure_factor
    richardson_weak_extrapolation
    strang_split_step

Usage
-----
    import exponax_spde as spde

    stepper = spde.StochasticAllenCahn(
        num_spatial_dims=1, domain_extent=1.0, num_points=128, dt=5e-4,
        diffusivity=0.01, lambda_=1.0, sigma=0.1, noise_alpha=1.5,
    )
    trajectory = jax.jit(
        spde.stochastic_rollout(stepper, T=500, include_init=True)
    )(u0, jax.random.PRNGKey(0))
"""

import importlib.metadata

from . import _spectral as spectral
from . import etdrk, ic, metrics, nonlin_fun, stepper, viz
from ._base_stepper import BaseStepper
from ._forced_stepper import ForcedStepper
from ._interpolation import FourierInterpolator, map_between_resolutions
from ._repeated_stepper import RepeatedStepper
from ._spectral import derivative, fft, get_spectrum, ifft
from ._stochastic_utils import (
    richardson_weak_extrapolation,
    stochastic_ensemble_rollout,
    stochastic_rollout,
    strang_split_step,
    structure_factor,
)
from ._utils import (
    build_ic_set,
    make_grid,
    repeat,
    rollout,
    stack_sub_trajectories,
    wrap_bc,
)

__version__ = importlib.metadata.version("exponax-spde")

__all__ = [
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
    "richardson_weak_extrapolation",
    "stack_sub_trajectories",
    "stochastic_ensemble_rollout",
    "stochastic_rollout",
    "strang_split_step",
    "structure_factor",
    "build_ic_set",
    "wrap_bc",
    "metrics",
    "etdrk",
    "ic",
    "nonlin_fun",
    "stepper",
    "viz",
    "spectral",
    "FourierInterpolator",
    "map_between_resolutions",
]
