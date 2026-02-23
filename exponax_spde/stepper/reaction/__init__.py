"""
Reaction-Diffusion Steppers.

They are in their own submodule because they often differ greatly from the ones
in the `exponax_spde.stepper` module. Oftentimes they also come with their own
nonlinear function. If not, they most often use
`exponax_spde.nonlin_fun.PolynomialNonlinearFun`.

They often also require carefully tuned initial conditions, whereas the other
steppers often operate on a wide range of initial conditions.
"""

from ._allen_cahn import AllenCahn

__all__ = [
    "AllenCahn",
]
