"""
exponax_spde.stepper.stochastic
===============================
Stochastic stepper sub-package.

Public API
----------
StochasticAllenCahn
    EEM integrator for the stochastic Allen-Cahn SPDE.
    See ``exponax_spde.stepper.stochastic`` for the full implementation.

Future additions should follow the same pattern (see docs/extending.md).

Extending
---------
To add a new SPDE stepper (e.g. stochastic Kuramoto-Sivashinsky):

1. Create ``exponax_spde/stepper/stochastic/_stochastic_ks.py``
2. Subclass ``exponax.BaseStepper``
3. Implement ``_build_linear_operator`` and ``_build_nonlinear_fun``
4. Use ``TamedPolynomialNonlinearFun`` for super-linearly growing terms
5. Disable ``step()``, require a PRNGKey in ``__call__``
6. Export from ``exponax_spde/stepper/stochastic/__init__.py``
"""

from ._stochastic_allen_cahn import StochasticAllenCahn

__all__ = [
    "StochasticAllenCahn",
]
