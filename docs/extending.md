# Extending exponax-spde with New SPDE Steppers

This guide walks through the exact steps needed to add a new stochastic PDE
stepper to `exponax_spde`, following the same pattern as `StochasticAllenCahn`.

> **Note**: All future commits to this repository must follow the
> [Conventional Commits v1.0.0 spec](https://www.conventionalcommits.org/en/v1.0.0/) or the commit-msg hook will reject
> them. See `.commitlintrc.yaml` for the full rule set.

---

## Template: stochastic Kuramoto-Sivashinsky

The stochastic KS equation in Itô form is:

$$\partial_t u + \tfrac{1}{2}\|\nabla u\|^2 + \Delta u + \Delta^2 u = \sigma(u)\,\xi(x,t)$$

which splits as $L(u) = -\Delta u - \Delta^2 u$ (linear, ETD) and
$\mathcal{N}(u) = -\tfrac{1}{2}\|\nabla u\|^2$ (nonlinear, explicit).

### Step 1 — create the stepper module

`exponax_spde/stepper/stochastic/_stochastic_ks.py`:

```python
from __future__ import annotations
from typing import Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray
from exponax._base_stepper import BaseStepper
from exponax._spectral import build_derivative_operator, build_scaling_array, fft, ifft
from exponax.nonlin_fun import ConvectionNonlinearFun   # or write a custom one

class StochasticKuramotoSivashinsky(BaseStepper):
    diffusivity: float
    sigma: float
    noise_alpha: float
    noise_type: str = eqx.field(static=True)
    _noise_std: Array
    _noise_std_base: Array

    def __init__(self, num_spatial_dims, domain_extent, num_points, dt, *,
                 diffusivity=1.0, sigma=0.1, noise_alpha=1.0,
                 noise_type="additive", order=1):
        self.diffusivity = diffusivity
        self.sigma = sigma
        self.noise_alpha = noise_alpha
        self.noise_type = noise_type

        deriv_op = build_derivative_operator(num_spatial_dims, domain_extent, num_points)
        k_sq = -jnp.sum(deriv_op**2, axis=0, keepdims=True).real
        dx_d = (domain_extent / num_points) ** num_spatial_dims
        filter_k = (1.0 + k_sq) ** (-noise_alpha / 2.0)
        sqrt_dt_over_dxd = jnp.sqrt(dt / dx_d)
        self._noise_std = sigma * filter_k * sqrt_dt_over_dxd
        self._noise_std_base = filter_k * sqrt_dt_over_dxd

        super().__init__(num_spatial_dims, domain_extent, num_points, dt,
                         num_channels=1, order=order)

    def _build_linear_operator(self, derivative_operator):
        laplacian = jnp.sum(derivative_operator**2, axis=0, keepdims=True)
        # L = -Δ - Δ²  (instability at intermediate scales)
        return -laplacian - laplacian**2

    def _build_nonlinear_fun(self, derivative_operator):
        # Gradient-norm nonlinearity: N(u) = -½|∇u|²
        return GradientNormNonlinearFun(
            self.num_spatial_dims, self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=2/3,
            zero_mode_fix=True,
        )

    def step(self, u):
        raise NotImplementedError("Use stepper(u, key=key)")

    def __call__(self, u, *, key):
        # Copy the _stochastic_step pattern from StochasticAllenCahn verbatim,
        # substituting self._noise_std / self._noise_std_base.
        ...
```

### Step 2 — use `TamedPolynomialNonlinearFun` if needed

If the new SPDE has a polynomial nonlinearity of order ≥ 3, import and use
`TamedPolynomialNonlinearFun` from `exponax_spde.nonlin_fun` instead of the
exponax built-in:

```python
from exponax_spde.nonlin_fun import TamedPolynomialNonlinearFun

def _build_nonlinear_fun(self, derivative_operator):
    # Example: cubic term only
    coefficients = [0.0, 0.0, 0.0, -float(self.lambda_)]
    return TamedPolynomialNonlinearFun(
        self.num_spatial_dims, self.num_points,
        dt=self.dt, coefficients=coefficients,
        dealiasing_fraction=2/3, use_taming=self.use_taming,
    )
```

### Step 3 — export from the stochastic subpackage

`exponax_spde/stepper/stochastic/__init__.py`:

```python
from ._stochastic_allen_cahn import StochasticAllenCahn
from ._stochastic_ks import StochasticKuramotoSivashinsky   # add this line

__all__ = ["StochasticAllenCahn", "StochasticKuramotoSivashinsky"]
```

`exponax_spde/stepper/__init__.py`:

```python
from .stochastic import StochasticAllenCahn, StochasticKuramotoSivashinsky
__all__ = ["StochasticAllenCahn", "StochasticKuramotoSivashinsky"]
```

`exponax_spde/__init__.py`:

```python
from .stepper import StochasticAllenCahn, StochasticKuramotoSivashinsky
```

### Step 4 — add a deterministic limit test

```python
class TestDeterministicLimitKS:
    def test_single_step_sigma_zero_1d(self):
        p = dict(num_spatial_dims=1, domain_extent=100.0, num_points=128, dt=0.1)
        det = ex.stepper.KuramotoSivashinskyConservative(**p)
        sto = StochasticKuramotoSivashinsky(**p, sigma=0.0)
        u0 = ex.ic.RandomTruncatedFourierSeries(1, cutoff=5)(128, key=KEY0)
        assert jnp.allclose(det(u0), sto(u0, key=KEY0), atol=1e-5)
```

---

## Checklist

- [ ] `_build_linear_operator` returns `L_k` with the correct sign convention
- [ ] `_build_nonlinear_fun` uses dealiasing (`dealiasing_fraction=2/3`)
- [ ] `_noise_std` precomputed with `σ · filter_k · √(Δt/dx^d)` in `__init__`
- [ ] `step()` raises `NotImplementedError`
- [ ] `__call__` and `step_fourier` both require `key=`
- [ ] `TestDeterministicLimit` passes with `sigma=0, use_taming=False`
- [ ] Exported from all three `__init__.py` files
- [ ] Module-level docstring lists known limitations
