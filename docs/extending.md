# Extending exponax-spde with New SPDE Steppers

This guide walks through the exact steps needed to add a new stochastic PDE
stepper to `exponax_spde`, following the same pattern as `StochasticAllenCahn`. Use the public `exponax`
API for the deterministic base classes and helpers.
`StochasticAllenCahn` supports drift orders 1–4 and defaults to order 1.
Use the same order for the template's deterministic update and its reference.

> **Note**: All future commits to this repository must follow the
> [Conventional Commits v1.0.0 spec](https://www.conventionalcommits.org/en/v1.0.0/) or the commit-msg hook will reject
> them. See `.commitlintrc.yaml` for the full rule set.

---

## Template: stochastic Kuramoto-Sivashinsky

The 1-D stochastic KS equation in Itô form is:

$$\partial_t u + \tfrac{1}{2}\|\nabla u\|^2 + \Delta u + \partial_x^4 u = \sigma\,\xi(x,t)$$

which splits as $L(u) = -\Delta u - \partial_x^4 u$ (linear, ETD) and
$\mathcal{N}(u) = -\tfrac{1}{2}\|\nabla u\|^2$ (nonlinear, explicit).

See the [additive noise model](noise_model.md) for variance units and coupling.
External extensions can import `phi1` from `exponax_spde.stepper`.

### Step 1 — create the stepper module

`exponax_spde/stepper/_stochastic_ks.py`:

<!-- non-executable template -->
```python
import jax
import jax.numpy as jnp
from exponax import BaseStepper, fft, ifft, spectral
from exponax.nonlin_fun import GradientNormNonlinearFun
from jaxtyping import Array

from ._allen_cahn import phi1


class StochasticKuramotoSivashinsky(BaseStepper):
    _additive_amplitude: Array

    def __init__(
        self,
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        *,
        sigma=0.1,
        noise_alpha=1.0,
    ):
        derivative = spectral.build_derivative_operator(
            num_spatial_dims, domain_extent, num_points
        )
        linear = self._build_linear_operator(derivative).real
        k_sq = -jnp.sum(derivative**2, axis=0, keepdims=True).real
        cell_volume = (domain_extent / num_points) ** num_spatial_dims
        q = (1 + k_sq) ** (-noise_alpha)
        self._additive_amplitude = sigma * jnp.sqrt(
            q * dt * phi1(2 * dt * linear) / cell_volume
        )
        super().__init__(
            num_spatial_dims, domain_extent, num_points, dt, num_channels=1, order=1
        )

    def _build_linear_operator(self, derivative_operator):
        laplacian = spectral.build_laplace_operator(derivative_operator, order=2)
        fourth_order = spectral.build_laplace_operator(derivative_operator, order=4)
        return -laplacian - fourth_order

    def _build_nonlinear_fun(self, derivative_operator):
        return GradientNormNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            derivative_operator=derivative_operator,
            dealiasing_fraction=2 / 3,
            zero_mode_fix=True,
        )

    def step(self, u):
        raise NotImplementedError("Use stepper(u, key=key)")

    def sample_increment(self, key):
        shape = (1,) + (self.num_points,) * self.num_spatial_dims
        white = jax.random.normal(key, shape, dtype=self._additive_amplitude.dtype)
        return self._additive_amplitude * fft(white)

    def step_fourier(self, u_hat, *, key):
        return super().step_fourier(u_hat) + self.sample_increment(key)

    def __call__(self, u, *, key):
        return ifft(self.step_fourier(fft(u), key=key), num_points=self.num_points)
```


### Step 2 — use `TamedPolynomialNonlinearFun` if needed

If the new SPDE has a polynomial nonlinearity of order ≥ 3, import and use
`TamedPolynomialNonlinearFun` from `exponax_spde.nonlin_fun` instead of the
exponax built-in:

<!-- non-executable template -->
```python
from exponax_spde.nonlin_fun import TamedPolynomialNonlinearFun


def _build_nonlinear_fun(self, derivative_operator):
    # Example: cubic term only
    coefficients = [0.0, 0.0, 0.0, -float(self.lambda_)]
    return TamedPolynomialNonlinearFun(
        self.num_spatial_dims,
        self.num_points,
        dt=self.dt,
        coefficients=coefficients,
        dealiasing_fraction=2 / 3,
        use_taming=self.use_taming,
    )
```

### Step 3 — export from the stepper subpackage

`exponax_spde/stepper/__init__.py`:

<!-- non-executable template -->
```python
from ._allen_cahn import StochasticAllenCahn, phi1
from ._stochastic_ks import StochasticKuramotoSivashinsky

__all__ = ["StochasticAllenCahn", "StochasticKuramotoSivashinsky", "phi1"]
```

`exponax_spde/__init__.py`:

<!-- non-executable template -->
```python
from .stepper import StochasticAllenCahn, StochasticKuramotoSivashinsky
```

### Step 4 — add a deterministic limit test

<!-- non-executable template -->
```python
class TestDeterministicLimitKS:
    def test_single_step_sigma_zero_1d(self):
        p = dict(num_spatial_dims=1, domain_extent=100.0, num_points=128, dt=0.1)
        det = ex.stepper.KuramotoSivashinsky(**p, order=1)
        sto = StochasticKuramotoSivashinsky(**p, sigma=0.0)
        u0 = ex.ic.RandomTruncatedFourierSeries(1, cutoff=5)(128, key=KEY0)
        assert jnp.allclose(det(u0), sto(u0, key=KEY0), atol=1e-5)
```

---

## Multiplicative increments

Draw a σ-free Wiener increment by transforming a real Gaussian field.
Multiply the physical increment by the step-start field at each grid point
and propagate its FFT with the linear exponential. Dealias the deterministic
nonlinearity only. For a Milstein correction, centre the squared Wiener
increment by its scalar point variance and apply the same propagator.
Coarse steps sum the fine Wiener increments before computing the product or
correction. The [noise model](noise_model.md) gives the formulas and
regularity requirement.

Use `equinox.filter_jit` or a JIT-compiled closure over the stepper.
The test extra includes physverify for checking host-side NumPy errors
after the JAX solve.

## Checklist

- [ ] `_build_linear_operator` returns `L_k` with the correct sign convention
- [ ] `_build_nonlinear_fun` uses dealiasing (`dealiasing_fraction=2/3`)
- [ ] Additive amplitude uses the exact convolution factor and the FFT of a real draw
- [ ] `step()` raises `NotImplementedError`
- [ ] `__call__` and `step_fourier` both require `key=`
- [ ] `TestDeterministicLimit` passes with `sigma=0, use_taming=False`
- [ ] Exported from the stepper and root `__init__.py` files
- [ ] Module-level docstring lists known limitations
