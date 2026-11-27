"""Run with JAX_ENABLE_X64 unset."""

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import exponax_spde as spde
from exponax_spde.stepper.stochastic import StochasticAllenCahn
from scripts.gpu_smoke import white_noise_variance

pytestmark = pytest.mark.skipif(
    jax.config.jax_enable_x64, reason="Float32 smoke requires JAX_ENABLE_X64 unset."
)


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("scheme", ["additive", "eem", "milstein"])
def test_float32_step_and_rollouts(d, scheme):
    n, steps, samples = 8, 5, 7
    shape = (1,) + (n,) * d
    initial = jnp.linspace(0.1, 0.5, n**d, dtype=jnp.float32).reshape(shape)
    key = jax.random.key(4)
    stepper = StochasticAllenCahn(
        d,
        1.0,
        n,
        0.002,
        noise_alpha=1.5,
        noise_type="additive" if scheme == "additive" else "multiplicative",
        use_milstein=scheme == "milstein",
    )
    rollout = spde.stochastic_rollout(stepper, steps, stride=2)
    ensemble = spde.stochastic_ensemble_rollout(stepper, steps, samples, batch_size=3)
    spectrum = spde.stochastic_structure_factor(stepper, steps, samples, batch_size=3)
    physical = (
        jax.jit(lambda u, k: stepper(u, key=k))(initial, key),
        jax.jit(rollout)(initial, key),
        jax.jit(ensemble)(initial, key),
        jax.jit(spectrum)(initial, key),
    )
    for result in physical:
        assert result.dtype == jnp.float32
        assert np.isfinite(np.asarray(result)).all()
    spectral = jax.jit(lambda u, k: stepper.step_fourier(ex.fft(u), key=k))(
        initial, key
    )
    assert spectral.dtype == jnp.complex64
    assert np.isfinite(np.asarray(spectral)).all()
    print(
        f"{d}D {scheme}: float32 step, rollout, ensemble, spectrum; "
        "complex64 Fourier step"
    )


def test_float32_white_noise_variance():
    white_noise_variance()
