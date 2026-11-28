"""Compare mapped ensembles against the default path on identical keys."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import exponax_spde as spde
from exponax_spde.stepper import StochasticAllenCahn


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_batch_size_preserves_ensemble_statistics(batch_size):
    """Seven paths exercise the remainder at the same seed and grid.

    XLA can round the step arithmetic differently at different batch
    widths. Use 1e-12 relative to compare the resulting float64 paths
    and statistics.
    """
    n, samples, steps = 8, 7, 6
    x = jnp.arange(n, dtype=jnp.float64) / n
    initial = (0.3 + 0.1 * jnp.sin(2 * jnp.pi * (x[:, None] + x[None, :])))[None]
    coarse = StochasticAllenCahn(2, 1.0, n, 0.002)
    fine = StochasticAllenCahn(2, 1.0, n, 0.001)
    key = jax.random.PRNGKey(64)

    def run(batch):
        paths = spde.stochastic_ensemble_rollout(
            coarse, steps, samples, batch_size=batch
        )
        spectrum = spde.stochastic_structure_factor(
            coarse, steps, samples, batch_size=batch
        )
        estimates = spde.richardson_weak_extrapolation(
            coarse, fine, initial, steps, key, num_samples=samples, batch_size=batch
        )
        return jax.jit(paths)(initial, key), jax.jit(spectrum)(initial, key), estimates

    expected, spectrum, estimates = run(None)
    actual, batched_spectrum, batched_estimates = run(batch_size)
    error = np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
    print(f"batch_size={batch_size}: trajectory max absolute difference={error:.9g}")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)

    def relative_error(observed, reference):
        ref = np.asarray(reference)
        difference = np.abs(np.asarray(observed) - ref)
        return np.max(difference / np.maximum(np.abs(ref), np.finfo(float).tiny))

    spectral_error = relative_error(batched_spectrum, spectrum)
    richardson_error = max(
        relative_error(batched_estimates[name], value)
        for name, value in estimates.items()
    )
    print(
        f"batch_size={batch_size}: structure factor max relative={spectral_error:.6g}, "
        f"Richardson max relative={richardson_error:.6g}"
    )
    np.testing.assert_allclose(batched_spectrum, spectrum, rtol=1e-12, atol=0)
    for name, value in estimates.items():
        np.testing.assert_allclose(batched_estimates[name], value, rtol=1e-12, atol=0)
