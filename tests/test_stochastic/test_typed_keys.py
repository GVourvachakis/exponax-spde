"""Both Threefry key representations carry the same seed and implementation."""

import jax
import jax.numpy as jnp
import numpy as np

import exponax_spde as spde
from exponax_spde.stepper.stochastic import StochasticAllenCahn


def test_typed_and_legacy_threefry_keys_match():
    n = 8
    initial = jnp.linspace(0.1, 0.5, n, dtype=jnp.float64)[None]
    coarse = StochasticAllenCahn(1, 1.0, n, 0.002)
    fine = StochasticAllenCahn(1, 1.0, n, 0.001)
    multiplicative = StochasticAllenCahn(1, 1.0, n, 0.002, noise_type="multiplicative")
    legacy = jax.random.PRNGKey(42, impl="threefry2x32")
    typed = jax.random.key(42, impl="threefry2x32")
    np.testing.assert_array_equal(jax.random.key_data(typed), legacy)

    def richardson(key):
        return spde.richardson_weak_extrapolation(
            coarse, fine, initial, 2, key, num_samples=4, batch_size=3
        )

    rollout = spde.stochastic_rollout(coarse, 4, stride=2)
    ensemble = spde.stochastic_ensemble_rollout(coarse, 4, 4, batch_size=3)
    spectrum = spde.stochastic_structure_factor(coarse, 4, 4, batch_size=3)
    for fn in (
        lambda key: coarse(initial, key=key),
        lambda key: multiplicative(initial, key=key),
        lambda key: coarse.sample_increment(key),
        lambda key: rollout(initial, key),
        lambda key: ensemble(initial, key),
        lambda key: spectrum(initial, key),
        richardson,
    ):
        old, new = jax.jit(fn)(legacy), jax.jit(fn)(typed)
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True):
            assert np.isfinite(np.asarray(b)).all()
            np.testing.assert_array_equal(a, b)
