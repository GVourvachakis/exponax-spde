"""Analytical float32 smoke checks and a small device throughput benchmark."""

from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from exponax_spde import stochastic_ensemble_rollout, stochastic_structure_factor
from exponax_spde.stepper.stochastic import StochasticAllenCahn


def white_noise_variance():
    """At L=0, MN independent squared kicks estimate σ² h/dx to 5√(2/(MN)) SE."""
    n, samples, h, sigma = 32, 8192, 0.002, 0.3
    stepper = StochasticAllenCahn(
        1,
        1.0,
        n,
        h,
        diffusivity=0,
        lambda_=0,
        sigma=sigma,
        noise_alpha=0,
    )
    initial = jnp.zeros((1, n), dtype=jnp.float32)
    keys = jax.random.split(jax.random.key(604), samples)
    powers = jax.jit(
        lambda keys: jax.lax.map(
            lambda key: jnp.mean(stepper(initial, key=key) ** 2), keys, batch_size=128
        )
    )(keys)
    target = sigma**2 * h * n
    ratio = float(powers.mean()) / target
    z = abs(ratio - 1) / np.sqrt(2 / (samples * n))
    print(f"white noise variance/expected={ratio:.8f}, deviation={z:.4f} SE")
    assert powers.dtype == jnp.float32
    assert z <= 5


def _ou_structure_factor():
    """At T=4h from zero, S_k has exponential power and SE E[S_k]/sqrt(M).

    For a non-boundary 1-D mode on the unit torus, coefficient extraction
    gives mean power 4σ² q_k (1-exp(-2νk²T))/(2νk²). Keeping just the
    final snapshot leaves one independent exponential power per path.
    """
    n, samples, steps = 32, 8192, 4
    h, nu, sigma, alpha = 0.03, 0.01, 0.3, 1.5
    mode = 2
    stepper = StochasticAllenCahn(
        1, 1.0, n, h, diffusivity=nu, lambda_=0, sigma=sigma, noise_alpha=alpha
    )
    spectrum = stochastic_structure_factor(
        stepper,
        steps,
        samples,
        burn_in_fraction=(steps - 1) / steps,
        batch_size=128,
    )
    value = jax.jit(spectrum)(jnp.zeros((1, n), dtype=jnp.float32), jax.random.key(942))
    k_squared = (2 * np.pi * mode) ** 2
    decay = 2 * nu * k_squared
    target = (
        4
        * sigma**2
        * (1 + k_squared) ** (-alpha)
        * (-np.expm1(-decay * steps * h))
        / decay
    )
    ratio = float(value[0, mode]) / target
    z = abs(ratio - 1) * np.sqrt(samples)
    print(f"OU structure factor mode {mode}/expected={ratio:.8f}, deviation={z:.4f} SE")
    assert value.dtype == jnp.float32
    assert np.isfinite(np.asarray(value)).all()
    assert z <= 5


def run_smoke(*, time_runs=False):
    """Check both ensemble paths and analytic statistics in the float32 process."""
    if jax.config.jax_enable_x64:
        raise ValueError("Run the smoke script with JAX_ENABLE_X64 unset.")
    white_noise_variance()
    _ou_structure_factor()

    n, steps, samples = 64, 64, 128
    initial = jnp.full((1, n), 0.25, dtype=jnp.float32)
    key = jax.random.key(79)
    for noise in ("additive", "multiplicative"):
        stepper = StochasticAllenCahn(
            1, 1.0, n, 0.001, noise_type=noise, noise_alpha=1.5
        )
        ensemble = stochastic_ensemble_rollout(
            stepper, steps, samples, include_init=False, batch_size=32
        )
        run = jax.jit(ensemble)
        values = run(initial, key).block_until_ready()
        assert values.dtype == jnp.float32
        assert np.isfinite(np.asarray(values)).all()
        print(f"{noise} ensemble: {values.shape}, {values.dtype}, finite")
        if time_runs:
            durations = []
            for i in range(1, 6):
                draw = jax.random.fold_in(key, i).block_until_ready()
                start = perf_counter()
                run(initial, draw).block_until_ready()
                durations.append(perf_counter() - start)
            seconds = median(durations)
            print(
                f"{noise}: {steps / seconds:.3f} ensemble steps/s, "
                f"{samples * steps / seconds:.3f} member steps/s (median of 5)"
            )


if __name__ == "__main__":
    print(jax.devices())
    run_smoke(time_runs=True)
