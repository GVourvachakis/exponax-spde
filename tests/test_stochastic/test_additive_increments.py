import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exponax_spde import fft
from exponax_spde.stepper.stochastic import StochasticAllenCahn


@pytest.mark.parametrize("ratio", [2, 4])
def test_coarsened_ou_path(ratio):
    """The linear drift composes exactly, and O_j gets exp((r-1-j) h L)."""
    key = jax.random.PRNGKey(41)
    fine = StochasticAllenCahn(2, 1.0, 16, 0.002, lambda_=0)
    coarse = StochasticAllenCahn(2, 1.0, 16, ratio * fine.dt, lambda_=0)
    u = fft(jax.random.normal(key, (1, 16, 16)))
    increments = jax.vmap(fine.sample_increment)(
        jax.random.split(jax.random.fold_in(key, 1), ratio)
    )
    expected = u
    for increment in increments:
        expected = fine.step_with_increment(expected, increment)
    actual = coarse.step_with_increment(
        u, jax.jit(lambda x: fine.coarsen_increments(x))(increments)
    )
    relative = float(jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected)))
    print(f"r={ratio}: relative sup error={relative:.3e}")
    assert relative <= 8 * np.finfo(np.float64).eps


def test_coarsened_increment_covariance():
    """The weighted sum has variance P σ² q expm1(2Lrh)/(2L dx).

    Independent complex squared magnitudes have relative SE 1/sqrt(M).
    The two self-conjugate modes are real and use sqrt(2/M).
    """
    n, m, chunk, ratio = 16, 8192, 256, 4
    h, nu, lam, sigma, alpha = 0.01, 0.09, 0.7, 0.3, 0.75
    stepper = StochasticAllenCahn(
        1, 1.0, n, h, diffusivity=nu, lambda_=lam, sigma=sigma, noise_alpha=alpha
    )

    def power(key):
        increments = jax.vmap(stepper.sample_increment)(jax.random.split(key, ratio))
        return jnp.abs(stepper.coarsen_increments(increments)[0]) ** 2

    @jax.jit
    def batch(key):
        return jax.vmap(power)(jax.random.split(key, chunk)).sum(axis=0)

    total = np.zeros(n // 2 + 1)
    for key in jax.random.split(jax.random.PRNGKey(61), m // chunk):
        total += np.asarray(batch(key))
    k2 = (2 * np.pi * np.arange(n // 2 + 1)) ** 2
    linear = lam - nu * k2
    target = (
        n**2
        * sigma**2
        * (1 + k2) ** (-alpha)
        * np.expm1(2 * linear * ratio * h)
        / (2 * linear)
    )
    relative_error = np.abs(total / (m * target) - 1)
    se = np.full_like(target, m**-0.5)
    se[[0, -1]] *= np.sqrt(2)
    print(
        f"coarsened power max error={relative_error.max():.3e}, "
        f"max SE={max(relative_error / se):.3f}"
    )
    assert np.all(relative_error <= 5 * se)


def test_nonlinear_increment_replay():
    """A supplied sequence reproduces the same λ≠0 keyed update exactly."""
    stepper = StochasticAllenCahn(1, 1.0, 16, 0.002, lambda_=0.7, sigma=0.3)
    initial = fft(jnp.linspace(-0.1, 0.3, 16)[None, :])
    keyed = replayed = initial
    for key in jax.random.split(jax.random.PRNGKey(62), 4):
        keyed = stepper.step_fourier(keyed, key=key)
        replayed = stepper.step_with_increment(replayed, stepper.sample_increment(key))
    np.testing.assert_array_equal(keyed, replayed)
    print("nonlinear replay: exact over 4 steps")


@pytest.mark.parametrize("noise_type", ["additive", "multiplicative"])
def test_increment_api_validation(noise_type):
    """The raw rFFT shape and two-step minimum are part of the API."""
    stepper = StochasticAllenCahn(1, 1.0, 16, 0.002, noise_type=noise_type)
    key = jax.random.PRNGKey(43)
    increment = stepper.sample_increment(key)
    with pytest.raises(ValueError, match="increment shape"):
        stepper.step_with_increment(increment, increment[:, :-1])
    with pytest.raises(ValueError, match="r >= 2"):
        stepper.coarsen_increments(increment[None])
    with pytest.raises(ValueError, match="shape"):
        stepper.coarsen_increments(jnp.stack((increment[:, :-1],) * 2))
    print("increment shape and minimum length guards: passed")
