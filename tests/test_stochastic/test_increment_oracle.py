"""Independent NumPy oracle for supplied stochastic increments."""

import exponax as ex
import jax
import numpy as np
import pytest

from exponax_spde.stepper import StochasticAllenCahn


@pytest.mark.parametrize(("d", "alpha"), [(1, 2.0), (2, 1.5)])
@pytest.mark.parametrize("n", [15, 16])
@pytest.mark.parametrize("scheme", ["additive", "eem", "milstein"])
def test_increment_matches_numpy_oracle(d, alpha, n, scheme):
    """Compare one finite-grid step in float64, allowing only FFT round-off."""
    h, nu, reaction, sigma, fraction = 0.013, 0.02, 0.7, 0.5, 0.5
    shape = (n,) * d
    axes = tuple(range(-d, 0))
    grid = np.meshgrid(*([np.arange(n) / n] * d), indexing="ij", sparse=True)
    u = (0.4 + sum(0.12 * np.sin(2 * np.pi * x) for x in grid))[None, ...]
    u_hat = np.fft.rfftn(u, axes=axes)
    stepper = StochasticAllenCahn(
        d,
        1.0,
        n,
        h,
        diffusivity=nu,
        lambda_=reaction,
        sigma=sigma,
        noise_alpha=alpha,
        noise_type="additive" if scheme == "additive" else "multiplicative",
        use_milstein=scheme == "milstein",
        use_taming=False,
        dealiasing_fraction=fraction,
    )
    deterministic = ex.stepper.reaction.AllenCahn(
        d,
        1.0,
        n,
        h,
        diffusivity=nu,
        first_order_coefficient=reaction,
        third_order_coefficient=-reaction,
        order=1,
        dealiasing_fraction=fraction,
    )
    drift = np.asarray(deterministic.step_fourier(u_hat))
    key = jax.random.PRNGKey(907)
    increment = stepper.sample_increment(key)
    dw_hat = np.asarray(increment)
    if scheme == "additive":
        expected = drift + dw_hat
    else:
        u = np.fft.irfftn(u_hat, s=shape, axes=axes)
        dw = np.fft.irfftn(dw_hat, s=shape, axes=axes)
        frequencies = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
        full_squared = sum(
            k**2 for k in np.meshgrid(*([frequencies] * d), indexing="ij", sparse=True)
        )
        k_squared = full_squared[..., : n // 2 + 1]
        propagator = np.exp(h * (reaction - nu * k_squared))
        point_variance = h * n**d * np.mean((1 + full_squared) ** (-alpha))
        kick = sigma * u * dw
        if scheme == "milstein":
            kick += 0.5 * sigma**2 * u * (dw**2 - point_variance)
        expected = drift + propagator * np.fft.rfftn(kick, axes=axes)

    supplied = np.asarray(stepper.step_with_increment(u_hat, increment))
    keyed = np.asarray(stepper.step_fourier(u_hat, key=key))
    for path, actual in (("supplied", supplied), ("keyed", keyed)):
        diff = np.abs(actual - expected)
        relative = np.max(diff / np.maximum(np.abs(expected), np.finfo(float).tiny))
        print(
            f"oracle {scheme} d={d} N={n} {path}: max relative={relative:.6g}, "
            f"max absolute={diff.max():.6g}"
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
