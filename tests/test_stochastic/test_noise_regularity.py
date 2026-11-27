import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exponax_spde import NoiseRegularityWarning
from exponax_spde.stepper.stochastic import StochasticAllenCahn

WARNING_MESSAGES = {
    "additive": (
        "Additive noise with noise_alpha <= d/2 - 1 in d >= 2 has divergent "
        "continuum point variance. The finite-grid model is available. "
        "Continuum Allen–Cahn in this regime requires a separate renormalisation "
        "treatment (#8)."
    ),
    "multiplicative": (
        "Multiplicative noise with noise_alpha <= d/2 - 1 in d >= 2 is too rough "
        "for the classical continuum pointwise product. This stepper simulates "
        "the finite-grid collocation model (#8)."
    ),
}


@pytest.mark.parametrize("noise_type", ["additive", "multiplicative"])
@pytest.mark.parametrize(
    ("d", "alpha"),
    [(2, 0.0), (3, 0.45), (3, 0.5)],
    ids=["2d-white", "3d-below", "3d-boundary"],
)
def test_rough_noise_warns(d, alpha, noise_type):
    with pytest.warns(NoiseRegularityWarning) as notices:
        StochasticAllenCahn(d, 1.0, 4, 0.001, noise_alpha=alpha, noise_type=noise_type)
    assert len(notices) == 1
    assert notices[0].category is NoiseRegularityWarning
    assert issubclass(notices[0].category, UserWarning)
    assert str(notices[0].message) == WARNING_MESSAGES[noise_type]


@pytest.mark.parametrize("noise_type", ["additive", "multiplicative"])
@pytest.mark.parametrize(
    ("d", "alpha", "options"),
    [
        (2, 0.05, {}),
        (3, 0.55, {}),
        (1, 0.0, {}),
        (3, 0.0, {"sigma": 0}),
        (3, 0.0, {"diffusivity": 0}),
    ],
    ids=["2d-above", "3d-above", "1d-white", "zero-sigma", "zero-diffusivity"],
)
def test_warning_exemptions(d, alpha, options, noise_type):
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseRegularityWarning)
        StochasticAllenCahn(
            d, 1.0, 4, 0.001, noise_alpha=alpha, noise_type=noise_type, **options
        )


def test_selective_regularity_warning_suppression():
    with warnings.catch_warnings(record=True) as notices:
        warnings.filterwarnings("always", category=NoiseRegularityWarning)
        warnings.filterwarnings(
            "ignore", message="^Additive noise", category=NoiseRegularityWarning
        )
        StochasticAllenCahn(2, 1.0, 4, 0.001, noise_alpha=0)
        StochasticAllenCahn(
            2, 1.0, 4, 0.001, noise_alpha=0, noise_type="multiplicative"
        )
    assert [str(notice.message) for notice in notices] == [
        WARNING_MESSAGES["multiplicative"]
    ]


def test_no_regularity_warning_from_jitted_step():
    with pytest.warns(NoiseRegularityWarning):
        solver = StochasticAllenCahn(2, 1.0, 4, 0.001, noise_alpha=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseRegularityWarning)
        step = jax.jit(lambda u, key: solver(u, key=key))
        actual = step(jnp.zeros((1, 4, 4)), jax.random.key(0)).block_until_ready()
    assert jnp.isfinite(actual).all()


@pytest.mark.parametrize("name", ["noise_alpha", "sigma", "diffusivity"])
@pytest.mark.parametrize("wrap", [float, jnp.asarray], ids=["python", "array"])
def test_negative_noise_parameters_are_rejected(name, wrap):
    with pytest.raises(ValueError, match=rf"^{name} must be non-negative"):
        StochasticAllenCahn(1, 1.0, 4, 0.001, **{name: wrap(-0.1)})


@pytest.mark.parametrize("transform", ["grad", "vmap"])
def test_constructing_with_traced_sigma(transform):
    """The λ=0 spatial mean is σ√(h/dx²) times the mean of the same draw."""
    n, dt, key = 8, 0.015, jax.random.key(123)
    scale = n * np.sqrt(dt) * np.mean(np.asarray(jax.random.normal(key, (1, n, n))))

    def loss(sigma):
        stepper = StochasticAllenCahn(
            2, 1.0, n, dt, lambda_=0, sigma=sigma, noise_alpha=0
        )
        return jnp.mean(stepper(jnp.zeros((1, n, n)), key=key))

    with warnings.catch_warnings():
        warnings.simplefilter("error", NoiseRegularityWarning)
        if transform == "grad":
            actual, expected = jax.grad(loss)(0.2), scale
        else:
            sigmas = jnp.array([0.1, 0.2, 0.4])
            actual, expected = jax.vmap(loss)(sigmas), np.asarray(sigmas) * scale

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)


def test_constructing_with_traced_diffusivity():
    """The first cosine mode at λ=σ=0 decays as exp(-ν h (2π)²)."""
    n, dt, nu = 8, 0.015, 0.07
    initial = jnp.cos(2 * jnp.pi * jnp.arange(n) / n)[None, :]
    key = jax.random.key(123)

    def loss(diffusivity):
        stepper = StochasticAllenCahn(
            1, 1.0, n, dt, lambda_=0, sigma=0, diffusivity=diffusivity
        )
        return 2 * jnp.mean(initial * stepper(initial, key=key))

    k2_dt = (2 * np.pi) ** 2 * dt
    expected = -k2_dt * np.exp(-nu * k2_dt)
    np.testing.assert_allclose(jax.grad(loss)(nu), expected, rtol=2e-6, atol=1e-8)


def test_warning_points_to_constructor_call():
    with pytest.warns(NoiseRegularityWarning) as records:
        line = sys._getframe().f_lineno + 1
        StochasticAllenCahn(2, 1.0, 4, 0.001, noise_alpha=0)
    (record,) = records
    assert record.filename == __file__
    assert record.lineno == line
    print(f"{record.filename}:{record.lineno}")
