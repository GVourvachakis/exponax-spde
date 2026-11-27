import warnings

import jax
import jax.numpy as jnp
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
def test_negative_noise_parameters_are_rejected(name):
    with pytest.raises(ValueError, match=rf"^{name} must be non-negative"):
        StochasticAllenCahn(1, 1.0, 4, 0.001, **{name: -0.1})
