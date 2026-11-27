"""The nested scan saves the same states as slicing the ordinary rollout."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import exponax_spde as spde
from exponax_spde.stepper.stochastic import StochasticAllenCahn


@pytest.mark.parametrize("stride", [2, 5])
@pytest.mark.parametrize("include_init", [False, True])
def test_rollout_stride_preserves_saved_steps(stride, include_init):
    stepper = StochasticAllenCahn(1, 1.0, 16, 0.002)
    initial = jnp.linspace(0.1, 0.4, 16, dtype=jnp.float64)[None]
    key = jax.random.PRNGKey(63)
    full = spde.stochastic_rollout(stepper, 13, include_init=include_init)
    strided = spde.stochastic_rollout(
        stepper, 13, include_init=include_init, stride=stride
    )
    offset = 0 if include_init else stride - 1
    np.testing.assert_array_equal(
        jax.jit(strided)(initial, key), jax.jit(full)(initial, key)[offset::stride]
    )


def test_rollout_stride_larger_than_run_has_no_saved_steps():
    stepper = StochasticAllenCahn(1, 1.0, 8, 0.002)
    initial = jnp.ones((1, 8), dtype=jnp.float64)
    key = jax.random.PRNGKey(30)
    for include_init in (False, True):
        rollout = spde.stochastic_rollout(
            stepper, 3, include_init=include_init, stride=5
        )
        expected = initial[None] if include_init else np.empty((0, 1, 8))
        np.testing.assert_array_equal(jax.jit(rollout)(initial, key), expected)
    with pytest.raises(ValueError, match="stride"):
        spde.stochastic_rollout(stepper, 3, stride=0)
