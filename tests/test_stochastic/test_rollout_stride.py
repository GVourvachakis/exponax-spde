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
    full = spde.stochastic_rollout(stepper, 20, include_init=include_init)
    strided = spde.stochastic_rollout(
        stepper, 20, include_init=include_init, stride=stride
    )
    offset = 0 if include_init else stride - 1
    np.testing.assert_array_equal(
        jax.jit(strided)(initial, key), jax.jit(full)(initial, key)[offset::stride]
    )


@pytest.mark.parametrize(("num_steps", "stride"), [(3, 5), (13, 2), (13, 5), (0, 2)])
def test_rollout_rejects_incomplete_stride_blocks(num_steps, stride):
    with pytest.raises(ValueError, match=rf"num_steps={num_steps}.*stride={stride}"):
        spde.stochastic_rollout(None, num_steps, stride=stride)


def test_rollout_rejects_zero_stride():
    with pytest.raises(ValueError, match="stride"):
        spde.stochastic_rollout(None, 3, stride=0)


@pytest.mark.parametrize("stride", [2, 5])
@pytest.mark.parametrize("include_init", [False, True])
@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_ensemble_stride_preserves_saved_steps(stride, include_init, batch_size):
    stepper = StochasticAllenCahn(1, 1.0, 16, 0.002)
    initial = jnp.linspace(0.1, 0.4, 16, dtype=jnp.float64)[None]
    key = jax.random.PRNGKey(64)
    options = dict(include_init=include_init, batch_size=batch_size)
    full = spde.stochastic_ensemble_rollout(stepper, 20, 5, **options)
    strided = spde.stochastic_ensemble_rollout(stepper, 20, 5, stride=stride, **options)
    offset = 0 if include_init else stride - 1
    np.testing.assert_array_equal(
        jax.jit(strided)(initial, key), jax.jit(full)(initial, key)[:, offset::stride]
    )


@pytest.mark.parametrize(
    ("num_steps", "stride"), [(13, 2), (3, 5), (0, 2), (3, 0), (3, -1)]
)
def test_ensemble_rejects_invalid_stride(num_steps, stride):
    with pytest.raises(ValueError) as caught:
        spde.stochastic_ensemble_rollout(None, num_steps, 5, stride=stride)
    assert f"num_steps={num_steps}" in str(caught.value)
    assert f"stride={stride}" in str(caught.value)
