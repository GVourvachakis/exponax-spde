import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exponax_spde.stepper import phi1


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64], ids=["float32", "float64"]
)
def test_phi1_values_and_gradients(dtype):
    """The quotient limits are phi1(0)=1 and phi1'(0)=1/2 by the Taylor series."""
    z = jnp.asarray(
        [-1e6, -100, -1, -0.02, -1e-4, -1e-7, 0, 1e-7, 1e-4, 0.02, 1, 50], dtype=dtype
    )
    reference_z = np.asarray(z, dtype=np.float64)
    reference = np.divide(
        np.expm1(reference_z),
        reference_z,
        out=np.ones_like(reference_z),
        where=reference_z != 0,
    )
    actual = np.asarray(jax.jit(phi1)(z))
    relative = np.max(np.abs(actual / reference - 1))
    derivative = jax.jit(jax.vmap(jax.grad(phi1)))(z)
    print(f"relative error={relative:.3e}; phi1 prime at zero={float(derivative[6]):g}")
    np.testing.assert_allclose(actual, reference, rtol=4 * jnp.finfo(dtype).eps, atol=0)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(derivative[6]) == 0.5
