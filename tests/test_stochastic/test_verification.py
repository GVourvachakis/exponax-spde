"""Host-side verification of float64 endpoint errors."""

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
from physverify import check_order

from exponax_spde.stepper.stochastic import StochasticAllenCahn


def test_deterministic_etdrk1_order():
    """ETDRK1 has O(h²) local drift error and O(h) global error at fixed T.

    Compare with the installed ETDRK4 drift at one sixteenth of the finest
    step. The deterministic fit needs no Monte Carlo error bar.
    """
    n, horizon = 32, 0.4
    x = jnp.arange(n, dtype=jnp.float64) / n
    u0 = (0.25 + 0.3 * jnp.cos(2 * jnp.pi * x) + 0.07 * jnp.sin(4 * jnp.pi * x))[
        None, :
    ]
    diffusivity, reaction, fraction = 0.025, 1.0, 0.5
    key = jax.random.PRNGKey(51)
    counts = np.array([8, 16, 32, 64])
    ref_steps = int(counts[-1]) * 16
    reference = ex.stepper.reaction.AllenCahn(
        1,
        1.0,
        n,
        horizon / ref_steps,
        diffusivity=diffusivity,
        first_order_coefficient=reaction,
        third_order_coefficient=-reaction,
        dealiasing_fraction=fraction,
        order=4,
    )
    target = jax.jit(
        lambda initial: jax.lax.fori_loop(
            0, ref_steps, lambda _, u: reference(u), initial
        )
    )(u0)

    def endpoint(steps):
        stepper = StochasticAllenCahn(
            1,
            1.0,
            n,
            horizon / steps,
            lambda_=reaction,
            diffusivity=diffusivity,
            dealiasing_fraction=fraction,
            sigma=0.0,
            use_taming=False,
        )
        return jax.jit(
            lambda initial: jax.lax.fori_loop(
                0, steps, lambda _, u: stepper(u, key=key), initial
            )
        )(u0)

    errors = np.array(
        [
            np.sqrt(np.mean(np.asarray(endpoint(int(steps)) - target) ** 2))
            for steps in counts
        ]
    )
    result = check_order(horizon / counts, errors, expected=1)
    print(f"ETDRK1 errors={errors}; {result.status}: {result.message}")
    assert result.status == "PASS", result.message
