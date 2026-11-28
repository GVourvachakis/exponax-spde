import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exponax import fft, ifft, spectral

from exponax_spde.stepper import StochasticAllenCahn


@pytest.mark.slow
def test_coupled_additive_strong_order():
    """Order near one for smooth additive increments at this fixed grid.

    A local drift quadrature defect has mean O(h²) and centred RMS
    O(h**1.5). Over O(1/h) steps the mean and accumulated martingale
    RMS are O(h). The paired 95% slope lower bound must exceed 0.75.
    The first reference step is 1/8 of the smallest measured step.
    Its difference from a reference twice as fine must be below 20%
    of the smallest measured error, allowing for sampling error.
    """
    n, batch, paths, finest_steps, end = 32, 256, 4096, 2048, 0.25
    h = end / finest_steps
    factors = (128, 64, 32, 16, 2)
    config = dict(
        diffusivity=0.05,
        lambda_=1.0,
        sigma=0.3,
        noise_alpha=1.5,
        use_taming=False,
        dealiasing_fraction=0.5,
    )
    fine = StochasticAllenCahn(1, 1.0, n, h, **config)
    coarser = tuple(
        StochasticAllenCahn(1, 1.0, n, factor * h, **config) for factor in factors
    )
    initial = fft((0.3 * jnp.cos(2 * jnp.pi * jnp.arange(n) / n) + 0.15)[None])
    initial = jnp.broadcast_to(initial, (batch,) + initial.shape)
    zero = jnp.zeros_like(initial)
    # Running Horner coarsening from one draw per path and fine step.
    modes = spectral.build_derivative_operator(1, 1.0, n)
    propagation = jnp.exp(
        h * (config["lambda_"] + config["diffusivity"] * (modes**2).real)
    )
    fine_step = jax.vmap(fine.step_with_increment)
    coarse_steps = tuple(jax.vmap(model.step_with_increment) for model in coarser)
    draws = jax.vmap(fine.sample_increment)

    @jax.jit
    def run_batch(key):
        def advance(carry, _):
            idx, key, exact, fields, sums = carry
            idx = idx + 1
            key, noise_key = jax.random.split(key)
            inc = draws(jax.random.split(noise_key, batch))
            fields_next, sums_next = [], []
            for factor, step, field, partial in zip(
                factors, coarse_steps, fields, sums, strict=True
            ):
                partial = propagation * partial + inc
                field, partial = jax.lax.cond(
                    idx % factor == 0,
                    lambda pair, step=step: (step(*pair), zero),
                    lambda pair: pair,
                    (field, partial),
                )
                fields_next.append(field)
                sums_next.append(partial)
            return (
                idx,
                key,
                fine_step(exact, inc),
                tuple(fields_next),
                tuple(sums_next),
            ), None

        start = 0, key, initial, (initial,) * len(factors), (zero,) * len(factors)
        (_, _, finer, fields, _), _ = jax.lax.scan(
            advance, start, None, length=finest_steps
        )

        def squared_error(u, v):
            return jnp.mean(
                jax.vmap(lambda diff: ifft(diff, num_points=n))(u - v) ** 2, axis=(1, 2)
            )

        ref = fields[-1]
        return jnp.stack(
            (
                jnp.stack([squared_error(u, ref) for u in fields[:-1]]),
                jnp.stack([squared_error(u, finer) for u in fields[:-1]]),
            ),
        ), squared_error(ref, finer)

    by_batch = [
        run_batch(k) for k in jax.random.split(jax.random.PRNGKey(806), paths // batch)
    ]
    errors = np.concatenate([np.asarray(a) for a, _ in by_batch], axis=-1)
    reference_errors = np.concatenate([np.asarray(b) for _, b in by_batch])
    # Retain one squared error per path and level. Re-sample joint path indices
    # across levels, in batches, so uncertainty respects the coupling.
    times = np.array(factors[:-1]) * h
    x = np.log(times)
    x = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    rng = np.random.default_rng(812)
    bootstrap = []
    for _ in range(10):
        indices = rng.integers(0, paths, (100, paths))
        bootstrap.append(
            np.einsum("i,dib->db", x, 0.5 * np.log(errors[:, :, indices].mean(axis=-1)))
        )
    lower = np.quantile(np.concatenate(bootstrap, axis=-1), 0.025, axis=-1)
    rms = np.sqrt(errors.mean(axis=-1))
    p = np.einsum("i,di->d", x, np.log(rms))
    rse = np.std(errors, axis=-1, ddof=1) / (2 * errors.mean(axis=-1) * np.sqrt(paths))
    bias_rms = np.sqrt(reference_errors.mean())
    refined_error = rms[1, -1]
    print(
        f"paired RMS by h={rms.tolist()}; slopes={p.tolist()}; "
        f"95% lower bounds={lower.tolist()}; "
        f"reference difference / finest measured RMS={bias_rms / refined_error:.4f}; "
        f"max RMS relative SE={rse.max():.4f}"
    )
    assert np.max(rse) <= 0.03
    assert np.all(lower >= 0.75)
    assert bias_rms <= 0.2 * refined_error
