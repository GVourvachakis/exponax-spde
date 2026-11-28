"""Stochastic power spectra and weak Richardson extrapolation."""

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from exponax import spectral
from jaxtyping import Array, Float, PRNGKeyArray

from ._rollout import _map_paths, _validate_batch_size


def structure_factor(
    ensemble: Float[Array, "M T 1 *N"],
    *,
    burn_in_fraction: float = 0.5,
) -> Float[Array, "1 * K"]:
    """Compute the empirical structure factor from a trajectory ensemble.

    Returns ``S(k) = ⟨|û_k|²⟩`` where the average is taken over all
    Monte-Carlo samples and over the stationary portion of each
    trajectory (the last ``1 - burn_in_fraction`` of time steps).

    Coefficients are extracted in *coefficient-space* convention (i.e.
    with ``scaling_compensation_mode="coef_extraction"`` and
    ``round=None``), matching the analytical formula

    ```
        C_k = Q_k / (2 ν |k|²_phys)
    ```

    which is the invariant measure of the linear (λ=0) stochastic
    Allen-Cahn equation.

    **Arguments:**

    - `ensemble`: Array of shape ``(M, T, 1, *N)`` where M is the
        number of Monte-Carlo trajectories and T the number of time
        steps (excluding the initial condition when ``include_init=False``
        was used in ``stochastic_ensemble_rollout``).
    - `burn_in_fraction`: Fraction of time steps to discard as burn-in
        before accumulating statistics.  Defaults to ``0.5``.

    **Returns:**

    - ``S_k``: Array of shape ``(1, *K)`` where ``*K`` is the
        wavenumber-space shape of the rfft of the spatial field.
    """
    M, T, *spatial_shape = ensemble.shape
    burn_in = int(T * burn_in_fraction)
    tail = ensemble[:, burn_in:, ...]  # (M, T_stat, 1, *N)
    flat = tail.reshape((-1,) + tuple(spatial_shape))  # (M*T_stat, 1, *N)

    coeffs_per_snapshot = jax.vmap(
        lambda u: spectral.get_fourier_coefficients(
            u, scaling_compensation_mode="coef_extraction", round=None
        )
    )(flat)  # (M*T_stat, 1, *K)

    S_k = jnp.mean(jnp.abs(coeffs_per_snapshot) ** 2, axis=0)  # (1, *K)
    return S_k


def stochastic_structure_factor(
    stepper: eqx.Module,
    num_steps: int,
    num_samples: int,
    *,
    burn_in_fraction: float = 0.5,
    batch_size: int | None = None,
) -> Callable[[Float[Array, "1 * N"], PRNGKeyArray], Float[Array, "1 * K"]]:
    """Return a rollout function that accumulates the empirical structure factor.

    With the same initial state and key, its spectrum agrees up to rounding
    with that of ``structure_factor`` applied to ``stochastic_ensemble_rollout``
    with ``include_init=False`` and the same burn-in fraction. The returned
    function ``fn(u_init, key)`` has shape ``(1, *K)`` in real-FFT coefficient
    extraction units.

    The scan carries the current fields and spectral sums, so peak memory
    is O(num_samples × N^d), independent of the number of steps. The step and
    sample counts must be positive, and ``0 <= burn_in_fraction < 1``.
    Set ``batch_size`` to limit the number of paths advanced and transformed
    together. The scan still carries one field per sample.
    """
    _validate_batch_size(batch_size)
    if num_steps < 1 or num_samples < 1:
        raise ValueError("num_steps and num_samples must be positive.")
    if not 0.0 <= burn_in_fraction < 1.0:
        raise ValueError("burn_in_fraction must be in [0, 1).")

    burn_in = int(num_steps * burn_in_fraction)

    def modal_power(u):
        coefficients = spectral.get_fourier_coefficients(
            u, scaling_compensation_mode="coef_extraction", round=None
        )
        return jnp.abs(coefficients) ** 2

    def spectrum(u_init, key):
        zero_power = jnp.zeros_like(modal_power(u_init))
        fields = jnp.broadcast_to(u_init, (num_samples,) + u_init.shape)
        keys = jax.random.split(key, num_samples)

        def advance(u, k):
            return stepper(u, key=k)

        def step(carry, _):
            fields, keys, index, total, correction = carry
            split = jax.vmap(jax.random.split)(keys)
            fields = _map_paths(advance, (fields, split[:, 1]), batch_size)

            def accumulate():
                # Compensate round-off over long averaging windows.
                powers = _map_paths(modal_power, (fields,), batch_size)
                adjusted = jnp.sum(powers, axis=0) - correction
                updated = total + adjusted
                return updated, (updated - total) - adjusted

            total, correction = jax.lax.cond(
                index >= burn_in, accumulate, lambda: (total, correction)
            )
            return (fields, split[:, 0], index + 1, total, correction), None

        (_, _, _, total, _), _ = jax.lax.scan(
            step, (fields, keys, 0, zero_power, zero_power), None, length=num_steps
        )
        return total / (num_samples * (num_steps - burn_in))

    return spectrum


def _validate_richardson_pair(coarse, fine):
    methods = ("sample_increment", "step_with_increment", "coarsen_increments")
    for name, stepper in (("coarse", coarse), ("fine", fine)):
        for method in methods:
            if not callable(getattr(stepper, method, None)):
                raise ValueError(f"Coupled Richardson requires {name}.{method}().")
        if getattr(stepper, "dt", None) is None:
            raise ValueError(f"Coupled Richardson requires {name}.dt.")

    traced_dt = any(isinstance(dt, jax.core.Tracer) for dt in (coarse.dt, fine.dt))
    if not traced_dt and not math.isclose(
        float(coarse.dt), 2.0 * float(fine.dt), rel_tol=1e-12
    ):
        raise ValueError("Coupled Richardson requires coarse dt == 2 × fine dt.")

    for name in (
        "num_spatial_dims",
        "num_points",
        "domain_extent",
        "noise_type",
        "sigma",
        "noise_alpha",
    ):
        a = getattr(coarse, name, None)
        b = getattr(fine, name, None)
        if (
            a is None
            or b is None
            or any(isinstance(v, jax.core.Tracer) for v in (a, b))
        ):
            continue
        if a != b:
            raise ValueError(f"Coupled Richardson requires matching {name}.")


def richardson_weak_extrapolation(
    stepper_coarse: eqx.Module,
    stepper_fine: eqx.Module,
    u0: Float[Array, "1 * N"],
    num_steps_coarse: int,
    key: PRNGKeyArray,
    *,
    num_samples: int = 512,
    batch_size: int | None = None,
    weak_order: float = 1.0,
    coupled: bool = False,
    observable: Callable[[Array], Array] | None = None,
) -> dict[str, Array]:
    """Extrapolate the expected value of a terminal observable.

    Take `num_steps_coarse` coarse steps of size h and twice as many fine
    steps of size h/2 from the same physical initial field `u0`. For a common
    leading weak-bias term proportional to h**p, where p is `weak_order`,
    the weighted estimate (2**p * mean_fine - mean_coarse) / (2**p - 1)
    cancels that term. Pass a finite, positive `weak_order`.

    The default samples independent ensembles from `key`. Use
    `coupled=True` with compatible stochastic steppers to share the
    driving noise. The experimental protocol requires both steppers to
    expose `dt` and implement `sample_increment(key)`,
    `step_with_increment(u_hat, increment)` and
    `coarsen_increments(increments)`. Use steppers for the same model.
    The fine stepper coarsens each consecutive pair of fine increments.
    With concrete time steps, the coarse step must be twice the fine step.
    Concrete grid and noise settings must match wherever both expose them.

    `num_samples` counts paths per ensemble, or pairs in coupled mode,
    and must be at least two. `batch_size` bounds the number of paths or
    pairs advanced at once. An optional pure JAX `observable` maps each
    terminal physical field to a scalar or fixed-shape array. The default
    uses the physical field itself.

    The result contains `mean_coarse`, `mean_fine`, `mean_rich`,
    `var_coarse` and `var_fine` (population variances), plus
    `se_coarse`, `se_fine`, `se_rich` and `se_difference` (for the
    fine-minus-coarse mean). Standard errors use sample variances
    with ddof=1. In coupled mode, the last two use the paired samples
    to account for covariance. All results are componentwise.
    """
    _validate_batch_size(batch_size)
    if num_samples < 2:
        raise ValueError("Richardson extrapolation requires num_samples >= 2.")
    try:
        finite_order = math.isfinite(weak_order)
        weak_order = float(weak_order)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("weak_order must be finite and greater than zero.") from error
    if not finite_order or weak_order <= 0:
        raise ValueError("weak_order must be finite and greater than zero.")
    if num_steps_coarse == 0:
        raise IndexError("Cannot select a final state from a zero-step rollout.")

    def final_states(stepper, num_steps, key):
        def ensemble(u_init, ensemble_key):
            def one_path(path_key):
                def step(carry, _):
                    u, rng = carry
                    rng, subkey = jax.random.split(rng)
                    return (stepper(u, key=subkey), rng), None

                (final, _), _ = jax.lax.scan(
                    step, (u_init, path_key), None, length=num_steps
                )
                return final if observable is None else observable(final)

            keys = jax.random.split(ensemble_key, num_samples)
            return _map_paths(one_path, (keys,), batch_size)

        return jax.jit(ensemble)(u0, key)

    if coupled:
        _validate_richardson_pair(stepper_coarse, stepper_fine)
        dims = u0.ndim - 1
        points = u0.shape[-1]

        @jax.jit
        def paired_ensemble(u_init, ensemble_key):
            u_hat = spectral.fft(u_init, num_spatial_dims=dims)

            def one_pair(path_key):
                def step(carry, _):
                    coarse, fine, rng = carry
                    rng, first_key = jax.random.split(rng)
                    rng, second_key = jax.random.split(rng)
                    first = stepper_fine.sample_increment(first_key)
                    second = stepper_fine.sample_increment(second_key)
                    coarse_increment = stepper_fine.coarsen_increments(
                        jnp.stack((first, second))
                    )
                    coarse = stepper_coarse.step_with_increment(
                        coarse, coarse_increment
                    )
                    fine = stepper_fine.step_with_increment(fine, first)
                    fine = stepper_fine.step_with_increment(fine, second)
                    return (coarse, fine, rng), None

                (coarse, fine, _), _ = jax.lax.scan(
                    step, (u_hat, u_hat, path_key), None, length=num_steps_coarse
                )
                coarse = spectral.ifft(coarse, num_spatial_dims=dims, num_points=points)
                fine = spectral.ifft(fine, num_spatial_dims=dims, num_points=points)
                if observable is None:
                    return coarse, fine
                return observable(coarse), observable(fine)

            keys = jax.random.split(ensemble_key, num_samples)
            return _map_paths(one_pair, (keys,), batch_size)

        final_c, final_f = paired_ensemble(u0, key)
    else:
        key_c, key_f = jax.random.split(key)
        final_c = final_states(stepper_coarse, num_steps_coarse, key_c)
        final_f = final_states(stepper_fine, 2 * num_steps_coarse, key_f)

    mean_c = jnp.mean(final_c, axis=0)
    mean_f = jnp.mean(final_f, axis=0)
    log_inverse_weight = -weak_order * math.log(2.0)
    inverse_weight = math.exp(log_inverse_weight)
    denominator = -math.expm1(log_inverse_weight)

    def extrapolate(coarse, fine):
        if weak_order == 1.0:
            return 2.0 * fine - coarse
        return fine + inverse_weight * (fine - coarse) / denominator

    mean_rich = extrapolate(mean_c, mean_f)
    var_c = jnp.var(final_c, axis=0)
    var_f = jnp.var(final_f, axis=0)
    sample_scale = math.sqrt(num_samples)

    def standard_error(values):
        return jnp.std(values, axis=0, ddof=1) / sample_scale

    se_c = standard_error(final_c)
    se_f = standard_error(final_f)
    if coupled:
        se_rich = standard_error(extrapolate(final_c, final_f))
        se_difference = standard_error(final_f - final_c)
    else:
        se_rich = jnp.hypot(se_f, inverse_weight * se_c) / denominator
        se_difference = jnp.hypot(se_f, se_c)

    return {
        "mean_coarse": mean_c,
        "mean_fine": mean_f,
        "mean_rich": mean_rich,
        "var_coarse": var_c,
        "var_fine": var_f,
        "se_coarse": se_c,
        "se_fine": se_f,
        "se_rich": se_rich,
        "se_difference": se_difference,
    }
