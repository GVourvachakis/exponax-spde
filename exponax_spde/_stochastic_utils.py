"""
Stochastic utilities for Exponax.

Contents
--------
stochastic_rollout(stepper, num_steps, *, include_init)
    Like ``exponax.rollout`` but threads a PRNGKey through a
    stochastic stepper via ``jax.lax.scan``.  Fully JIT-compatible.

stochastic_ensemble_rollout(stepper, num_steps, num_samples, *, include_init)
    ``jax.vmap``-wrapped version of ``stochastic_rollout`` that
    produces M independent trajectories from a single call.
    Returns shape ``(M, T, 1, *N)`` when ``include_init=False``,
    or ``(M, T+1, 1, *N)`` when ``include_init=True``.

structure_factor(ensemble, *, burn_in_fraction)
    Compute ``S(k) = ⟨|û_k|²⟩`` from an ensemble of physical-space
    trajectories (average over the last ``(1 - burn_in_fraction)``
    fraction of time steps).

stochastic_structure_factor(stepper, num_steps, num_samples, *, burn_in_fraction)
    Accumulate the same structure factor directly during a stochastic rollout.

richardson_weak_extrapolation(stepper_coarse, stepper_fine, u0, *, ...)
    Cancel the leading weak bias at Δt and Δt/2, with optional paired sampling.

strang_split_step(spectral_stepper, ssa_step_fn, u, ssa_state, dt, key, *)
    Scaffold for operator-split hybrid SSA / spectral stepping
    (Strang splitting: half-SSA → full-spectral → half-SSA).
    The SSA half-steps are host-side Python calls. The split time loop
    cannot be JIT-compiled and each accelerator step returns to the host.

NOTE on what is NOT provided
-----------------------------
* Richardson *strong* extrapolation: improving the strong order
    requires coupling at the path level (same Wiener increments), which
    exponential Euler-Maruyama does not yet support at order > 1/2.
* Full Milstein for non-gradient noise: iterated Wiener integrals are
    required; not implemented.
* Adaptive Δt: contradicts Exponax's fixed-step design.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from exponax import spectral
from jaxtyping import Array, Float, PRNGKeyArray

# ---------------------------------------------------------------------------
# stochastic_rollout
# ---------------------------------------------------------------------------


def stochastic_rollout(
    stepper: eqx.Module,
    num_steps: int,
    *,
    include_init: bool = True,
    stride: int = 1,
) -> Callable[[Float[Array, "1 * N"], PRNGKeyArray], Float[Array, "T 1 *N"]]:
    """Return a function that rolls a stochastic stepper forward.

    The returned function has signature::

        trajectory = rollout_fn(u_init, key)

    The trajectory contains steps ``stride, 2*stride, ...`` up to
    ``num_steps``. It contains ``num_steps // stride`` states, plus step zero
    if ``include_init=True``.
    Unlike this stochastic utility, upstream ``exponax.rollout`` has no
    ``stride`` argument.

    The function is fully JIT-compatible because it uses
    ``jax.lax.scan`` internally rather than a Python-level loop.

    **Arguments:**

    - `stepper`: A stochastic stepper whose ``__call__`` accepts
        ``(u, key=key)`` as a keyword-only key argument.
    - `num_steps`: Number of time steps T to take.
    - `include_init`: Whether to prepend the initial condition to the
        returned trajectory array.
    - `stride`: Positive number of time steps between saved states.
        The number of steps must be a multiple of the stride.

    **Example:**

    ```python
    rollout_fn = stochastic_rollout(stepper, 500)
    trajectory = jax.jit(rollout_fn)(u_0, jax.random.PRNGKey(0))
    trajectory.shape  # (501, 1, N)
    ```
    """

    if stride < 1:
        raise ValueError("stride must be positive.")
    if num_steps % stride or (stride > 1 and stride > num_steps):
        raise ValueError(
            f"num_steps={num_steps} must be a positive multiple of stride={stride}."
        )

    def _rollout(
        u_init: Float[Array, "1 * N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "T 1 *N"]:
        def _step(carry, _):
            u, rng = carry
            rng, subkey = jax.random.split(rng)
            u_next = stepper(u, key=subkey)
            return (u_next, rng), u_next

        if stride == 1:
            (_, _), trajectory = jax.lax.scan(
                _step, (u_init, key), None, length=num_steps
            )
        else:

            def block(carry, _):
                def take_step(state, _):
                    return _step(state, None)[0], None

                carry, _ = jax.lax.scan(take_step, carry, None, length=stride)
                return carry, carry[0]

            (_, _), trajectory = jax.lax.scan(
                block, (u_init, key), None, length=num_steps // stride
            )

        if include_init:
            trajectory = jnp.concatenate([u_init[jnp.newaxis], trajectory], axis=0)

        return trajectory

    return _rollout


# ---------------------------------------------------------------------------
# stochastic_ensemble_rollout
# ---------------------------------------------------------------------------


def _map_paths(fn, inputs, batch_size):
    if batch_size is None:
        return jax.vmap(fn)(*inputs)
    return jax.lax.map(lambda args: fn(*args), inputs, batch_size=batch_size)


def _validate_batch_size(batch_size):
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive.")


def stochastic_ensemble_rollout(
    stepper: eqx.Module,
    num_steps: int,
    num_samples: int,
    *,
    include_init: bool = True,
    batch_size: int | None = None,
) -> Callable[
    [Float[Array, "1 * N"], PRNGKeyArray],
    Float[Array, "M T 1 *N"],
]:
    """Return a batched rollout function producing M independent paths.

    The returned function has signature::

        ensemble = ensemble_fn(u_init, key)

    where ``ensemble`` has shape ``(M, T+1, 1, *N)``
    (or ``(M, T, 1, *N)`` when ``include_init=False``).

    **Arguments:**

    - `stepper`: Forwarded to ``stochastic_rollout``.
    - `num_steps`: Number of time steps T to take.
    - `num_samples`: Number of Monte-Carlo trajectories M.
    - `include_init`: Whether to prepend the initial condition.
    - `batch_size`: Number of paths to advance together. ``None`` maps over
        the full sample axis. The output still contains every trajectory.

    **Example:**

    ```python
    ens_fn = stochastic_ensemble_rollout(stepper, 200, num_samples=256)
    ensemble = jax.jit(ens_fn)(u_0, jax.random.PRNGKey(0))
    ensemble.shape  # (256, 201, 1, N)
    ```
    """
    _validate_batch_size(batch_size)
    single_rollout = stochastic_rollout(stepper, num_steps, include_init=include_init)

    def _ensemble(
        u_init: Float[Array, "1 * N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "M T 1 *N"]:
        keys = jax.random.split(key, num_samples)
        return _map_paths(lambda k: single_rollout(u_init, k), (keys,), batch_size)

    return _ensemble


# ---------------------------------------------------------------------------
# structure_factor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# richardson_weak_extrapolation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# strang_split_step  (hybrid SSA scaffold — NOT JIT-compiled)
# ---------------------------------------------------------------------------


def strang_split_step(
    spectral_stepper: eqx.Module,
    ssa_step_fn: Callable,
    u: Float[Array, "1 * N"],
    ssa_state: dict,
    dt: float,
    key: PRNGKeyArray,
    *,
    domain_extent: float | None = None,
    num_points: int,
    mollifier_cutoff: float = 0.5,
) -> tuple[Float[Array, "1 * N"], dict]:
    """One Strang-split hybrid spectral / SSA step.

    Implements the pattern: half-SSA(Δt/2) → full-spectral(Δt) →
    half-SSA(Δt/2), which is second-order accurate in Δt for operator
    splitting (Strang, 1968).

    The SSA half-steps call Python on the host, so the split time loop
    cannot be JIT-compiled. On accelerators each split step round-trips
    to the host for SSA and sends the resulting increments to the device.
    The spectral step can be compiled separately.

    **Arguments:**

    - `spectral_stepper`: A ``StochasticAllenCahn`` instance.
    - `ssa_step_fn`: Callable with signature::

            new_ssa_state = ssa_step_fn(ssa_state, dt_half)

        Must update the SSA compartment counts and return the updated
        state dict including a ``"delta_concentration"`` field
        (numpy array of shape ``(1, *N)``) representing the concentration
        change to inject into the PDE field. A missing key or a shape
        different from ``u.shape`` raises ``ValueError``.
    - `u`: Current PDE field, shape ``(1, *N)``.
    - `ssa_state`: Current SSA state dictionary (Python-level, not JAX).
    - `dt`: Full time step; the SSA steps use ``dt/2`` each.
    - `key`: PRNGKey for the spectral noise increment.
    - `domain_extent`: Deprecated and ignored. Passing a domain length
        emits ``DeprecationWarning``. Removal is planned for v0.3.0.
    - `num_points`: Grid points N per dimension.
    - `mollifier_cutoff`: Fraction of Fourier modes retained by the
        band-limited mollifier that converts discrete SSA outcomes to a
        smooth field perturbation.  Default ``0.5`` retains the lowest
        half of resolved modes.

    **Returns:**

    - ``(u_new, ssa_state_new)``: Updated PDE field and SSA state.

    **Notes:**

    Gibbs artefacts from injecting discrete SSA outcomes into the spectral
    field are suppressed by a sharp Fourier truncation mollifier.  A
    smoother Gaussian window is straightforward to substitute.

    The papers below provide context for PDE–compartment coupling and the
    two-regime method. This scaffold is limited to Strang composition and
    Fourier mollification. The complete algorithms from both papers are
    outside its scope.

    **References:**

    - Harrison, J. U., & Yates, C. A. (2016).
        A hybrid algorithm for coupling partial differential equation and
        compartment-based dynamics.
        *Journal of the Royal Society Interface*, 13(122), 20160335.
        https://doi.org/10.1098/rsif.2016.0335 (PDE–compartment coupling.)

    - Flegg, M. B., Chapman, S. J., & Erban, R. (2012).
        The two-regime method for optimizing stochastic reaction–diffusion
        simulations. *Journal of the Royal Society Interface*, 9(70), 859–868.
        https://doi.org/10.1098/rsif.2011.0574 (The two-regime method.)

    - Strang, G. (1968). On the construction and comparison of difference
        schemes. *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
        https://doi.org/10.1137/0705041 (Second-order operator splitting.)

    - Gillespie, D. T. (1977). Exact stochastic simulation of coupled
        chemical reactions. *The Journal of Physical Chemistry*, 81(25),
        2340-2361. https://doi.org/10.1021/j100540a008
        (Stochastic simulation algorithm.)
    """
    if domain_extent is not None:
        warnings.warn(
            "domain_extent is ignored and will be removed in v0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )

    num_spatial_dims = u.ndim - 1
    spatial_axes = tuple(range(-num_spatial_dims, 0))
    grid_shape = (num_points,) * num_spatial_dims
    max_k = int(mollifier_cutoff * (num_points // 2))
    indices = jnp.arange(num_points)
    keep_full = jnp.minimum(indices, num_points - indices) <= max_k
    mask = indices[: num_points // 2 + 1] <= max_k
    for axis in range(num_spatial_dims - 1):
        broadcast_shape = [1] * num_spatial_dims
        broadcast_shape[axis] = num_points
        mask = mask & keep_full.reshape(broadcast_shape)

    def _mollify(state: dict) -> Float[Array, "1 * N"]:
        """Validate and band-limit the half-step concentration increment."""
        if "delta_concentration" not in state:
            raise ValueError('SSA half-step must return "delta_concentration".')
        delta = jnp.asarray(state["delta_concentration"])
        if delta.shape != u.shape:
            raise ValueError(
                f"delta_concentration must have shape {u.shape}, got {delta.shape}."
            )
        delta_hat = jnp.fft.rfftn(delta, axes=spatial_axes)
        return jnp.fft.irfftn(delta_hat * mask, s=grid_shape, axes=spatial_axes).real

    # ── Half SSA step ────────────────────────────────────────────────
    ssa_state_mid = ssa_step_fn(ssa_state, dt / 2.0)
    u = u + _mollify(ssa_state_mid)

    # ── Full spectral step ───────────────────────────────────────────
    key, subkey = jax.random.split(key)
    u = spectral_stepper(u, key=subkey)

    # ── Half SSA step ────────────────────────────────────────────────
    ssa_state_new = ssa_step_fn(ssa_state_mid, dt / 2.0)
    u = u + _mollify(ssa_state_new)

    return u, ssa_state_new
