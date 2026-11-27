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
    Combine runs at Δt and Δt/2 to boost the *weak* order by one via
    Romberg extrapolation:  E_rich ≈ 2·E_fine - E_coarse.

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
) -> Callable[[Float[Array, "1 * N"], PRNGKeyArray], Float[Array, "T 1 *N"]]:
    """Return a function that rolls a stochastic stepper forward.

    The returned function has signature::

        trajectory = rollout_fn(u_init, key)

    where ``trajectory`` has shape ``(T+1, 1, *N)`` if
    ``include_init=True`` else ``(T, 1, *N)``.

    The function is fully JIT-compatible because it uses
    ``jax.lax.scan`` internally rather than a Python-level loop.

    **Arguments:**

    - `stepper`: A stochastic stepper whose ``__call__`` accepts
        ``(u, key=key)`` as a keyword-only key argument.
    - `num_steps`: Number of time steps T to take.
    - `include_init`: Whether to prepend the initial condition to the
        returned trajectory array.

    **Example:**

    ```python
    rollout_fn = stochastic_rollout(stepper, 500)
    trajectory = jax.jit(rollout_fn)(u_0, jax.random.PRNGKey(0))
    trajectory.shape  # (501, 1, N)
    ```
    """

    def _rollout(
        u_init: Float[Array, "1 * N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "T 1 *N"]:
        def _step(carry, _):
            u, rng = carry
            rng, subkey = jax.random.split(rng)
            u_next = stepper(u, key=subkey)
            return (u_next, rng), u_next

        (_, _), trajectory = jax.lax.scan(_step, (u_init, key), None, length=num_steps)

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


def richardson_weak_extrapolation(
    stepper_coarse: eqx.Module,
    stepper_fine: eqx.Module,
    u0: Float[Array, "1 * N"],
    num_steps_coarse: int,
    key: PRNGKeyArray,
    *,
    num_samples: int = 512,
    batch_size: int | None = None,
) -> dict:
    """Boost weak order via Romberg/Richardson extrapolation.

    Runs two independent ensembles at the same physical end-time
    ``T = num_steps_coarse * Δt_coarse``:

    - **Coarse**: ``stepper_coarse`` for ``num_steps_coarse`` steps at ``Δt``.
    - **Fine**:   ``stepper_fine``   for ``2 * num_steps_coarse`` steps at
        ``Δt/2`` (same wall-clock time T).

    The Richardson extrapolated mean field is::

        E_rich[u(T)] ≈ 2·E_fine[u(T)] - E_coarse[u(T)]

    which doubles the weak convergence order (e.g. order 1 → order 2
    for additive noise with Q-Wiener forcing).

    **Arguments:**

    - `stepper_coarse`: Stepper at time step Δt.
    - `stepper_fine`: Stepper at time step Δt/2 (same physical
        parameters, half the ``dt``).
    - `u0`: Initial condition, shape ``(1, *N)``.
    - `num_steps_coarse`: Number of coarse steps T.
    - `key`: Master PRNGKey; split internally for the two ensembles.
    - `num_samples`: Monte-Carlo sample size M for both ensembles.
    - `batch_size`: Number of paths advanced together. ``None`` uses the
        full ensemble as one batch.

    **Returns:**

    A ``dict`` with keys:

    - ``"mean_coarse"``, ``"mean_fine"``, ``"mean_rich"``: Ensemble-mean
        final states.
    - ``"var_coarse"``, ``"var_fine"``: Ensemble variance of the final
        state.
    """
    _validate_batch_size(batch_size)
    if num_steps_coarse == 0:
        raise IndexError("Cannot select a final state from a zero-step rollout.")
    key_c, key_f = jax.random.split(key)

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
                return final

            keys = jax.random.split(ensemble_key, num_samples)
            return _map_paths(one_path, (keys,), batch_size)

        return jax.jit(ensemble)(u0, key)

    final_c = final_states(stepper_coarse, num_steps_coarse, key_c)
    final_f = final_states(stepper_fine, 2 * num_steps_coarse, key_f)

    mean_c = jnp.mean(final_c, axis=0)
    mean_f = jnp.mean(final_f, axis=0)
    mean_rich = 2.0 * mean_f - mean_c

    var_c = jnp.var(final_c, axis=0)
    var_f = jnp.var(final_f, axis=0)

    return {
        "mean_coarse": mean_c,
        "mean_fine": mean_f,
        "mean_rich": mean_rich,
        "var_coarse": var_c,
        "var_fine": var_f,
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

    **References:**

    - Harrison, J. U., & Yates, C. A. (2016).
        The two-regime method for optimizing stochastic
        reaction-diffusion simulations.
        *Journal of The Royal Society Interface*,
        9(70), 859-868. https://doi.org/10.1098/rsif.2011.0574
        (Strang-split PDE/SSA coupling with band-limited mollifier.)

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
