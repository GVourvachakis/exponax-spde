"""Host-side Strang composition of a stochastic stepper and an SSA step."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


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
