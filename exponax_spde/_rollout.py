"""Stochastic trajectory and ensemble rollouts."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


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
        raise ValueError(f"stride={stride} must be positive for num_steps={num_steps}.")
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
    stride: int = 1,
) -> Callable[
    [Float[Array, "1 * N"], PRNGKeyArray],
    Float[Array, "M T 1 *N"],
]:
    """Return a batched rollout function producing M independent paths.

    The returned function has signature::

        ensemble = ensemble_fn(u_init, key)

    where ``ensemble`` has shape ``(M, S, 1, *N)`` and ``S`` is
    ``num_steps // stride + int(include_init)``. The saved states use
    the same keys as the corresponding states with ``stride=1``.

    **Arguments:**

    - `stepper`: Forwarded to ``stochastic_rollout``.
    - `num_steps`: Number of time steps T to take.
    - `num_samples`: Number of Monte-Carlo trajectories M.
    - `include_init`: Whether to prepend the initial condition.
    - `batch_size`: Number of paths to advance together. ``None`` maps over
        the full sample axis.
    - `stride`: Positive number of time steps between saved states.
        The number of steps must be a multiple of the stride.

    **Example:**

    ```python
    ens_fn = stochastic_ensemble_rollout(stepper, 200, num_samples=256)
    ensemble = jax.jit(ens_fn)(u_0, jax.random.PRNGKey(0))
    ensemble.shape  # (256, 201, 1, N)
    ```
    """
    _validate_batch_size(batch_size)
    single_rollout = stochastic_rollout(
        stepper, num_steps, include_init=include_init, stride=stride
    )

    def _ensemble(
        u_init: Float[Array, "1 * N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "M T 1 *N"]:
        keys = jax.random.split(key, num_samples)
        return _map_paths(lambda k: single_rollout(u_init, k), (keys,), batch_size)

    return _ensemble
