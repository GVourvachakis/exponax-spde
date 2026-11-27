"""All Allen–Cahn interfaces, deterministic limits and legacy multiplicative checks."""

from __future__ import annotations

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exponax_spde import stochastic_structure_factor
from exponax_spde._spectral import fft, ifft
from exponax_spde._stochastic_utils import (
    stochastic_ensemble_rollout,
    stochastic_rollout,
    structure_factor,
)
from exponax_spde.stepper.stochastic import StochasticAllenCahn

# ── Double precision is important for convergence and invariant-measure tests
jax.config.update("jax_enable_x64", True)


# ============================================================================
# Shared helpers
# ============================================================================

PARAMS_1D = dict(
    num_spatial_dims=1,
    domain_extent=1.0,
    num_points=64,
    dt=1e-3,
    diffusivity=0.05,
    lambda_=1.0,
)
PARAMS_2D = dict(
    num_spatial_dims=2,
    domain_extent=1.0,
    num_points=32,
    dt=1e-3,
    diffusivity=0.05,
    lambda_=1.0,
)
KEY0 = jax.random.PRNGKey(42)


def _random_ic(params: dict, key=KEY0) -> jnp.ndarray:
    N = params["num_points"]
    d = params["num_spatial_dims"]
    return 0.3 * jax.random.normal(key, (1,) + (N,) * d)


def _get_deterministic_allen_cahn(p: dict):
    """Construct the built-in deterministic Allen-Cahn stepper.

    Upstream API (exponax.stepper.reaction.AllenCahn):
        AllenCahn(num_spatial_dims, domain_extent, num_points, dt,
                  *, diffusivity, first_order_coefficient,
                    third_order_coefficient, order, dealiasing_fraction)

    Mapping to match ν Δu + λ(u − u³):
        diffusivity             = diffusivity
        first_order_coefficient = lambda_
        third_order_coefficient = -lambda_
    """
    return ex.stepper.reaction.AllenCahn(
        num_spatial_dims=p["num_spatial_dims"],
        domain_extent=p["domain_extent"],
        num_points=p["num_points"],
        dt=p["dt"],
        diffusivity=p["diffusivity"],
        first_order_coefficient=p["lambda_"],
        third_order_coefficient=-p["lambda_"],
        order=1,
        dealiasing_fraction=2.0 / 3.0,
    )


def _make_stochastic(p: dict, **kwargs) -> StochasticAllenCahn:
    return StochasticAllenCahn(
        num_spatial_dims=p["num_spatial_dims"],
        domain_extent=p["domain_extent"],
        num_points=p["num_points"],
        dt=p["dt"],
        diffusivity=p["diffusivity"],
        lambda_=p["lambda_"],
        **kwargs,
    )


@pytest.mark.parametrize("order", [-1, 0, 2, 3, 4])
def test_unsupported_order(order):
    with pytest.raises(ValueError, match="order=1"):
        _make_stochastic(PARAMS_1D, order=order)


def test_default_order_matches_explicit_order_one():
    u0 = _random_ic(PARAMS_1D)
    default = _make_stochastic(PARAMS_1D)
    explicit = _make_stochastic(PARAMS_1D, order=1)
    np.testing.assert_array_equal(default(u0, key=KEY0), explicit(u0, key=KEY0))


# ============================================================================
# Test 1 — Deterministic limit (σ → 0)
# ============================================================================


class TestDeterministicLimit:
    """StochasticAllenCahn(sigma=0) ≈ deterministic AllenCahn."""

    def _det_stepper(self, p):
        det = _get_deterministic_allen_cahn(p)
        if det is None:
            pytest.skip("Could not construct deterministic Allen-Cahn stepper.")
        return det

    def test_single_step_1d(self):
        p = PARAMS_1D
        u0 = _random_ic(p)
        det = self._det_stepper(p)
        u_det = det(u0)
        sto = _make_stochastic(p, sigma=0.0, use_taming=False)
        u_sto = sto(u0, key=KEY0)
        assert jnp.allclose(u_det, u_sto, atol=1e-5), (
            f"Max diff: {jnp.max(jnp.abs(u_det - u_sto)):.2e}"
        )

    def test_rollout_1d(self):
        p = PARAMS_1D
        u0 = _random_ic(p)
        T = 100
        det = self._det_stepper(p)
        det_trj = ex.rollout(det, T, include_init=True)(u0)
        sto = _make_stochastic(p, sigma=0.0, use_taming=False)
        sto_rollout = stochastic_rollout(sto, T, include_init=True)
        sto_trj = jax.jit(sto_rollout)(u0, KEY0)
        assert jnp.max(jnp.abs(det_trj - sto_trj)) < 1e-4

    def test_single_step_2d(self):
        p = PARAMS_2D
        u0 = _random_ic(p)
        det = self._det_stepper(p)
        u_det = det(u0)
        sto = _make_stochastic(p, sigma=0.0, use_taming=False)
        u_sto = sto(u0, key=KEY0)
        assert jnp.allclose(u_det, u_sto, atol=1e-5)

    @pytest.mark.parametrize("p", [PARAMS_1D, PARAMS_2D], ids=["1d", "2d"])
    @pytest.mark.parametrize("space", ["physical", "fourier"])
    def test_exact_upstream_order_one(self, p, space):
        det = _get_deterministic_allen_cahn(p)
        sto = _make_stochastic(
            p, sigma=0.0, use_taming=False, order=1, dealiasing_fraction=2.0 / 3.0
        )
        initial = _random_ic(p)
        if space == "fourier":
            initial = fft(initial, num_spatial_dims=p["num_spatial_dims"])
            det = det.step_fourier
            sto = sto.step_fourier

        np.testing.assert_array_equal(sto(initial, key=KEY0), det(initial))
        np.testing.assert_array_equal(
            stochastic_rollout(sto, 50, include_init=False)(initial, KEY0),
            ex.rollout(det, 50, include_init=False)(initial),
        )

    def test_sigma_zero_no_noise(self):
        """σ=0: two calls with different keys must give identical results."""
        p = PARAMS_1D
        u0 = _random_ic(p)
        sto = _make_stochastic(p, sigma=0.0)
        u1 = sto(u0, key=jax.random.PRNGKey(0))
        u2 = sto(u0, key=jax.random.PRNGKey(99))
        assert jnp.allclose(u1, u2, atol=1e-12), (
            "Different keys with sigma=0 produced different results — noise leaking."
        )

    def test_step_raises(self):
        """BaseStepper.step() must be disabled (raises NotImplementedError)."""
        sto = _make_stochastic(PARAMS_1D, sigma=0.1)
        u0 = _random_ic(PARAMS_1D)
        with pytest.raises(NotImplementedError):
            sto.step(u0)


# ============================================================================
# Streamed structure factor equivalence
# ============================================================================


@pytest.mark.parametrize(
    ("d", "n", "m", "t"), [(1, 32, 16, 200), (2, 16, 8, 100)], ids=["1d", "2d"]
)
@pytest.mark.parametrize("noise_type", ["additive", "multiplicative"])
def test_streamed_structure_factor_matches_ensemble(d, n, m, t, noise_type):
    params = dict(PARAMS_1D, num_spatial_dims=d, num_points=n)
    stepper = _make_stochastic(params, noise_type=noise_type)
    u0 = _random_ic(params)
    burn_in_fraction = 0.337
    ensemble = jax.jit(stochastic_ensemble_rollout(stepper, t, m, include_init=False))(
        u0, KEY0
    )
    expected = structure_factor(ensemble, burn_in_fraction=burn_in_fraction)
    streamed = jax.jit(
        stochastic_structure_factor(stepper, t, m, burn_in_fraction=burn_in_fraction)
    )(u0, KEY0)
    np.testing.assert_allclose(streamed, expected, rtol=1e-12, atol=0)


# ============================================================================
# JAX-compatibility tests
# ============================================================================


class TestJAXCompatibility:
    """JIT, vmap, PyTree, and interface tests — all fast (no slow mark)."""

    def test_jit_1d(self):
        ste = _make_stochastic(PARAMS_1D, sigma=0.1)
        u0 = _random_ic(PARAMS_1D)
        u1 = jax.jit(lambda u, k: ste(u, key=k))(u0, KEY0)
        assert u1.shape == u0.shape
        assert jnp.all(jnp.isfinite(u1))

    def test_jit_2d(self):
        p = PARAMS_2D
        ste = _make_stochastic(p, sigma=0.1, noise_alpha=1.5)
        N = p["num_points"]
        u0 = jnp.zeros((1, N, N))
        u1 = jax.jit(lambda u, k: ste(u, key=k))(u0, KEY0)
        assert u1.shape == u0.shape
        assert jnp.all(jnp.isfinite(u1))

    def test_vmap_over_keys(self):
        ste = _make_stochastic(PARAMS_1D, sigma=0.1)
        u0 = _random_ic(PARAMS_1D)
        M = 8
        keys = jax.random.split(KEY0, M)
        u_batch = jax.jit(jax.vmap(lambda k: ste(u0, key=k)))(keys)
        assert u_batch.shape == (M,) + u0.shape
        assert not jnp.allclose(u_batch[0], u_batch[1])

    def test_vmap_over_initial_conditions(self):
        ste = _make_stochastic(PARAMS_1D, sigma=0.1)
        B = 4
        N = PARAMS_1D["num_points"]
        u_batch = jax.random.normal(KEY0, (B, 1, N)) * 0.3
        keys = jax.random.split(KEY0, B)
        u_next = jax.jit(jax.vmap(lambda u, k: ste(u, key=k)))(u_batch, keys)
        assert u_next.shape == (B, 1, N)
        assert jnp.all(jnp.isfinite(u_next))

    def test_stacked_steppers_vmap(self):
        """
        vmap over steppers with different diffusivity values (all static fields equal).
        """
        N = 32
        diffusivitys = [0.02, 0.05, 0.08]
        steppers = [
            StochasticAllenCahn(
                num_spatial_dims=1,
                domain_extent=1.0,
                num_points=N,
                dt=1e-3,
                diffusivity=eps,
                sigma=0.1,
                noise_alpha=1.0,
            )
            for eps in diffusivitys
        ]
        stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *[eqx.filter(s, eqx.is_array) for s in steppers],
        )
        stacked = eqx.combine(
            stacked, eqx.filter(steppers[0], eqx.is_array, inverse=True)
        )
        u0 = jnp.zeros((1, N))
        keys = jax.random.split(jax.random.PRNGKey(0), len(diffusivitys))
        u_batch = eqx.filter_vmap(lambda ste, k: ste(u0, key=k))(stacked, keys)
        assert u_batch.shape == (len(diffusivitys), 1, N)
        assert jnp.all(jnp.isfinite(u_batch))

    def test_pytree_leaves_finite(self):
        ste = _make_stochastic(PARAMS_1D, sigma=0.1, noise_alpha=1.5)
        for leaf in jax.tree_util.tree_leaves(ste):
            if not isinstance(leaf, jnp.ndarray):
                continue
            assert jnp.all(jnp.isfinite(leaf)), f"Non-finite leaf: {leaf}"

    def test_rollout_scan(self):
        ste = _make_stochastic(PARAMS_1D, sigma=0.1, noise_alpha=1.5)
        u0 = jnp.zeros((1, PARAMS_1D["num_points"]))
        rollout_fn = jax.jit(stochastic_rollout(ste, 50, include_init=True))
        trj = rollout_fn(u0, KEY0)
        assert trj.shape == (51, 1, PARAMS_1D["num_points"])
        assert jnp.all(jnp.isfinite(trj))

    def test_step_fourier_1d(self):
        """step_fourier works and gives the same result as __call__."""
        ste = _make_stochastic(PARAMS_1D, sigma=0.1, noise_alpha=1.5)
        u0 = _random_ic(PARAMS_1D)
        u_hat = fft(u0, num_spatial_dims=1)
        u_hat_next = ste.step_fourier(u_hat, key=KEY0)
        u_next_from_fourier = ifft(
            u_hat_next, num_spatial_dims=1, num_points=PARAMS_1D["num_points"]
        ).real
        assert u_next_from_fourier.shape == u0.shape
        assert jnp.all(jnp.isfinite(u_next_from_fourier))
        np.testing.assert_array_equal(u_next_from_fourier, ste(u0, key=KEY0))
