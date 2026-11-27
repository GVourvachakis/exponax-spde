"""Fourier projection and input checks for the host-side Strang scaffold."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exponax_spde import strang_split_step
from exponax_spde.stepper.stochastic import StochasticAllenCahn


def _identity(u, *, key):
    return u


def _apply(delta, cutoff=0.5):
    half_steps = iter((delta, np.zeros_like(delta)))
    result, _ = strang_split_step(
        _identity,
        lambda state, dt: {"delta_concentration": next(half_steps)},
        jnp.zeros_like(delta),
        {},
        0.01,
        jax.random.key(81),
        num_points=delta.shape[-1],
        mollifier_cutoff=cutoff,
    )
    return np.asarray(result)


def _oracle(delta, cutoff):
    d, n = delta.ndim - 1, delta.shape[-1]
    axes = tuple(range(-d, 0))
    freq = np.fft.fftfreq(n) * n
    inside = np.abs(np.rint(freq)) <= int(cutoff * (n // 2))
    keep = np.logical_and.reduce(
        np.broadcast_arrays(*np.meshgrid(*([inside] * d), sparse=True, indexing="ij"))
    )
    return np.fft.ifftn(np.fft.fftn(delta, axes=axes) * keep, axes=axes).real


@pytest.mark.parametrize("n", [9, 10])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_mollifier_retains_both_frequency_signs(d, n):
    """A unit cosine has a conjugate pair of modes on the full FFT grid."""
    coords = np.meshgrid(*([np.arange(n) / n] * d), sparse=True, indexing="ij")
    for axis in range(d):
        anchor = coords[(axis + 1) % d] if d > 1 else 0
        for sign in (-1, 1):
            for k in (2, 3):
                phase = sign * k * coords[axis] + anchor
                field = np.broadcast_to(np.cos(2 * np.pi * phase), (1,) + (n,) * d)
                np.testing.assert_allclose(
                    _apply(field), field if k == 2 else 0.0, rtol=0, atol=1e-14
                )


@pytest.mark.parametrize("n", [9, 10])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_mollifier_is_real_mean_preserving_projection(d, n):
    """A real 0/1 FFT multiplier commutes with conjugation and squares to itself."""
    rng = np.random.default_rng(244)
    shape = (1,) + (n,) * d
    x = 0.5 + rng.normal(size=shape)
    y = rng.normal(size=shape)
    projected = _apply(x)
    np.testing.assert_allclose(projected, _oracle(x, 0.5), rtol=0, atol=1e-14)
    np.testing.assert_allclose(_apply(projected), projected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        _apply(2 * x - y), 2 * projected - _apply(y), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(projected.mean(), x.mean(), rtol=0, atol=1e-14)
    assert np.isrealobj(projected)
    # Compare in coefficient units, independent of grid volume.
    spectrum = np.fft.fftn(projected[0], norm="forward")
    wrap = (-np.arange(n)) % n
    np.testing.assert_allclose(
        spectrum[np.ix_(*([wrap] * d))], spectrum.conj(), rtol=1e-14, atol=1e-14
    )


def test_strang_split_matches_independent_composition():
    """Dyadic increments with DC/quarter/Nyquist modes filter exactly."""
    n, dt = 16, 0.01
    stepper = StochasticAllenCahn(1, 1.0, n, dt, sigma=0.0, use_taming=False)
    u = jnp.linspace(-0.2, 0.4, n, dtype=jnp.float64)[None, :]
    parity = (-1.0) ** np.arange(n)
    quarter = np.tile([1.0, 0.0, -1.0, 0.0], n // 4)
    increments = (
        (0.125 + 0.25 * parity)[None, :],
        (0.5 * quarter - 0.25 * parity)[None, :],
    )
    calls = []

    def ssa(state, half_dt):
        assert state["index"] == len(calls)
        calls.append(half_dt)
        return {"index": len(calls), "delta_concentration": increments[len(calls) - 1]}

    key = jax.random.key(30)
    actual, state = strang_split_step(
        stepper, ssa, u, {"index": 0}, dt, key, num_points=n
    )
    _, subkey = jax.random.split(key)
    mid = u + _oracle(increments[0], 0.5)
    expected = stepper(mid, key=subkey) + _oracle(increments[1], 0.5)
    np.testing.assert_array_equal(actual, expected)
    assert state["index"] == 2
    assert calls == [dt / 2, dt / 2]


@pytest.mark.parametrize("at_half_step", [0, 1])
@pytest.mark.parametrize("fault", ["missing", "wrong-shape"])
def test_strang_split_rejects_bad_concentration_increment(at_half_step, fault):
    valid = {"delta_concentration": np.zeros((1, 8))}
    bad = {} if fault == "missing" else {"delta_concentration": np.zeros((8,))}
    states = [valid, valid]
    states[at_half_step] = bad
    iterator = iter(states)
    with pytest.raises(ValueError, match="delta_concentration"):
        strang_split_step(
            _identity,
            lambda state, dt: next(iterator),
            jnp.zeros((1, 8)),
            {},
            0.1,
            jax.random.key(5),
            num_points=8,
        )


def test_strang_split_domain_extent_is_deprecated():
    kwargs = dict(
        spectral_stepper=_identity,
        ssa_step_fn=lambda state, dt: {"delta_concentration": np.ones((1, 8))},
        u=jnp.zeros((1, 8)),
        ssa_state={},
        dt=0.1,
        key=jax.random.key(12),
        num_points=8,
    )
    ref, _ = strang_split_step(**kwargs)
    with pytest.warns(DeprecationWarning, match=r"domain_extent.*v0\.3\.0") as record:
        deprecated, _ = strang_split_step(**kwargs, domain_extent=2.0)
    assert record[0].filename == __file__
    np.testing.assert_array_equal(ref, deprecated)
