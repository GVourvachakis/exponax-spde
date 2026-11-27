"""Analytical and coupled-path checks of the multiplicative Itô step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exponax import fft, ifft
from physverify import check_order

from exponax_spde.stepper.stochastic import StochasticAllenCahn


def _stepper(d=1, n=16, h=0.01, **kwargs):
    options = dict(diffusivity=0.0, lambda_=0.0, noise_type="multiplicative", sigma=0.6)
    return StochasticAllenCahn(d, 1.0, n, h, **(options | kwargs))


def _point_variance(d, n, h, alpha):
    k = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
    k_sq = sum(axis**2 for axis in np.meshgrid(*([k] * d), sparse=True, indexing="ij"))
    return h * n**d * np.mean((1 + k_sq) ** (-alpha))


def _sample_mean(observe, samples, seed):
    """Reduce independent draws in blocks without retaining the fields."""
    batch_size = 256
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def block(i):
        keys = jax.random.split(jax.random.fold_in(key, i), batch_size)
        return jax.tree.map(lambda x: jnp.sum(x, axis=0), jax.vmap(observe)(keys))

    assert samples % batch_size == 0
    total = block(0)
    for i in range(1, samples // batch_size):
        total = jax.tree.map(jnp.add, total, block(i))
    return jax.tree.map(lambda x: np.asarray(x) / samples, total)


@pytest.mark.parametrize("milstein", [False, True], ids=["eem", "milstein"])
def test_one_step_conditional_kick(milstein):
    """At L=0, the kick is u₀(aZ + b(Z²−1)) with Z standard normal.

    Its mean is zero, its variance is u₀²(a²+2b²), and its fourth moment is
    u₀⁴(3a⁴+60a²b²+60b⁴). Use 5 analytic SE for each point's sample mean and
    unbiased variance, including the non-Gaussian term when b is nonzero.
    """
    n, h, alpha, sigma, samples = 16, 0.05, 0.75, 0.8, 8192
    stepper = _stepper(n=n, h=h, noise_alpha=alpha, sigma=sigma, use_milstein=milstein)
    u = jnp.linspace(0.2, 0.9, n, dtype=jnp.float64)[None, :]
    u_hat = fft(u)
    a = sigma * np.sqrt(_point_variance(1, n, h, alpha))
    b = 0.5 * a**2 if milstein else 0.0

    def observe(key):
        kick = ifft(stepper.step_fourier(u_hat, key=key) - u_hat, num_points=n)
        return kick, kick**2

    mean, second = _sample_mean(observe, samples, seed=21)
    var = (second - mean**2) * samples / (samples - 1)
    u = np.asarray(u)
    expected = u**2 * (a**2 + 2 * b**2)
    fourth = u**4 * (3 * a**4 + 60 * a**2 * b**2 + 60 * b**4)
    mean_z = np.max(np.abs(mean) / np.sqrt(expected / samples))
    var_se = np.sqrt((fourth - (samples - 3) / (samples - 1) * expected**2) / samples)
    var_z = np.max(np.abs(var - expected) / var_se)
    print(f"conditional kick: max mean z={mean_z:.4f}, max variance z={var_z:.4f}")
    assert mean_z <= 5
    assert var_z <= 5


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("n", [7, 8])
def test_milstein_centring(d, n):
    """Full-spectrum q sums give v_W = h mean(q)/dxᵈ by Parseval.

    A single spatial point has Gaussian variance v_W and its square has
    variance 2v_W². Compare MC with 5 SE and recover the scalar actually
    subtracted at every point using a zero supplied increment, to round-off.
    """
    h, alpha, sigma, samples = 0.05, d / 2 + 0.2, 0.6, 8192
    stepper = _stepper(d, n, h, noise_alpha=alpha, use_milstein=True, sigma=sigma)
    expected = _point_variance(d, n, h, alpha)
    selected = (0,) + (n // 2,) * d

    def observe(key):
        dw = ifft(stepper.sample_increment(key), num_spatial_dims=d, num_points=n)
        return dw[selected] ** 2

    mean = _sample_mean(observe, samples, seed=27)
    z = abs(mean - expected) / (expected * np.sqrt(2 / samples))
    u = jnp.ones((1,) + (n,) * d, dtype=jnp.float64)
    u_hat = fft(u, num_spatial_dims=d)
    centred = stepper.step_with_increment(u_hat, jnp.zeros_like(u_hat)) - u_hat
    np.testing.assert_allclose(
        np.asarray(ifft(centred, num_spatial_dims=d, num_points=n)),
        -0.5 * sigma**2 * expected,
        rtol=1e-12,
        atol=1e-14,
    )
    print(f"centring: empirical={mean:.8g}, target={expected:.8g}, z={z:.4f}")
    assert z <= 5


def test_uniform_propagated_mode_power():
    """A uniform state multiplies each raw ΔŴ mode by σu₀ exp(hL_k).

    The expected raw power is N σ²u₀² q_k h exp(2hL_k)/dx. Use 5 SE
    of the mean power: E/√M in circular modes, √2 E/√M on real modes.
    """
    n, h, nu, alpha, u0, sigma, samples = 16, 0.01, 0.05, 1.0, 0.7, 0.6, 8192
    stepper = _stepper(n=n, h=h, diffusivity=nu, noise_alpha=alpha, sigma=sigma)
    u_hat = fft(jnp.full((1, n), u0, dtype=jnp.float64))

    def observe(key):
        kick = stepper.step_fourier(u_hat, key=key) - u_hat
        return jnp.abs(kick[0]) ** 2

    measured = _sample_mean(observe, samples, seed=81)
    k = 2 * np.pi * np.arange(n // 2 + 1)
    expected = (
        n**2
        * (sigma * u0) ** 2
        * (1 + k**2) ** (-alpha)
        * h
        * np.exp(-2 * nu * k**2 * h)
    )
    se = np.full(n // 2 + 1, 1 / np.sqrt(samples))
    se[[0, -1]] *= np.sqrt(2)
    z = np.max(np.abs(measured / expected - 1) / se)
    print(f"propagated mode power: max z={z:.4f}")
    assert z <= 5


def test_multiplicative_eem_kick_scales_with_sigma():
    """At a fixed increment, doubling σ doubles the kick in float64."""
    u_hat = fft(jnp.linspace(0.2, 0.9, 16, dtype=jnp.float64)[None, :])
    first = _stepper(sigma=0.3, lambda_=0.7, diffusivity=0.01)
    second = _stepper(sigma=0.6, lambda_=0.7, diffusivity=0.01)
    key = jax.random.PRNGKey(84)
    increment = first.sample_increment(key)
    np.testing.assert_array_equal(increment, second.sample_increment(key))
    drift = first.step_with_increment(u_hat, jnp.zeros_like(increment))
    a = np.asarray(first.step_with_increment(u_hat, increment) - drift)
    b = np.asarray(second.step_with_increment(u_hat, increment) - drift)
    relative = np.linalg.norm(b - 2 * a) / np.linalg.norm(b)
    print(f"sigma-linearity: relative difference={relative:.3g}")
    assert relative <= 1e-12


@pytest.mark.parametrize("milstein", [False, True], ids=["eem", "milstein"])
def test_multiplicative_replay_and_jit(milstein):
    """Both kicks and the drift vanish at zero; replay and JIT preserve the formula.

    For a nonzero state allow only relative 1e-12 and absolute 1e-14 for
    floating-point FFT and fused arithmetic differences.
    """
    u = jnp.linspace(-0.2, 0.9, 16, dtype=jnp.float64)[None, :]
    u_hat = fft(u)
    stepper = _stepper(lambda_=0.7, diffusivity=0.01, use_milstein=milstein)
    key = jax.random.PRNGKey(42)
    increment = stepper.sample_increment(key)
    actual = stepper.step_with_increment(u_hat, increment)
    np.testing.assert_array_equal(stepper.step_fourier(u_hat, key=key), actual)
    np.testing.assert_array_equal(stepper(u, key=key), ifft(actual, num_points=16))
    expected_zero = jnp.zeros_like(u_hat)
    np.testing.assert_array_equal(
        stepper.step_with_increment(expected_zero, increment), expected_zero
    )
    compiled = jax.jit(lambda a, b: stepper.step_with_increment(a, b))
    observed = np.asarray(compiled(u_hat, increment))
    error = np.max(np.abs(observed - np.asarray(actual)))
    print(f"zero is exact, key replay is exact, max jit-eager difference={error:.3g}")
    np.testing.assert_allclose(observed, actual, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_milstein_trace_class_threshold(d):
    """The tail ∑ |k|^(-2α) converges exactly when 2α>d."""
    for alpha in (0, d / 2 - 0.05, d / 2, d / 2 + 0.05):
        _stepper(d, 8, noise_alpha=alpha)
        if alpha <= d / 2:
            with pytest.raises(ValueError, match=r"trace-class.*#8"):
                _stepper(d, 8, noise_alpha=alpha, use_milstein=True)
        else:
            _stepper(d, 8, noise_alpha=alpha, use_milstein=True)


def _white_variance(n, h, samples):
    u0, sigma = 0.7, 0.3
    stepper = _stepper(
        n=n, h=h, diffusivity=1e-12, lambda_=1e-12, noise_alpha=0.0, sigma=sigma
    )
    u_hat = fft(jnp.full((1, n), u0, dtype=jnp.float64))
    drift = stepper.step_with_increment(u_hat, jnp.zeros_like(u_hat))

    def observe(key):
        kick = ifft(stepper.step_fourier(u_hat, key=key) - drift, num_points=n)
        return jnp.mean(kick**2)

    measured = float(_sample_mean(observe, samples, seed=601))
    expected = (sigma * u0) ** 2 * h * n
    return measured, expected


def test_white_noise_resolution_and_step_scaling():
    """With negligible drift, MN independent squared kicks estimate σ²u₀² h/dx.

    Compare each ratio with 5√(2/(MN)); compare h and N slopes with 5 SE
    from the log-variance delta method. The same keys across h ensure that
    errors within a fixed N are correlated, so use the sum of the SEs there.
    """
    samples = 8192
    grids, steps = np.array([64, 128]), np.array([0.001, 0.002, 0.004])
    measured = np.empty((len(grids), len(steps)))
    for i, n in enumerate(grids):
        for j, h in enumerate(steps):
            value, target = _white_variance(int(n), float(h), samples)
            measured[i, j] = value
            ratio = value / target
            print(
                f"white noise N={n}, h={h:g}, variance={value:.8g}, ratio={ratio:.6f}"
            )
            assert abs(ratio - 1) <= 5 * np.sqrt(2 / (samples * n))
    se = np.sqrt(2 / (samples * grids))
    time_ratio = steps[-1] / steps[0]
    n_ratio = grids[-1] / grids[0]
    time_slopes = np.log(measured[:, -1] / measured[:, 0]) / np.log(time_ratio)
    n_slopes = np.log(measured[1] / measured[0]) / np.log(n_ratio)
    print(f"white noise slopes: h={time_slopes}, N={n_slopes}")
    assert np.all(np.abs(time_slopes - 1) <= 5 * 2 * se / np.log(time_ratio))
    assert np.all(np.abs(n_slopes - 1) <= 5 * se.sum() / np.log(n_ratio))


@pytest.mark.slow
def test_coupled_geometric_brownian_motion_orders():
    """The near-DC covariance reduces the uniform-state model to Itô GBM.

    Use the same W_T across schemes and levels for the exact exponential
    endpoint, with 32768 paths for RMS relative SE below 3%. Bootstrap
    paired squared errors across all refinements and require lower 95%
    slopes of 0.40 and 0.85 for orders 1/2 and 1.
    """
    n, sigma, u0, alpha = 8, 0.5, 0.8, 20
    levels = np.array([16, 32, 64, 128, 256])
    samples, batch, horizon = 32768, 256, 1.0
    block_steps = levels[-1] // levels[0]
    hs = horizon / levels
    methods = [
        tuple(
            _stepper(
                n=n,
                h=float(h),
                noise_alpha=alpha,
                diffusivity=0.05,
                sigma=sigma,
                use_milstein=milstein,
            )
            for h in hs
        )
        for milstein in (False, True)
    ]
    fine = methods[0][-1]
    initial = fft(jnp.full((batch, 1, n), u0, dtype=jnp.float64), num_spatial_dims=1)
    ratios = levels[-1] // levels

    def advance(stepper, initial, increments):
        apply = jax.vmap(stepper.step_with_increment)
        return jax.lax.scan(
            lambda state, dw: (apply(state, dw), None), initial, increments
        )[0]

    @jax.jit
    def sample_batch(key):
        def block(carry, index):
            states, brownian = carry
            keys = jax.random.split(
                jax.random.fold_in(key, index), (block_steps, batch)
            )
            sequential = jax.vmap(jax.vmap(fine.sample_increment))(keys)
            increments = [
                jax.vmap(jax.vmap(fine.coarsen_increments, in_axes=1))(
                    sequential.reshape(-1, int(r), *sequential.shape[1:])
                )
                if r > 1
                else sequential
                for r in ratios
            ]
            next_states = tuple(
                tuple(
                    advance(stepper, state, dw)
                    for stepper, state, dw in zip(
                        steppers, path, increments, strict=True
                    )
                )
                for steppers, path in zip(methods, states, strict=True)
            )
            return (
                next_states,
                brownian + sequential[..., 0, 0].real.sum(axis=0) / n,
            ), None

        states = tuple(tuple(initial for _ in levels) for _ in methods)
        (states, brownian), _ = jax.lax.scan(
            block,
            (states, jnp.zeros((batch,), dtype=jnp.float64)),
            jnp.arange(levels[0]),
        )
        target = u0 * jnp.exp(sigma * brownian - 0.5 * sigma**2 * horizon)
        terminal = jnp.asarray(states)
        squared_error = (terminal[..., 0, 0].real / n - target) ** 2
        uniformity = jnp.max(jnp.abs(terminal[..., 0, 1:])) / n
        return squared_error, uniformity

    errors = []
    uniformity = 0.0
    key = jax.random.PRNGKey(224)
    for i in range(samples // batch):
        squared, deviation = sample_batch(jax.random.fold_in(key, i))
        errors.append(np.asarray(squared))
        uniformity = max(uniformity, float(deviation))
    squared = np.concatenate(errors, axis=-1)
    mse = squared.mean(axis=-1)
    relative_se = squared.std(axis=-1, ddof=1) / (2 * np.sqrt(samples) * mse)
    rng = np.random.default_rng(225)
    log_h = np.log(hs)
    weights = (log_h - log_h.mean()) / np.sum((log_h - log_h.mean()) ** 2)
    slopes = np.empty((2, 500))
    for i in range(slopes.shape[1]):
        indices = rng.integers(samples, size=samples)
        slopes[:, i] = 0.5 * np.log(squared[..., indices].mean(axis=-1)) @ weights
    lower = np.quantile(slopes, 0.025, axis=1)
    amplitude = (1 + (2 * np.pi) ** 2) ** (-alpha / 2)
    print(
        f"GBM: first non-DC amplitude/DC={amplitude:.5g}, "
        f"max terminal mode amplitude={uniformity:.3g}"
    )
    assert uniformity <= 1e-14
    assert np.all(relative_se <= 0.03)
    for rms, se, bound, floor, expected in zip(
        np.sqrt(mse), relative_se, lower, (0.4, 0.85), (0.5, 1.0), strict=True
    ):
        result = check_order(hs, rms, expected=expected)
        print(
            f"GBM RMS={rms}, relative SE={se}, lower95={bound:.5f}; "
            f"{result.status}: {result.message}"
        )
        assert bound >= floor
        assert result.status == "PASS", result.message
