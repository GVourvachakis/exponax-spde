import jax
import jax.numpy as jnp
import numpy as np
import pytest
from exponax import fft, spectral

from exponax_spde import stochastic_structure_factor
from exponax_spde.stepper import StochasticAllenCahn


def _modal_geometry(d, n, extent):
    """Integer modes, self-conjugacy and the last-axis rFFT multiplicities."""
    axes = [np.fft.fftfreq(n) * n] * (d - 1) + [np.fft.rfftfreq(n) * n]
    modes = np.stack(np.meshgrid(*axes, indexing="ij"))
    special = modes == 0
    if n % 2 == 0:
        special |= np.abs(modes) == n // 2
    k2 = np.sum((2 * np.pi * modes / extent) ** 2, axis=0)
    return k2, (~special).sum(axis=0), np.where(special[-1], 1, 2)


def _convolution_factor(linear, h):
    z = 2 * h * linear
    return np.divide(np.expm1(z), z, out=np.ones_like(z), where=z != 0) * h


def _sample_mean(observe, key, m=8192, chunk=256):
    """Reduce independent draws within each chunk before transferring to host."""

    @jax.jit
    def chunk_sum(key):
        return jax.vmap(observe)(jax.random.split(key, chunk)).sum(axis=0)

    total = 0
    for subkey in jax.random.split(key, m // chunk):
        total = total + np.asarray(chunk_sum(subkey))
    return total / m


def _stationary_power(d, n, h, steps, key):
    # Common h*steps=5.12, first quarter discarded. At ν=.1 and M=256
    # the first nonzero mode has RSE <2%, and all higher modes have lower RSE.
    nu, sigma, alpha, m, chunk = 0.1, 0.3, 1.5, 256, 32
    fraction = 0.25
    model = StochasticAllenCahn(
        d, 1.0, n, h, diffusivity=nu, lambda_=0, sigma=sigma, noise_alpha=alpha
    )
    remaining = steps - int(steps * fraction)
    initial = jnp.zeros((1,) + (n,) * d)
    spectrum = jax.jit(
        stochastic_structure_factor(model, steps, chunk, burn_in_fraction=fraction)
    )
    total = 0
    for subkey in jax.random.split(key, m // chunk):
        total = total + np.asarray(spectrum(initial, subkey))[0]
    observed = total / (m // chunk)

    k2, nonspecial, _ = _modal_geometry(d, n, 1.0)
    rate = 2 * nu * k2
    target = np.divide(
        4.0**nonspecial * sigma**2 * (1 + k2) ** (-alpha),
        rate,
        out=np.ones_like(k2),
        where=rate != 0,
    )
    lag = np.arange(1, remaining)
    correlation = np.exp(-rate[..., None] * h * lag)
    neff = (
        m
        * remaining**2
        / (remaining + 2 * (correlation * (remaining - lag)).sum(axis=-1))
    )
    rel_se = np.sqrt(np.where(nonspecial == 0, 2, 1) / neff)
    burn = steps - remaining
    bias = np.exp(-rate[..., None] * h * np.arange(burn + 1, steps + 1)).mean(axis=-1)
    return observed, target, rel_se, bias, k2 > 0


def _compare_stationary(first, second, mask, description):
    y1, target, se1, bias1, _ = first
    y2, target2, se2, bias2, _ = second
    np.testing.assert_allclose(target[mask], target2[mask], rtol=1e-14, atol=0)
    err1 = np.abs(y1 / target - 1)
    err2 = np.abs(y2 / target - 1)
    difference = np.abs(y1 - y2) / target
    gap_se = np.sqrt(se1**2 + se2**2)
    threshold = 5 * gap_se + bias1 + bias2
    print(
        f"{description}: max reference relative errors={err1[mask].max():.4f}, "
        f"{err2[mask].max():.4f}; max difference={difference[mask].max():.4f}; "
        f"largest paired tolerance={threshold[mask].max():.4f}; "
        f"largest paired SE={gap_se[mask].max():.4f}"
    )
    assert np.max(gap_se[mask]) < 0.03
    assert np.all(err1[mask] <= (5 * se1 + bias1)[mask])
    assert np.all(err2[mask] <= (5 * se2 + bias2)[mask])
    assert np.all(difference[mask] <= threshold[mask])


@pytest.mark.slow
@pytest.mark.parametrize(("d", "n"), [(1, 64), (2, 32)], ids=["1d", "2d"])
def test_invariant_measure(d, n):
    """OU stationarity gives S=4**m σ² q/(2ν|k|² ℓ**d) for nonzero modes.

    Divide one-step raw power by 1-exp(-2ν|k|²h), then by D².
    Correlation of squared modes is exp(-2ν|k|²h lag), giving the
    effective sample size for the two h values at the same horizon.
    Use five SEs plus the exponentially decaying zero-start bias.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(701 + d))
    first = _stationary_power(d, n, 0.02, 256, k1)
    second = _stationary_power(d, n, 0.04, 128, k2)
    _compare_stationary(first, second, first[-1], f"d={d}, h=0.02 and 0.04")


@pytest.mark.slow
@pytest.mark.parametrize(("d", "n"), [(1, 64), (2, 32)], ids=["1d", "2d"])
def test_structure_factor_grid_convergence(d, n):
    """At λ=0 both grids share the OU S=4**m σ² q/(2ν|k|² ℓ**d).

    Restrict to the same signed physical modes below the small-grid Nyquist.
    Use five SEs with both independent ensembles' autocorrelation-adjusted
    variances, plus the two transient bounds from the zero initial state.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(799 + d))
    small = _stationary_power(d, n, 0.02, 256, k1)
    large = _stationary_power(d, 2 * n, 0.02, 256, k2)
    cutoff = n // 4
    full = np.r_[np.arange(cutoff + 1), np.arange(-cutoff, 0)]
    half = np.arange(cutoff + 1)
    index = np.ix_(*([full] * (d - 1) + [half]))
    small = tuple(a[index] for a in small)
    large = tuple(a[index] for a in large)
    _compare_stationary(small, large, small[-1], f"d={d}, N={n} and {2 * n}")


def test_white_variance_and_scaling():
    """At L=0, inverse FFT gives independent N(0,σ²h/dx) grid values.

    With M=8192, relative SE sqrt(2/(M N)) is at most 0.28%. The 1.8%
    variance band covers six SEs. Across a factor four in h or N,
    propagating the log-variance errors gives slope SE below 0.003,
    against the 0.025 band around each analytical slope of one.
    """
    ns = (32, 64, 128)
    steps = (0.001, 0.002, 0.004)
    ratios = np.empty((3, 3))
    measured = np.empty_like(ratios)
    sigma = 0.3
    for i, n in enumerate(ns):
        u0 = jnp.zeros((1, n))
        for j, h in enumerate(steps):
            model = StochasticAllenCahn(
                1, 1.0, n, h, diffusivity=0, lambda_=0, noise_alpha=0, sigma=sigma
            )
            measured[i, j] = _sample_mean(
                lambda key, model=model, u0=u0: jnp.mean(model(u0, key=key) ** 2),
                jax.random.PRNGKey(812 + 3 * i + j),
            )
            ratios[i, j] = measured[i, j] / (sigma**2 * h * n)
    time_slopes = np.polyfit(np.log(steps), np.log(measured.T), 1)[0]
    grid_slopes = np.polyfit(np.log(ns), np.log(measured), 1)[0]
    print(
        f"white variance ratios={ratios.tolist()}; "
        f"h slopes={time_slopes.tolist()}; N slopes={grid_slopes.tolist()}"
    )
    np.testing.assert_allclose(ratios, 1, rtol=0.018, atol=0)
    np.testing.assert_allclose(time_slopes, 1, rtol=0, atol=0.025)
    np.testing.assert_allclose(grid_slopes, 1, rtol=0, atol=0.025)


@pytest.mark.parametrize(
    ("lam", "mode"),
    [(0, 0), (0, 32), (2, 1)],
    ids=["linear-zero", "strongly-damped", "unstable"],
)
def test_ou_mode_variance(lam, mode):
    """Itô isometry gives raw power P σ² q F_h(L)/dx.

    Each complex mode has relative SE 1/sqrt(M); DC and Nyquist are
    real and have sqrt(2/M). A five-SE band covers the selected mode.
    """
    n, h, alpha, sigma, m = 64, 0.1, 0.75, 0.3, 8192
    nu = 0.01 if lam else 1.0
    stepper = StochasticAllenCahn(
        1, 1.0, n, h, diffusivity=nu, lambda_=lam, sigma=sigma, noise_alpha=alpha
    )
    u0 = jnp.zeros((1, n))
    u_hat = fft(u0)
    key = jax.random.PRNGKey(731 + mode)
    linear = lam - nu * (2 * np.pi * mode) ** 2
    q = (1 + (2 * np.pi * mode) ** 2) ** (-alpha)
    target = n**2 * sigma**2 * q * _convolution_factor(linear, h)
    actual = _sample_mean(
        lambda rng: jnp.abs(stepper.step_fourier(u_hat, key=rng)[0, mode]) ** 2, key, m
    )
    rel = abs(actual / target - 1)
    se = np.sqrt((2 if mode in (0, n // 2) else 1) / m)
    out = jax.jit(lambda x, k: stepper(x, key=k))(u0, key)
    print(
        f"Lh={linear * h:.6g}; variance={actual:.9g}; expected={target:.9g}; "
        f"relative SE={se:.5f}; NaNs={int(jnp.isnan(out).sum())}"
    )
    assert np.isfinite(np.asarray(out)).all()
    assert rel <= 5 * se


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("n", [7, 8])
def test_real_fft_conventions_and_colour(d, n):
    """A real FFT has Hermitian boundary planes and P-normalised Parseval.

    A separable cosine has raw corner coefficient 3P/2**d. Multiplying a
    known real draw by the analytic OU amplitude isolates the q filter.
    Checks are deterministic, with margins of a few FFT round-off units.
    """
    extent, nu, lam, sigma, h, alpha = 1.5, 0.05, 0.7, 0.3, 0.012, 1.2
    k2, nonspecial, weights = _modal_geometry(d, n, extent)
    p = n**d
    shape = (1,) + (n,) * d
    stepper = StochasticAllenCahn(
        d, extent, n, h, diffusivity=nu, lambda_=lam, sigma=sigma, noise_alpha=alpha
    )
    key = jax.random.PRNGKey(761)
    white = np.asarray(jax.random.normal(key, shape))
    axes = tuple(range(1, d + 1))
    raw = np.fft.rfftn(white, axes=axes)
    variance = (
        sigma**2
        * (1 + k2) ** (-alpha)
        * _convolution_factor(lam - nu * k2, h)
        / (extent / n) ** d
    )
    expected = raw * np.sqrt(variance)
    actual = np.asarray(stepper.sample_increment(key))
    error = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
    assert error <= 1e-14
    np.testing.assert_allclose(
        np.sum(white**2), np.sum(weights * np.abs(raw) ** 2) / p, rtol=1e-14, atol=0
    )
    reverse = (-np.arange(n)) % n
    for boundary in (0, n // 2) if n % 2 == 0 else (0,):
        plane = actual[0, ..., boundary]
        conjugate = plane.conj()
        for axis in range(d - 1):
            conjugate = conjugate.take(reverse, axis=axis)
        np.testing.assert_allclose(plane, conjugate, rtol=1e-14, atol=1e-14)
    for mode, expected_scale in (
        ("reconstruction", p / weights),
        ("coef_extraction", p / 2.0**nonspecial),
    ):
        np.testing.assert_array_equal(
            spectral.build_scaling_array(d, n, mode=mode)[0], expected_scale
        )
    coords = np.indices(shape[1:])
    field = (2 + 3 * np.prod(np.cos(2 * np.pi * coords / n), axis=0))[None]
    extracted = np.asarray(
        spectral.get_fourier_coefficients(jnp.asarray(field), round=None)
    )
    np.testing.assert_allclose(extracted[(0,) * (d + 1)].real, 2, rtol=1e-14)
    np.testing.assert_allclose(extracted[(0,) + (1,) * d].real, 3, rtol=1e-14)
    print(
        f"d={d}, N={n}: relative sup error={error:.3e}; "
        "Parseval and Hermitian planes passed"
    )


@pytest.mark.parametrize(("d", "steps"), [(1, 15), (2, 10)], ids=["1d", "2d"])
def test_dc_mean_and_variance(d, steps):
    """The spatial mean at λ=0 is Brownian, with V_DC=σ² t/ℓ**d.

    DC survives diffusion unchanged. With M=4096 the mean's SE is
    sqrt(V_DC/M), while sample variance has relative SE sqrt(2/(M-1)).
    Use five SEs for both independent-path moment checks.
    """
    n, h, sigma, m = 8, 0.02, 0.1, 4096
    model = StochasticAllenCahn(
        d, 1.0, n, h, diffusivity=0.05, lambda_=0, sigma=sigma, noise_alpha=1.5
    )
    u_hat = fft(jnp.zeros((1,) + (n,) * d))

    def statistical_sample(key):
        def advance(carry, _):
            u, rng = carry
            rng, draw_key = jax.random.split(rng)
            return (model.step_fourier(u, key=draw_key), rng), None

        (end, _), _ = jax.lax.scan(advance, (u_hat, key), None, length=steps)
        dc = end[(0,) * (d + 1)].real / n**d
        return jnp.array((dc, dc**2))

    mean, second_moment = _sample_mean(
        statistical_sample, jax.random.PRNGKey(730 + d), m
    )
    measured = (second_moment - mean**2) * m / (m - 1)
    target = sigma**2 * steps * h
    mean_band = 5 * np.sqrt(target / m)
    variance_band = 5 * np.sqrt(2 / (m - 1))
    print(
        f"DC d={d}: mean={mean:.7g}, mean band={mean_band:.7g}; "
        f"var={measured:.7g}, target={target:.7g}, rel band={variance_band:.5f}"
    )
    assert abs(mean) <= mean_band
    assert abs(measured / target - 1) <= variance_band


def test_antithetic_odd_symmetry():
    """Odd drift and opposite additive increments yield opposite trajectories."""
    model = StochasticAllenCahn(2, 1.0, 8, 0.005, lambda_=0.7, sigma=0.3)
    positive = fft(jax.random.normal(jax.random.PRNGKey(746), (1, 8, 8)))
    negative = -positive
    for key in jax.random.split(jax.random.PRNGKey(759), 6):
        increment = model.sample_increment(key)
        positive = model.step_with_increment(positive, increment)
        negative = model.step_with_increment(negative, -increment)
    relative = float(jnp.max(jnp.abs(positive + negative)) / jnp.max(jnp.abs(positive)))
    print(f"antithetic relative sup error={relative:.3e}")
    assert relative <= 8 * np.finfo(float).eps


def test_noise_is_active():
    """Independent one-step fields give mean squared difference 2 V_h.

    The grid covariance eigenvalues v_k=σ² q F_h/dx give SE² =
    2 sum_full(v_k²) / (P² M) for mean point energy (times four for
    the paired difference). Use five analytical SEs.
    """
    n, h, nu, lam, sigma, alpha, m = 64, 0.005, 0.05, 0.7, 0.3, 1.5, 8192
    model = StochasticAllenCahn(
        1, 1.0, n, h, diffusivity=nu, lambda_=lam, sigma=sigma, noise_alpha=alpha
    )
    u0 = jnp.zeros((1, n))

    def observe(key):
        a, b = jax.random.split(key)
        return jnp.mean((model(u0, key=a) - model(u0, key=b)) ** 2)

    actual = _sample_mean(observe, jax.random.PRNGKey(743), m)
    k2, _, weights = _modal_geometry(1, n, 1.0)
    v = sigma**2 * (1 + k2) ** (-alpha) * _convolution_factor(lam - nu * k2, h) * n
    target = 2 * np.sum(weights * v) / n
    se = 2 * np.sqrt(2 * np.sum(weights * v**2) / m) / n
    print(f"pair energy={actual:.8g}; expected={target:.8g}; 5 SE={5 * se:.8g}")
    assert abs(actual - target) <= 5 * se
