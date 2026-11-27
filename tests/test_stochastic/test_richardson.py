"""Weak errors at a fixed spatial grid and paired sampling uncertainty."""

from dataclasses import dataclass, replace

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physverify import check_order

from exponax_spde import richardson_weak_extrapolation
from exponax_spde.stepper.stochastic import StochasticAllenCahn

M = 4096
BATCH = 128
KEY = jax.random.key(17537)
RATE = 0.7
SIGMA = 0.0005


@dataclass(frozen=True)
class EulerMode:
    """Linear scalar Euler step, represented by the N=1 real Fourier mode."""

    dt: float
    sigma: float = SIGMA
    num_spatial_dims: int = 1
    num_points: int = 1
    domain_extent: float = 1.0
    noise_alpha: float = 0.0
    noise_type: str = "additive"

    def sample_increment(self, key):
        return self.sigma * jnp.sqrt(self.dt) * jax.random.normal(key)

    def coarsen_increments(self, increments):
        return jnp.sum(increments, axis=0)

    def step_with_increment(self, state, increment):
        return (1 - RATE * self.dt) * state + increment

    def __call__(self, state, *, key):
        return self.step_with_increment(state, self.sample_increment(key))


def test_weak_order_weights_and_observable():
    """Over T=1, deterministic steps h³ leave h² bias, cancelled for p=2.

    A signed-unit kick gives E[X²]=1 exactly at one step from X=0,
    regardless of the finite sample mean of X.
    """

    def stepper(dt):
        return lambda u, *, key: u + dt**3

    u0 = jnp.zeros((1, 1))
    for p, expected in [(1.0, -1 / 32), (2.0, 0.0)]:
        result = richardson_weak_extrapolation(
            stepper(0.25), stepper(0.125), u0, 4, KEY, weak_order=p, num_samples=5
        )
        np.testing.assert_array_equal(result["mean_rich"], expected)
        for name in ("se_coarse", "se_fine", "se_rich", "se_difference"):
            np.testing.assert_array_equal(result[name], 0.0)

    def signed_unit(u, *, key):
        return u + jnp.where(jax.random.bernoulli(key), 1.0, -1.0)

    result = richardson_weak_extrapolation(
        signed_unit,
        signed_unit,
        u0,
        1,
        KEY,
        observable=lambda u: u[0, 0] ** 2,
        num_samples=5,
    )
    assert result["mean_coarse"] == 1.0
    assert result["var_coarse"] == 0.0

    result = richardson_weak_extrapolation(
        signed_unit, signed_unit, u0, 1, KEY, num_samples=7
    )
    assert float(result["var_coarse"][0, 0]) > 0.0
    vc, vf = (float(result[f"var_{name}"][0, 0]) / 6 for name in ("coarse", "fine"))
    for name, expected_variance in (
        ("coarse", vc),
        ("fine", vf),
        ("rich", vc + 4 * vf),
        ("difference", vc + vf),
    ):
        assert float(result[f"se_{name}"][0, 0]) ** 2 == pytest.approx(
            expected_variance, rel=1e-14
        )


@pytest.mark.parametrize("weak_order", [0.0, -0.5, np.inf, np.nan, "2.0"])
def test_invalid_weak_order(weak_order):
    with pytest.raises(ValueError, match="weak_order"):
        richardson_weak_extrapolation(
            EulerMode(0.25),
            EulerMode(0.125),
            jnp.ones((1, 1)),
            1,
            KEY,
            num_samples=2,
            weak_order=weak_order,
        )


@pytest.mark.parametrize("num_samples", [0, 1])
def test_insufficient_samples(num_samples):
    with pytest.raises(ValueError, match="num_samples"):
        richardson_weak_extrapolation(
            EulerMode(0.25),
            EulerMode(0.125),
            jnp.ones((1, 1)),
            1,
            KEY,
            num_samples=num_samples,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt", 0.124),
        ("num_spatial_dims", 2),
        ("num_points", 2),
        ("domain_extent", 2.0),
        ("noise_type", "multiplicative"),
        ("sigma", 0.1),
        ("noise_alpha", 1.5),
    ],
)
def test_coupled_requires_matching_parameters(field, value):
    with pytest.raises(ValueError, match=field):
        richardson_weak_extrapolation(
            EulerMode(0.25),
            replace(EulerMode(0.125), **{field: value}),
            jnp.ones((1, 1)),
            1,
            KEY,
            coupled=True,
            num_samples=2,
        )


def test_coupled_requires_increment_protocol():
    with pytest.raises(ValueError, match="fine.sample_increment"):
        richardson_weak_extrapolation(
            EulerMode(0.25),
            lambda u, *, key: u,
            jnp.ones((1, 1)),
            1,
            KEY,
            coupled=True,
            num_samples=2,
        )


def test_coupled_validates_only_concrete_metadata():
    @jax.jit
    def run(sigma):
        return richardson_weak_extrapolation(
            EulerMode(0.25),
            EulerMode(0.125 * (1 + 4e-13), sigma=sigma),
            jnp.ones((1, 1)),
            2,
            KEY,
            coupled=True,
            num_samples=5,
        )["mean_rich"]

    assert np.isfinite(run(SIGMA)).all()


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_coupled_batches(batch_size):
    options = dict(num_samples=7, coupled=True)
    args = EulerMode(0.25), EulerMode(0.125), jnp.ones((1, 1)), 3, KEY
    reference = richardson_weak_extrapolation(*args, **options)
    result = richardson_weak_extrapolation(*args, batch_size=batch_size, **options)
    assert result.keys() == reference.keys()
    for level in ("coarse", "fine"):
        np.testing.assert_allclose(
            result[f"se_{level}"] ** 2, result[f"var_{level}"] / 6, rtol=1e-14
        )
    for name in reference:
        np.testing.assert_allclose(
            result[name], reference[name], rtol=1e-12, atol=1e-15
        )


def euler_moments(h, n):
    j = np.arange(n)
    bc, bf = 1 - RATE * h, 1 - RATE * h / 2
    v_c = SIGMA**2 * h * np.sum(bc ** (2 * j))
    v_f = SIGMA**2 * h / 2 * np.sum(bf ** (2 * np.arange(2 * n)))
    covariance = SIGMA**2 * h / 2 * np.sum(bc**j * (bf ** (2 * j) + bf ** (2 * j + 1)))
    return bc**n, bf ** (2 * n), v_c, v_f, covariance


def replay_pairs(key, h, n):
    """Return terminal scalars from the prescribed keys for pair resampling."""

    def one_path(path_key):
        def advance(carry, _):
            coarse, fine, rng = carry
            rng, a = jax.random.split(rng)
            rng, b = jax.random.split(rng)
            w1 = SIGMA * jnp.sqrt(h / 2) * jax.random.normal(a)
            w2 = SIGMA * jnp.sqrt(h / 2) * jax.random.normal(b)
            bc, bf = 1 - RATE * h, 1 - RATE * h / 2
            return (bc * coarse + (w1 + w2), bf * (bf * fine + w1) + w2, rng), None

        (coarse, fine, _), _ = jax.lax.scan(
            advance, (1.0, 1.0, path_key), None, length=n
        )
        return jnp.stack((coarse, 2 * fine - coarse))

    return jax.jit(lambda keys: jax.lax.map(one_path, keys, batch_size=BATCH))(
        jax.random.split(key, M)
    )


@pytest.mark.slow
def test_weak_signal_and_variance_reduction():
    """Analytic Euler means are bcⁿ, bf²ⁿ, with b=1−0.7h at each step size.

    Independent Wiener terms give geometric-sum variances. Summing each
    fine pair gives covariance proportional to sum bcʲ(bf²ʲ+bf²ʲ⁺¹).
    Centre means within 5 SE and variances within 5 sqrt(2/(M−1)).
    The variance-ratio band adds the squared relative SEs for independent
    samples by the delta method. Resample whole path pairs across levels
    for two-sided order intervals centred on the analytic finite-level slopes.
    """
    hs = np.array([0.25, 0.125, 0.0625])
    exact = np.exp(-RATE)
    fixed = np.array([euler_moments(h, round(1 / h)) for h in hs])
    weights = np.log(hs) - np.log(hs).mean()
    weights /= weights @ weights
    expected_errors = np.abs(
        np.stack((fixed[:, 0], 2 * fixed[:, 1] - fixed[:, 0]), axis=1) - exact
    )
    predicted_orders = weights @ np.log(expected_errors)
    bands = predicted_orders[:, None] + np.array([-0.20, 0.20])
    key_independent, key_pairs = jax.random.split(KEY)

    pairs = []
    actual_errors = []
    for h, (mean_c, mean_f, v_c, v_f, cov) in zip(hs, fixed, strict=True):
        n = round(1 / h)
        args = EulerMode(h), EulerMode(h / 2), jnp.ones((1, 1)), n
        options = dict(num_samples=M, batch_size=BATCH, observable=lambda u: u[0, 0])
        independent = richardson_weak_extrapolation(*args, key_independent, **options)
        paired = richardson_weak_extrapolation(
            *args, key_pairs, coupled=True, **options
        )
        v_diff = v_c + v_f
        v_rich = 4 * v_f + v_c
        true_means = np.array([mean_c, mean_f, 2 * mean_f - mean_c])
        se_var = np.sqrt(2 / (M - 1))
        for result, covariance in ((independent, 0.0), (paired, cov)):
            marginal_var = np.array([v_c, v_f])
            errors = np.sqrt(
                np.array([v_c, v_f, v_rich - 4 * covariance, v_diff - 2 * covariance])
                / M
            )
            measured = np.array(
                [result[f"mean_{key}"] for key in ("coarse", "fine", "rich")]
            )
            assert np.all(np.abs(measured - true_means) <= 5 * errors[:3])
            np.testing.assert_allclose(
                [result["var_coarse"], result["var_fine"]],
                marginal_var,
                rtol=5 * se_var,
            )
            # Squaring the estimated SE compares like Gaussian variance estimates.
            np.testing.assert_allclose(
                np.array(
                    [
                        result[f"se_{key}"]
                        for key in ("coarse", "fine", "rich", "difference")
                    ]
                )
                ** 2,
                errors**2,
                rtol=5 * se_var,
            )
            assert np.all(np.abs(true_means[[0, 2]] - exact) >= 5 * errors[[0, 2]])

        reduction = float((independent["se_difference"] / paired["se_difference"]) ** 2)
        target = v_diff / (v_diff - 2 * cov)
        ratio_band = 5 * se_var * np.sqrt(1 + (v_c**2 + v_f**2) / v_diff**2)
        assert reduction == pytest.approx(target, rel=ratio_band)
        for label, result in (("independent", independent), ("coupled", paired)):
            print(
                f"h={h:g} {label} marginal variance ratios="
                f"{np.array([result['var_coarse'], result['var_fine']]) / [v_c, v_f]}; "
                f"SE difference={result['se_difference']:.4g} "
                f"SE extrapolated={result['se_rich']:.4g}"
            )
        actual = np.array([paired["mean_coarse"], paired["mean_rich"]])
        actual_errors.append(np.abs(actual - exact))
        signal = abs(true_means[2] - exact) / independent["se_rich"]
        print(
            f"h={h:g} paired means={actual} analytic={true_means[[0, 2]]}; "
            f"correction variance ratio={reduction:.4g} target={target:.4g}; "
            f"independent Richardson signal/SE={signal:.3f}"
        )
        sample = np.asarray(replay_pairs(key_pairs, h, n))
        np.testing.assert_allclose(sample.mean(axis=0), actual, rtol=2e-14)
        pairs.append(sample)

    samples = np.stack(pairs)
    rng = np.random.default_rng(1386)
    slopes = []
    for _ in range(32):
        indices = rng.integers(M, size=(32, M))
        bias = np.abs(samples[:, indices].mean(axis=2) - exact)
        slopes.append(np.einsum("l,lrb->rb", weights, np.log(bias)))
    intervals = np.quantile(np.concatenate(slopes), (0.025, 0.975), axis=0).T
    for col, reference_order in enumerate((1, 2)):
        verdict = check_order(
            hs,
            np.array(actual_errors)[:, col],
            reference_order,
            tol_low=0.2,
            tol_high=0.25,
            fit_points=3,
        )
        print(
            f"analytic slope={predicted_orders[col]:.4f}, "
            f"95% paired interval={intervals[col]}, "
            f"acceptance={bands[col]}: {verdict.message}"
        )
        assert bands[col, 0] <= intervals[col, 0] <= intervals[col, 1] <= bands[col, 1]
        assert verdict.status == "PASS"


def test_exact_additive_null_and_spectrum():
    """A real cosine coefficient has variance 2σ²qF_T on the unit domain.

    Its sine partner is independent with the same variance V. The sum of
    their squares has mean m²+2V and variance 4V(m²+V). Means use 5 SE and
    Gaussian variances use 5 sqrt(2/(M−1)). The p=1 independent estimator
    adds 4V+V; the coupled estimator has variance V. The corresponding
    sample ratio has relative SE sqrt((2/(M−1))(1+17/25)).
    """
    n, nu, sigma, alpha, h, nsteps = 16, 0.05, 0.2, 1.5, 1 / 16, 8
    x = jnp.arange(n) / n
    cosine, sine = jnp.cos(2 * jnp.pi * x), jnp.sin(2 * jnp.pi * x)

    def observable(u):
        c = 2 * jnp.mean(u * cosine)
        s = 2 * jnp.mean(u * sine)
        return jnp.stack((c, c**2 + s**2))

    params = dict(lambda_=0.0, diffusivity=nu, sigma=sigma, noise_alpha=alpha)
    args = (
        StochasticAllenCahn(1, 1.0, n, h, **params),
        StochasticAllenCahn(1, 1.0, n, h / 2, **params),
        cosine[None, :],
        nsteps,
    )
    opts = dict(num_samples=M, batch_size=BATCH, observable=observable)
    independent = richardson_weak_extrapolation(
        *args, jax.random.fold_in(KEY, 1), **opts
    )
    paired = richardson_weak_extrapolation(
        *args, jax.random.fold_in(KEY, 2), coupled=True, **opts
    )
    symbol = -nu * (2 * np.pi) ** 2
    mu = np.exp(symbol * h * nsteps)
    v = (
        2
        * sigma**2
        * (1 + (2 * np.pi) ** 2) ** -alpha
        * np.expm1(2 * symbol * h * nsteps)
        / (2 * symbol)
    )
    means = np.array([mu, mu**2 + 2 * v])
    standard_error = np.sqrt([v / M, 4 * v * (mu**2 + v) / M])
    relative_se = np.sqrt(2 / (M - 1))
    for result in (independent, paired):
        for level in ("coarse", "fine"):
            assert np.all(np.abs(result[f"mean_{level}"] - means) <= 5 * standard_error)
            assert float(result[f"var_{level}"][0]) == pytest.approx(
                v, rel=5 * relative_se
            )
    assert np.max(np.abs(paired["mean_fine"] - paired["mean_coarse"])) <= 1e-14
    assert np.max(paired["se_difference"]) <= 1e-14
    ratio = float((independent["se_rich"][0] / paired["se_rich"][0]) ** 2)
    assert ratio == pytest.approx(5, rel=5 * relative_se * np.sqrt(1 + 17 / 25))
    print(
        f"independent cosine variances={independent['var_coarse'][0]:.5g}, "
        f"{independent['var_fine'][0]:.5g}; "
        f"coupled variance={paired['var_coarse'][0]:.5g}; target={v:.5g}"
    )
    print(
        f"paired mean={np.asarray(paired['mean_rich'])} target={means}; "
        f"independent/coupled extrapolated variance={ratio:.5g} target=5; "
        f"paired correction SE={np.asarray(paired['se_difference'])}; "
        "order INCONCLUSIVE: exact linear law and round-off correction"
    )


def test_multiplicative_coarsening_preserves_one_step_law():
    """For uniform u=1 and L=0, the mean kick is σ times the noise DC term.

    The Brownian sum gives Var(mean kick)=σ²h on a unit domain. The
    variance assertion uses 5 sqrt(2/(M−1)) relative Gaussian sampling SE.
    """
    h, grid, sigma = 0.125, 8, 0.2
    params = dict(
        lambda_=0.0,
        diffusivity=0.0,
        sigma=sigma,
        noise_alpha=1.5,
        noise_type="multiplicative",
    )
    coarse = StochasticAllenCahn(1, 1.0, grid, h, **params)
    fine = StochasticAllenCahn(1, 1.0, grid, h / 2, **params)
    u_hat = ex.fft(jnp.ones((1, grid)))
    keys = jax.random.split(KEY, M)

    @jax.jit
    def compare():
        def sample(key):
            a, b, c = jax.random.split(key, 3)
            first, second = fine.sample_increment(a), fine.sample_increment(b)
            summed = fine.coarsen_increments(jnp.stack((first, second)))
            estimates = [
                (coarse.step_with_increment(u_hat, draw)[0, 0].real / grid) - 1
                for draw in (summed, coarse.sample_increment(c))
            ]
            return jnp.stack(estimates), jnp.max(jnp.abs(summed - (first + second)))

        return jax.lax.map(sample, keys, batch_size=BATCH)

    kicks, sum_error = map(np.asarray, compare())
    variance = sigma**2 * h
    assert sum_error.max() == 0.0
    assert np.max(np.abs(kicks.mean(axis=0))) <= 5 * np.sqrt(variance / M)
    np.testing.assert_allclose(
        kicks.var(axis=0, ddof=1), variance, rtol=5 * np.sqrt(2 / (M - 1))
    )
    print(
        f"multiplicative coarse-sum/direct variances={kicks.var(axis=0, ddof=1)} "
        f"target={variance:g}, sum error={sum_error.max():g}"
    )
