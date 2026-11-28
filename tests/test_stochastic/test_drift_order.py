"""Independent references and paired fixed-grid drift-order studies."""

import sys
import warnings

import exponax as ex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from physverify import check_order

from exponax_spde import (
    TamingOrderWarning,
    richardson_weak_extrapolation,
    stochastic_ensemble_rollout,
)
from exponax_spde.stepper.stochastic import StochasticAllenCahn

ORDERS = (1, 2, 3, 4)
T = 0.5
U0 = 0.65
N = 8
PATHS = 1536
BATCH = 64
DIFFUSIVITY = 0.05


def stochastic(order, dt, n=N, *, use_taming=False, **options):
    return StochasticAllenCahn(
        1,
        1.0,
        n,
        dt,
        order=order,
        diffusivity=DIFFUSIVITY,
        lambda_=1.0,
        dealiasing_fraction=2 / 3,
        use_taming=use_taming,
        **options,
    )


def deterministic(order, dt, n=N):
    return ex.stepper.reaction.AllenCahn(
        1,
        1.0,
        n,
        dt,
        order=order,
        diffusivity=DIFFUSIVITY,
        first_order_coefficient=1,
        third_order_coefficient=-1,
        dealiasing_fraction=2 / 3,
    )


@pytest.mark.parametrize("order", ORDERS)
def test_untamed_drift_matches_upstream_bit_for_bit(order):
    """Zero noise leaves precisely the upstream map with matched drift."""
    n, h = 16, 0.005
    angle = 2 * jnp.pi * jnp.arange(n) / n
    field = (U0 + 0.12 * jnp.cos(angle) - 0.05 * jnp.sin(2 * angle))[None]
    solver = stochastic(order, h, n, sigma=0.0)
    reference = deterministic(order, h, n)
    key = jax.random.key(4035)

    def trajectory(step):
        def advance(u, _):
            u = step(u)
            return u, u

        return jax.jit(lambda u: jax.lax.scan(advance, u, None, length=20)[1])(field)

    actual = np.asarray(trajectory(lambda u: solver(u, key=key)))
    expected = np.asarray(trajectory(reference))
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual, expected)
    assert actual.tobytes() == expected.tobytes()
    print(f"order={order} single step and 20-step rollout bit-identical")


@pytest.mark.parametrize("order", ORDERS)
def test_deterministic_etdrk_orders(order):
    """The constant mode is u'=u−u³, so u(T)=[1+(u0⁻²−1)e⁻²ᵀ]⁻¹ᐟ²."""
    field = jnp.full((1, N), U0)
    counts = (4, 8, 16, 32)

    def endpoint(step, count):
        return np.asarray(
            jax.jit(lambda x: jax.lax.fori_loop(0, count, lambda _, v: step(v), x))(
                field
            )
        )

    fine = endpoint(deterministic(4, T / 1024), 1024)
    analytical = (1 + (U0**-2 - 1) * np.exp(-2 * T)) ** -0.5
    np.testing.assert_allclose(fine, analytical, rtol=0, atol=2e-13)
    key = jax.random.key(4035)
    errors = []
    for count in counts:
        solver = stochastic(order, T / count, sigma=0.0)
        result = endpoint(lambda u, step=solver: step(u, key=key), count)
        errors.append(np.sqrt(np.mean((result - fine) ** 2)))
    verdict = check_order(
        T / np.asarray(counts),
        np.asarray(errors),
        order,
        tol_low=0.2,
        tol_high=0.2,
        floor=1e-14,
        scale=float(np.max(np.abs(fine))),
        fit_points=4,
    )
    print(
        f"order={order} errors={np.asarray(errors)} {verdict.status}: {verdict.message}"
    )
    assert verdict.status == "PASS", verdict.message


def coupled_scores(order, counts, seed, *, paths=PATHS, **noise):
    """Final scores only, with all level endpoints on one common path."""
    steppers = tuple(stochastic(order, T / n, noise_alpha=12, **noise) for n in counts)
    finest = steppers[-1]
    block_size = counts[-1] // counts[0]
    original = ex.fft(jnp.full((1, N), U0))
    root = jax.random.key(seed)

    @jax.jit
    def compute(path_indices):
        keys = jax.vmap(lambda i: jax.random.fold_in(root, i))(path_indices)
        start = jnp.broadcast_to(original, (BATCH, *original.shape))

        def block(states, index):
            times = block_size * index + jnp.arange(block_size)
            increments = jax.vmap(
                lambda t: jax.vmap(
                    lambda key: finest.sample_increment(jax.random.fold_in(key, t))
                )(keys)
            )(times)
            level_inputs = [increments]
            for fine in reversed(steppers[1:]):
                pairs = increments.reshape((-1, 2, BATCH, *original.shape))
                coarsen = jax.vmap(fine.coarsen_increments, in_axes=1)
                increments = jax.vmap(coarsen)(pairs)
                level_inputs.append(increments)

            def advance(stepper, initial, draws):
                step = jax.vmap(stepper.step_with_increment)
                return jax.lax.scan(lambda u, d: (step(u, d), None), initial, draws)[0]

            return tuple(
                advance(solver, u, draws)
                for solver, u, draws in zip(
                    steppers, states, reversed(level_inputs), strict=True
                )
            ), None

        states, _ = jax.lax.scan(block, (start,) * len(counts), jnp.arange(counts[0]))
        differences = jnp.stack(states[:-1]) - jnp.stack(states[1:])
        physical = ex.ifft(differences, num_spatial_dims=1, num_points=N)
        return jnp.mean(physical**2, axis=(-1, -2)), differences[..., 0, 0].real / N

    batches = [
        tuple(map(np.asarray, compute(jnp.arange(i, i + BATCH))))
        for i in range(0, paths, BATCH)
    ]
    return tuple(
        np.concatenate(values, axis=-1) for values in zip(*batches, strict=True)
    )


def bootstrap_means(squares, seed):
    rng = np.random.default_rng(seed)
    paths = squares.shape[-1]
    return np.concatenate(
        [
            squares[..., rng.integers(paths, size=(32, paths))].mean(axis=-1)
            for _ in range(32)
        ],
        axis=-1,
    )


def rms_relative_se(squares):
    paths = squares.shape[-1]
    return squares.std(axis=-1, ddof=1) / (2 * np.sqrt(paths) * squares.mean(axis=-1))


# Frozen scalar forecasts for T=0.5, steps 4,8,16,32,64. Generated before
# sampling by the scalar Cox–Matthews map and exact tangent covariance.
LEVEL_BIAS = np.array(
    [
        [
            7.82898495e-03,
            3.69337148e-03,
            1.79506319e-03,
            8.85053506e-04,
            4.39458853e-04,
        ],
        [
            1.40080077e-03,
            3.15422913e-04,
            7.49017568e-05,
            1.82540306e-05,
            4.50595712e-06,
        ],
        [
            7.50558913e-05,
            8.95991341e-06,
            1.09114042e-06,
            1.34533306e-07,
            1.66990101e-08,
        ],
        [
            4.06325212e-06,
            2.34540500e-07,
            1.40880835e-08,
            8.63204730e-10,
            5.34060574e-11,
        ],
    ]
)


TANGENT_VARIANCE = np.array(
    [
        [
            2.302145278e-03,
            5.045539643e-04,
            1.182562912e-04,
            2.863562353e-05,
            7.046241578e-06,
        ],
        [
            2.417185784e-03,
            5.427318294e-04,
            1.284040921e-04,
            3.121775435e-05,
            7.695711754e-06,
        ],
        [
            2.423213193e-03,
            5.427425953e-04,
            1.283685001e-04,
            3.121131580e-05,
            7.694792570e-06,
        ],
        [
            2.419751601e-03,
            5.425520981e-04,
            1.283574903e-04,
            3.121065612e-05,
            7.694752229e-06,
        ],
    ]
)


@pytest.mark.slow
def test_small_additive_noise_crossover():
    """Predict the level differences by a deterministic bias and normal tangent.

    With drift derivatives Dc′, Df′, the centred pair's covariance obeys
    Σ←GΣGᵀ+(expm1(h)/2)BBᵀ, where G=diag(Dc′, Df′(mid)Df′(start)),
    B=[[exp(h/2), 1], [Df′(mid), 1]]. This gives unit-noise difference
    variance v. For a Normal(μ,σ²v) difference the mean-square SE equals
    sqrt((2σ⁴v² + 4μ²σ²v)/M). Test 5 SE and paired slope intervals.
    """
    counts = (4, 8, 16, 32, 64, 128)
    sigma = 1e-6
    squares, differences = (
        np.stack(values)
        for values in zip(
            *(coupled_scores(k, counts, 4036, sigma=sigma) for k in ORDERS), strict=True
        )
    )
    # Forecast standard deviations from the centred scalar covariance.
    variance = sigma**2 * TANGENT_VARIANCE
    expected = LEVEL_BIAS**2 + variance
    expected_se = np.sqrt((2 * variance**2 + 4 * LEVEL_BIAS**2 * variance) / PATHS)
    mean = squares.mean(axis=-1)
    measured_rms, predicted_rms = np.sqrt(mean), np.sqrt(expected)

    paired = bootstrap_means(squares, 4037)
    slopes = 0.5 * np.log2(paired[:, :-1] / paired[:, 1:])
    interval = np.quantile(slopes, (0.025, 0.975), axis=-1)
    predicted_slopes = np.log2(predicted_rms[:, :-1] / predicted_rms[:, 1:])
    ranking = np.quantile(np.sqrt(paired[:-1] / paired[1:]), 0.025, axis=-1)
    h_min = T / counts[-2]
    c2 = differences[0, -1].std(ddof=1) / (sigma * h_min)

    print(f"small σ measured RMS={measured_rms.tolist()}")
    print(f"small σ predicted RMS={predicted_rms.tolist()}")
    print(f"small σ SE deviations={((mean - expected) / expected_se).tolist()}")
    adjacent_slopes = np.log2(measured_rms[:, :-1] / measured_rms[:, 1:])
    print(f"small σ observed adjacent slopes={adjacent_slopes.tolist()}")
    print(
        f"small σ paired 95% slopes={interval.tolist()}, "
        f"ranking lower={ranking.tolist()}, C2={c2:.8g}"
    )

    assert np.max(rms_relative_se(squares)) < 0.03
    assert np.all(abs(mean - expected) <= 5 * expected_se)
    assert np.all(interval[0] >= predicted_slopes - 0.2)
    assert np.all(interval[1] <= predicted_slopes + 0.2)
    assert np.all(ranking > 1)
    assert c2 == pytest.approx(0.33977289771147356, rel=5 / np.sqrt(2 * (PATHS - 1)))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("noise_type", "milstein", "orders", "expected", "band"),
    [
        ("additive", False, ORDERS, 1.0, (0.8, 1.2)),
        ("multiplicative", False, (2, 4), 0.5, (0.35, 0.65)),
        ("multiplicative", True, (2, 4), 1.0, (0.8, 1.2)),
    ],
    ids=("additive", "multiplicative-em", "multiplicative-milstein"),
)
def test_moderate_noise_strong_orders(noise_type, milstein, orders, expected, band):
    """The shared-path level difference estimates the leading strong error.

    At fixed resolution the additive convolution omits the centred
    within-step nonlinear response O(σh**1.5), giving global order one.
    Euler–Maruyama multiplicative noise omits the Milstein term of local
    RMS O(h). Including it leaves O(h**1.5) terms for commutative noise.
    Predict slopes 1, 1/2, 1 respectively and use paired-bootstrap 95%
    intervals, with empirical RMS delta-method relative SE below 3%.
    """
    # The Milstein pilot's squared-error CV of 5.218 requires at least
    # ceil((5.218 / (2 * 0.03)) ** 2) = 7564 paths for the RMS SE gate.
    paths = 8192 if milstein else PATHS
    counts = (16, 32, 64, 128, 256)
    hs = T / np.asarray(counts[:-1])
    squares = np.stack(
        [
            coupled_scores(
                k,
                counts,
                4038,
                paths=paths,
                sigma=0.35,
                noise_type=noise_type,
                use_milstein=milstein,
            )[0]
            for k in orders
        ]
    )
    rms = np.sqrt(squares.mean(axis=-1))
    weights = np.log(hs) - np.log(hs).mean()
    weights /= weights @ weights
    estimates = np.einsum(
        "h,khb->kb", weights, 0.5 * np.log(bootstrap_means(squares, 4039))
    )
    intervals = np.quantile(estimates, (0.025, 0.975), axis=-1)
    relative_se = rms_relative_se(squares)
    maximum_rse = np.max(relative_se)
    results = [
        check_order(
            hs,
            errors,
            expected,
            tol_low=expected - band[0],
            tol_high=band[1] - expected,
            fit_points=4,
        )
        for errors in rms
    ]
    for idx, (k, result) in enumerate(zip(orders, results, strict=True)):
        print(
            f"moderate {noise_type} Milstein={milstein} order={k} "
            f"RMS={rms[idx].tolist()}, "
            f"paired 95%={intervals[:, idx].tolist()}, "
            f"RMS RSE={relative_se[idx].tolist()}, paths={paths}, "
            f"{result.status}: {result.message}"
        )

    assert maximum_rse < 0.03
    assert np.all(intervals[0] >= band[0]) and np.all(intervals[1] <= band[1])
    assert all(result.status == "PASS" for result in results)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_drift_order_preserves_increment_and_batch_contract(order):
    """Order changes the drift and keeps the increment law and key mapping."""
    key = jax.random.key(4040)
    h, members = 0.015625, 7
    u0 = jnp.full((1, N), U0)
    uhat = ex.fft(u0)
    base = stochastic(1, h / 2)
    fine = stochastic(order, h / 2)
    coarse = stochastic(order, h)
    keys = jax.random.split(key, 2)
    draws = jax.vmap(fine.sample_increment)(keys)
    np.testing.assert_array_equal(draws, jax.vmap(base.sample_increment)(keys))
    np.testing.assert_array_equal(
        fine.coarsen_increments(draws), base.coarsen_increments(draws)
    )
    np.testing.assert_array_equal(
        fine.step_fourier(uhat, key=keys[0]), fine.step_with_increment(uhat, draws[0])
    )
    expected = richardson_weak_extrapolation(
        coarse, fine, u0, 2, key, num_samples=members, coupled=True
    )
    rollout = stochastic_ensemble_rollout(coarse, 3, members)
    trajectories = rollout(u0, key)
    for width in (1, 3, members):
        result = richardson_weak_extrapolation(
            coarse,
            fine,
            u0,
            2,
            key,
            num_samples=members,
            coupled=True,
            batch_size=width,
        )
        for name in expected:
            np.testing.assert_allclose(
                result[name], expected[name], rtol=1e-12, atol=1e-15
            )
        np.testing.assert_allclose(
            stochastic_ensemble_rollout(coarse, 3, members, batch_size=width)(u0, key),
            trajectories,
            rtol=1e-12,
            atol=1e-15,
        )


@pytest.mark.filterwarnings("ignore::exponax_spde.TamingOrderWarning")
@pytest.mark.parametrize("order", ORDERS)
def test_tamed_drift_is_first_order_relative_to_untamed(order):
    """Taming -u³ replaces it with -u³/(1+h|u³|), adding hu⁶+O(h²).

    For the positive, constant field that O(h) drift perturbation persists
    over a fixed time, even with higher-order quadrature. Compare with the
    exact untamed constant-mode solution of u'=u−u³.
    """
    field = jnp.full((1, N), U0)
    counts = np.array([4, 8, 16, 32])
    analytic = (1 + (U0**-2 - 1) * np.exp(-2 * T)) ** -0.5
    key = jax.random.key(4035)

    def error(count):
        solver = stochastic(order, T / count, sigma=0.0, use_taming=True)

        @jax.jit
        def endpoint(initial):
            return jax.lax.fori_loop(0, count, lambda _, u: solver(u, key=key), initial)

        return np.sqrt(np.mean((np.asarray(endpoint(field)) - analytic) ** 2))

    errors = np.array([error(int(steps)) for steps in counts])
    verdict = check_order(
        T / counts,
        errors,
        expected=1,
        tol_low=0.2,
        tol_high=0.2,
        floor=1e-14,
        scale=analytic,
        fit_points=4,
    )
    print(f"tamed order={order} errors={errors} {verdict.status}: {verdict.message}")
    assert 0.8 <= verdict.fitted <= 1.2
    assert verdict.status == "PASS", verdict.message


def test_higher_order_taming_warning_points_to_caller():
    with pytest.warns(
        TamingOrderWarning,
        match=r"Taming changes the drift by O\(h\).*first order.*untamed",
    ) as records:
        line = sys._getframe().f_lineno + 1
        StochasticAllenCahn(1, 1.0, N, 0.01, order=2, use_taming=True)
    (record,) = records
    assert record.category is TamingOrderWarning
    assert record.filename == __file__
    assert record.lineno == line


def test_higher_order_taming_warning_selective_filter():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("ignore", TamingOrderWarning)
        stochastic(2, 0.01, use_taming=True)
        warnings.warn("unrelated warning", UserWarning, stacklevel=1)
    (record,) = records
    assert record.category is UserWarning


@pytest.mark.parametrize(("order", "taming"), [(1, True), (4, False)])
def test_higher_order_taming_warning_exemptions(order, taming):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        stochastic(order, 0.01, use_taming=taming)


def test_higher_order_taming_warning_skips_traced_noise_parameters():
    field = jnp.full((1, N), U0)

    @jax.jit
    def evolve(sigma, diffusivity, noise_alpha):
        solver = StochasticAllenCahn(
            1,
            1.0,
            N,
            0.01,
            order=2,
            use_taming=True,
            sigma=sigma,
            diffusivity=diffusivity,
            noise_alpha=noise_alpha,
        )
        return solver(field, key=jax.random.key(4035))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        actual = evolve(0.1, DIFFUSIVITY, 1.0)
    assert np.isfinite(actual).all()
