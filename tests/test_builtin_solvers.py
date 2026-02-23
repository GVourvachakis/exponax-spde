import jax
import jax.numpy as jnp
import pytest

import exponax_spde as ex


def test_instantiate():
    domain_extent = 10.0
    num_points = 25
    dt = 0.1

    for num_spatial_dims in [1, 2, 3]:
        for simulator in [
            ex.stepper.Burgers,
            ex.stepper.HeatEquation,
            ex.stepper.generic.GeneralLinearStepper,
            ex.stepper.generic.GeneralNonlinearStepper,
            ex.stepper.generic.GeneralPolynomialStepper,
        ]:
            simulator(num_spatial_dims, domain_extent, num_points, dt)

    for num_spatial_dims in [1, 2, 3]:
        for simulator in [
            ex.stepper.reaction.AllenCahn,
        ]:
            simulator(num_spatial_dims, domain_extent, num_points, dt)

    for num_spatial_dims in [1, 2, 3]:
        for normalized_simulator in [
            ex.stepper.generic.NormalizedLinearStepper,
            ex.stepper.generic.NormalizedPolynomialStepper,
            ex.stepper.generic.NormalizedNonlinearStepper,
        ]:
            normalized_simulator(num_spatial_dims, num_points)


@pytest.mark.parametrize(
    "specific_stepper,general_stepper_coefficients",
    [
        # Linear problems
        (
            ex.stepper.HeatEquation(1, 3.0, 50, 0.1, diffusivity=0.01),
            [0.0, 0.0, 0.01],
        ),
    ],
)
def test_specific_stepper_to_general_linear_stepper(
    specific_stepper,
    general_stepper_coefficients,
):
    num_spatial_dims = specific_stepper.num_spatial_dims
    domain_extent = specific_stepper.domain_extent
    num_points = specific_stepper.num_points
    dt = specific_stepper.dt

    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    general_stepper = ex.stepper.generic.GeneralLinearStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        linear_coefficients=general_stepper_coefficients,
    )

    specific_pred = specific_stepper(u_0)
    general_pred = general_stepper(u_0)

    assert specific_pred == pytest.approx(general_pred, rel=1e-4)


# The two stale @pytest.mark.parametrize decorators referencing
# specific_stepper / general_stepper_scale / conservative that were
# previously stacked here have been removed — they were leftover from
# test_specific_stepper_to_general_linear_stepper and caused a pytest
# collection error because this function only accepts `coefficients`.
@pytest.mark.parametrize(
    "coefficients",
    [
        [0.0, 0.0, 0.01],  # heat equation
    ],
)
def test_linear_normalized_stepper(coefficients):
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 50
    dt = 0.1

    u_0 = ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims,
        cutoff=5,
    )(num_points, key=jax.random.PRNGKey(0))

    regular_linear_stepper = ex.stepper.generic.GeneralLinearStepper(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        linear_coefficients=coefficients,
    )
    normalized_linear_stepper = ex.stepper.generic.NormalizedLinearStepper(
        num_spatial_dims,
        num_points,
        normalized_linear_coefficients=ex.stepper.generic.normalize_coefficients(
            coefficients,
            domain_extent=domain_extent,
            dt=dt,
        ),
    )

    regular_linear_pred = regular_linear_stepper(u_0)
    normalized_linear_pred = normalized_linear_stepper(u_0)

    assert regular_linear_pred == pytest.approx(normalized_linear_pred, rel=1e-4)


def test_nonlinear_normalized_stepper():
    num_spatial_dims = 1
    domain_extent = 3.0
    num_points = 50
    dt = 0.1
    diffusivity = 0.1
    convection_scale = 1.0

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = jnp.sin(2 * jnp.pi * grid / domain_extent) + 0.3

    regular_burgers_stepper = ex.stepper.Burgers(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        diffusivity=diffusivity,
        convection_scale=convection_scale,
    )
    normalized_burgers_stepper = ex.stepper.generic.NormalizedConvectionStepper(
        num_spatial_dims,
        num_points,
        normalized_linear_coefficients=ex.stepper.generic.normalize_coefficients(
            [0.0, 0.0, diffusivity],
            domain_extent=domain_extent,
            dt=dt,
        ),
        normalized_convection_scale=ex.stepper.generic.normalize_convection_scale(
            convection_scale,
            domain_extent=domain_extent,
            dt=dt,
        ),
    )

    regular_burgers_pred = regular_burgers_stepper(u_0)
    normalized_burgers_pred = normalized_burgers_stepper(u_0)

    assert regular_burgers_pred == pytest.approx(
        normalized_burgers_pred, rel=1e-5, abs=1e-5
    )


# ===========================================================================
# Difficulty-based stepper tests
# ===========================================================================


class TestDifficultySteppers:
    """Test the Difficulty*Stepper constructors that convert difficulty values
    into normalized coefficients."""

    def test_difficulty_linear_stepper(self):
        stepper = ex.stepper.generic.DifficultyLinearStepper(
            num_spatial_dims=1,
            num_points=48,
            linear_difficulties=(0.0, -2.0, 0.01),
        )
        u_0 = ex.ic.RandomTruncatedFourierSeries(1, cutoff=5)(
            48, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        assert u_1.shape == u_0.shape
        assert jnp.all(jnp.isfinite(u_1))

    def test_difficulty_polynomial_stepper(self):
        stepper = ex.stepper.generic.DifficultyPolynomialStepper(
            num_spatial_dims=1,
            num_points=48,
            linear_difficulties=(0.0, 0.0, 0.1),
            polynomial_difficulties=(0.0, 0.0, -0.01),
        )
        u_0 = ex.ic.RandomTruncatedFourierSeries(1, cutoff=5)(
            48, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        assert u_1.shape == u_0.shape
        assert jnp.all(jnp.isfinite(u_1))

    def test_difficulty_nonlinear_stepper(self):
        stepper = ex.stepper.generic.DifficultyNonlinearStepper(
            num_spatial_dims=1,
            num_points=48,
            linear_difficulties=(0.0, 0.0, 0.1),
            nonlinear_difficulties=(0.01, -0.05, 0.001),
        )
        u_0 = ex.ic.RandomTruncatedFourierSeries(1, cutoff=5)(
            48, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        assert u_1.shape == u_0.shape
        assert jnp.all(jnp.isfinite(u_1))

    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_difficulty_linear_multi_dim(self, D):
        stepper = ex.stepper.generic.DifficultyLinearStepper(
            num_spatial_dims=D,
            num_points=16,
        )
        u_0 = ex.ic.RandomTruncatedFourierSeries(D, cutoff=3)(
            16, key=jax.random.PRNGKey(0)
        )
        u_1 = stepper(u_0)
        assert u_1.shape == u_0.shape
        assert jnp.all(jnp.isfinite(u_1))


# ===========================================================================
# GeneralNonlinearStepper validation
# ===========================================================================


class TestGeneralNonlinearStepperValidation:
    def test_wrong_number_of_nonlinear_coefficients(self):
        """nonlinear_coefficients must have exactly 3 elements."""
        with pytest.raises(ValueError, match="3"):
            ex.stepper.generic.GeneralNonlinearStepper(
                1,
                1.0,
                32,
                0.01,
                nonlinear_coefficients=(0.0, -1.0),  # only 2
            )
        with pytest.raises(ValueError, match="3"):
            ex.stepper.generic.GeneralNonlinearStepper(
                1,
                1.0,
                32,
                0.01,
                nonlinear_coefficients=(0.0, -1.0, 0.5, 0.1),  # 4
            )


# ===========================================================================
# ETDRK order equivalence for linear problems
# ===========================================================================


class TestETDRKOrderConvergence:
    """Higher ETDRK orders should produce consistent results for smooth problems."""

    def test_orders_agree_on_burgers(self):
        """ETDRK orders 1-4 should agree on a smooth Burgers problem."""
        L, N, dt = 2 * jnp.pi, 64, 0.01
        u_0 = ex.ic.RandomTruncatedFourierSeries(1, cutoff=5)(
            N, key=jax.random.PRNGKey(0)
        )
        results = {}
        for order in [1, 2, 3, 4]:
            stepper = ex.stepper.Burgers(
                1,
                L,
                N,
                dt,
                diffusivity=0.1,
                convection_scale=1.0,
                order=order,
            )
            results[order] = stepper(u_0)

        # Orders 2-4 should closely agree (order 1 may differ slightly more)
        for order in [3, 4]:
            assert results[order] == pytest.approx(results[2], abs=5e-4)


# ===========================================================================
# BaseStepper input validation
# ===========================================================================


class TestBaseStepperValidation:
    def test_wrong_input_shape_raises(self):
        """Calling a stepper with wrong input shape should raise ValueError."""
        stepper = ex.stepper.HeatEquation(1, 1.0, 32, 0.01)
        wrong_shape = jnp.zeros((1, 64))  # Expected (1, 32)
        with pytest.raises(ValueError, match="Expected shape"):
            stepper(wrong_shape)

    def test_wrong_channels_raises(self):
        """Calling a stepper with wrong number of channels should raise."""
        stepper = ex.stepper.HeatEquation(1, 1.0, 32, 0.01)
        wrong_channels = jnp.zeros((2, 32))  # HeatEquation expects 1 channel
        with pytest.raises(ValueError, match="Expected shape"):
            stepper(wrong_channels)


# ===========================================================================
# Analytical correctness tests
# ===========================================================================


class TestDiffusionAnalytical:
    def test_mode_decay_rate(self):
        """Each Fourier mode should decay as exp(-ν*(2πk/L)²*dt)."""
        L, N, nu, dt = 2 * jnp.pi, 64, 0.1, 0.05
        k = 3
        stepper = ex.stepper.HeatEquation(1, L, N, dt, diffusivity=nu)
        grid = ex.make_grid(1, L, N)
        u_0 = jnp.sin(k * 2 * jnp.pi * grid / L)
        u_1 = stepper(u_0)
        decay_factor = jnp.exp(-nu * (k * 2 * jnp.pi / L) ** 2 * dt)
        expected = decay_factor * u_0
        assert u_1 == pytest.approx(expected, abs=1e-5)

    def test_higher_modes_decay_faster(self):
        """Higher wavenumbers should decay faster under diffusion."""
        L, N, nu, dt = 2 * jnp.pi, 64, 0.05, 0.1
        stepper = ex.stepper.HeatEquation(1, L, N, dt, diffusivity=nu)
        grid = ex.make_grid(1, L, N)

        u_low = jnp.sin(1 * 2 * jnp.pi * grid / L)
        u_high = jnp.sin(5 * 2 * jnp.pi * grid / L)

        u_low_1 = stepper(u_low)
        u_high_1 = stepper(u_high)

        # Ratio of amplitudes after diffusion
        ratio_low = float(jnp.max(jnp.abs(u_low_1))) / float(jnp.max(jnp.abs(u_low)))
        ratio_high = float(jnp.max(jnp.abs(u_high_1))) / float(jnp.max(jnp.abs(u_high)))
        assert ratio_high < ratio_low  # Higher mode decays more

    def test_energy_monotone_decrease(self):
        """Diffusion should monotonically decrease energy."""
        L, N, nu, dt = 3.0, 64, 0.01, 0.01
        stepper = ex.stepper.HeatEquation(1, L, N, dt, diffusivity=nu)
        u = ex.ic.RandomTruncatedFourierSeries(1, cutoff=10)(
            N, key=jax.random.PRNGKey(0)
        )
        prev_energy = float(jnp.sum(u**2))
        for _ in range(50):
            u = stepper(u)
            energy = float(jnp.sum(u**2))
            assert energy <= prev_energy + 1e-6
            prev_energy = energy
