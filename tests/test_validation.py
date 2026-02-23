import jax.numpy as jnp
import pytest

import exponax_spde as ex

# Linear steppers

# linear steppers do not make spatial and temporal truncation errors, hence we
# can directly compare them with the analytical solution without performing a
# convergence study


def test_heatequation_1d():
    num_spatial_dims = 1
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    diffusivity = 0.1

    def analytical_solution(t, x):
        return jnp.exp(
            -((4 * 2 * jnp.pi / domain_extent) ** 2) * diffusivity * t
        ) * jnp.sin(4 * 2 * jnp.pi * x / domain_extent)

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)
    u_0 = analytical_solution(0.0, grid)
    u_1 = analytical_solution(dt, grid)

    stepper = ex.stepper.HeatEquation(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        diffusivity=diffusivity,
    )

    u_1_pred = stepper(u_0)

    assert u_1_pred == pytest.approx(u_1, abs=1e-5)


def test_heatequation_2d():
    num_spatial_dims = 2
    domain_extent = 10.0
    num_points = 100
    dt = 0.1
    diffusivity = 0.1

    def analytical_solution(t, x):
        # Third sine mode in x-direction and fourth sine mode in y-direction
        third_sine_mode_x = jnp.sin(3 * 2 * jnp.pi * x[0:1] / domain_extent)
        fourth_sine_mode_y = jnp.sin(4 * 2 * jnp.pi * x[1:2] / domain_extent)
        exponent = (
            -diffusivity
            * (
                (3 * 2 * jnp.pi / domain_extent) ** 2
                + (4 * 2 * jnp.pi / domain_extent) ** 2
            )
            * t
        )
        exp_term = jnp.exp(exponent)
        return exp_term * third_sine_mode_x * fourth_sine_mode_y

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)

    u_0 = analytical_solution(0.0, grid)
    u_1 = analytical_solution(dt, grid)

    stepper = ex.stepper.HeatEquation(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        diffusivity=diffusivity,
    )

    u_1_pred = stepper(u_0)

    assert u_1_pred == pytest.approx(u_1, abs=1e-5)


def test_heatequation_3d():
    num_spatial_dims = 3
    domain_extent = 10.0
    num_points = 40
    dt = 0.1
    diffusivity = 0.1

    def analytical_solution(t, x):
        # Third sine mode in x-direction, fourth sine mode in y-direction, and
        # fifth sine mode in z-direction
        third_sine_mode_x = jnp.sin(3 * 2 * jnp.pi * x[0:1] / domain_extent)
        fourth_sine_mode_y = jnp.sin(4 * 2 * jnp.pi * x[1:2] / domain_extent)
        fifth_sine_mode_z = jnp.sin(5 * 2 * jnp.pi * x[2:3] / domain_extent)
        exponent = (
            -diffusivity
            * (
                (3 * 2 * jnp.pi / domain_extent) ** 2
                + (4 * 2 * jnp.pi / domain_extent) ** 2
                + (5 * 2 * jnp.pi / domain_extent) ** 2
            )
            * t
        )
        exp_term = jnp.exp(exponent)
        return exp_term * third_sine_mode_x * fourth_sine_mode_y * fifth_sine_mode_z

    grid = ex.make_grid(num_spatial_dims, domain_extent, num_points)

    u_0 = analytical_solution(0.0, grid)
    u_1 = analytical_solution(dt, grid)

    stepper = ex.stepper.HeatEquation(
        num_spatial_dims,
        domain_extent,
        num_points,
        dt,
        diffusivity=diffusivity,
    )

    u_1_pred = stepper(u_0)

    assert u_1_pred == pytest.approx(u_1, abs=1e-5)
