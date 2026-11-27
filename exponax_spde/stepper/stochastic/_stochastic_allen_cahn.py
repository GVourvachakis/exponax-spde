"""
Stochastic Allen-Cahn SPDE stepper.

Equation (Itô):  ∂ₜu = νΔu + λ(u - u³) + σ(u) ξ(x, t)

Method: Exponential Euler-Maruyama (EEM)
─────────────────────────────────────────
The PDE is split, following the built-in ``exponax.stepper.reaction.AllenCahn``
stepper convention, as

    L(u) = ν Δu + λu     ← handled analytically by ETD
    𝒩(u) = -λ u³         ← evaluated explicitly

so that L + 𝒩 = ν Δu + λu - λu³ = ν Δu + λ(u - u³).

The discrete update is

    û_k(t+Δt) = E_k û_k(t)
                + φ₁(L_k Δt) Δt · 𝒩̂_k(u(t))   [ETD1 step]
                + stochastic_increment_k           [EEM noise]

where E_k = exp(L_k Δt) and φ₁(z) = (eᶻ - 1)/z.

Additive stochastic convolution
────────────────────────────────
Let q_k = (1 + |k|²_phys)^(-α) and F_h(L_k) = h φ₁(2hL_k), for h = Δt.
For a real, independent unit Gaussian field ξ in physical space, draw

    Ô_k = σ sqrt(q_k F_h(L_k) / dx^d) · FFT(ξ)_k.

The unnormalised FFT has E|FFT(ξ)_k|² = N^d. The real draw supplies
the Hermitian symmetry on DC and Nyquist planes. F_h uses the stored
real symbol L_k, including when exp(hL_k) underflows. See
docs/noise_model.md for the variance units and migration.

Multiplicative noise
────────────────────
Draw a plain Wiener increment ΔŴ_k = sqrt(q_k Δt / dx^d) · FFT(ξ)_k.
Multiply by σu at the step-start grid points and propagate the FFT of
the kick by E_k. The optional Milstein kick is
½σ²u(ΔW² − v_W), with scalar v_W = Δt mean_full(q) / dx^d, and uses the
same propagator. The stochastic product uses collocation.
"""

from __future__ import annotations

import warnings
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from exponax import fft, ifft, spectral
from jaxtyping import Array, Complex, Float, PRNGKeyArray

from ..._base_stepper import BaseStepper
from ...nonlin_fun import TamedPolynomialNonlinearFun


class NoiseRegularityWarning(UserWarning):
    """The finite-grid noise is too rough for the classical continuum model."""


def phi1(z: Array) -> Array:
    """Evaluate (exp(z) - 1) / z with the analytic limit and gradient at zero."""
    z = jnp.asarray(z, dtype=jnp.result_type(z, 1.0))
    # The degree-four series is within round-off on this interval.
    small = jnp.abs(z) < jnp.finfo(z.dtype).eps ** 0.25
    safe = jnp.where(small, 1, z)
    t = jnp.where(small, z, 0)
    series = 1 + t * (1 / 2 + t * (1 / 6 + t * (1 / 24 + t / 120)))
    return jnp.where(small, series, jnp.expm1(safe) / safe)


def _spatial_shape(num_spatial_dims: int, num_points: int) -> tuple[int, ...]:
    return (num_points,) * num_spatial_dims


class StochasticAllenCahn(BaseStepper):
    """
    Timestepper for the d-dimensional (``d ∈ {1, 2, 3}``) stochastic
    Allen-Cahn SPDE on periodic boundary conditions, solved via the
    Exponential Euler-Maruyama (EEM) method.

    In Itô form the equation reads

    ```
        ∂ₜu = ν Δu + λ(u - u³) + σ(u) ξ(x, t)
    ```

    with ``ν > 0`` the interface width, ``λ ≥ 0`` the reaction rate,
    ``σ(u)`` the noise coefficient (see ``noise_type``), and ``ξ(x, t)``
    a Q-Wiener process with spectral covariance

    ```
        Q_k ∝ (1 + |k|²_phys)^{-α}
    ```

    The deterministic limit (``sigma=0``) recovers the classical Allen-Cahn
    reaction-diffusion equation, producing sharp diffuse interfaces whose
    width scales as ``√ν``.  The expected long-time behaviour is the
    formation and coarsening of phase domains separated by smooth interfaces
    of thickness ``O(ν)``.

    Note that the Allen-Cahn equation is often solved with Dirichlet boundary
    conditions; here we use periodic boundary conditions throughout.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions ``d``.
    - `domain_extent`: The size of the domain ``L``; the domain is assumed
        to be a scaled hypercube ``Ω = (0, L)ᵈ`` with periodic boundaries.
    - `num_points`: The number of points ``N`` used to discretize the
        domain. This **includes** the left boundary point and **excludes**
        the right boundary point. In higher dimensions the number of points
        in each dimension is the same. Hence, the total number of degrees of
        freedom is ``Nᵈ``.
    - `dt`: The timestep size ``Δt`` between two consecutive states.
    - `diffusivity`: The diffusion coefficient ``ν ≥ 0``. Smaller positive values
        produce thinner interfaces. With ``ν = 0`` the model is a grid-level
        SDE system with no continuum claim. Defaults to ``0.01``.
    - `lambda_`: The reaction rate ``λ ≥ 0``.  Setting ``lambda_=0``
        reduces the equation to the stochastic heat equation, whose
        nonzero modes have Gaussian invariant covariance ``C_k = Q_k/(2ν|k|²)``
        and can be validated analytically.  Defaults to ``1.0``.
    - `noise_type`: Either ``"additive"`` (``σ(u) = σ``) or
        ``"multiplicative"`` (``σ(u) = σu``, linear / gradient-type).
        Defaults to ``"additive"``.
    - `sigma`: The physical Q-Wiener amplitude ``σ ≥ 0``.
        Setting ``sigma=0`` recovers the deterministic Allen-Cahn solver; the output is
        then identical regardless of the PRNG key. Defaults to ``0.1``.
        Negative values of ``sigma``, ``diffusivity`` or ``noise_alpha`` raise
        ``ValueError``.
    - `noise_alpha`: Spectral colour index ``α ≥ 0`` for the Q-Wiener
        covariance ``Q_k ∝ (1 + |k|²_phys)^{-α}``.  ``α = 0`` gives
        space-time white noise; ``α > d/2`` gives a trace-class
        noise. With ``σ > 0``, ``ν > 0`` and ``d ≥ 2``, construction warns
        with ``exponax_spde.NoiseRegularityWarning`` for additive and
        multiplicative Euler–Maruyama noise when ``α ≤ d/2 - 1``.
        See ``docs/noise_model.md`` for the continuum assumptions and how to
        silence the warning for intentional finite-grid use. Defaults to ``1.0``.
    - `use_taming`: Whether to apply Hutzenthaler-Jentzen taming to the
        cubic nonlinear term

        ```
            𝒩_tamed(u) = 𝒩(u) / (1 + Δt |𝒩(u)|)
        ```

        Recommended for large ``lambda_`` or large ``sigma``.  Set to
        ``False`` only when comparing against an untamed deterministic
        reference (``exponax.stepper.reaction.AllenCahn``).  Defaults
        to ``True``.
    - `use_milstein`: Whether to add the first-order Milstein correction
        for multiplicative noise. Enabling it requires trace-class noise
        (``noise_alpha > num_spatial_dims / 2``); the term is applied only
        to multiplicative updates. Defaults to ``False``.
    - `order`: Must be ``1`` (exponential Euler–Maruyama, the default).
        Other values raise ``ValueError``. Higher-order stochastic
        exponential integrators are tracked in issue #14.
    - `dealiasing_fraction`: The fraction of the highest wavenumbers to
        dealias in the nonlinear evaluation.  Default ``2/3`` (Orszag's
        rule); ``1/2`` is tighter for the cubic nonlinearity.

    **Notes on calling convention:**

    ```python
    u_next = stepper(u, key=jax.random.PRNGKey(0))
    ```

    The ``BaseStepper.step()`` interface (no key) is deliberately
    disabled and raises ``NotImplementedError``.  Use the ``__call__``
    interface or ``step_fourier(u_hat, key=key)`` directly.

    For batched operation use ``jax.vmap``::

    ```python
    keys = jax.random.split(jax.random.PRNGKey(0), M)
    u_batch = jax.vmap(lambda k: stepper(u0, key=k))(keys)
    ```

    **Increments (experimental)**

    ``sample_increment(key)`` gives an additive stochastic convolution or
    a σ-free multiplicative Wiener increment in raw rFFT units. Replay it
    with ``step_with_increment(u_hat, increment)``. On the fine stepper,
    ``coarsen_increments(increments)`` combines consecutive draws along the
    leading axis for one longer step.

    Compile methods with ``equinox.filter_jit`` or JIT a closure over the
    stepper. Physverify runs only in tests, after the JAX solve.

    **References:**

    - Allen, S. M., & Cahn, J. W. (1979). A microscopic theory for
        antiphase boundary motion and its application to antiphase domain
        coarsening. *Acta Metallurgica*, 27(6), 1085-1095.
        https://doi.org/10.1016/0001-6160(79)90196-2

    - Lord, G. J., Powell, C. E., & Shardlow, T. (2014).
        *An Introduction to Computational Stochastic PDEs*.
        Cambridge University Press.
        https://doi.org/10.1017/CBO9781139017329
        (EEM method, exact variance formula, and Q-Wiener
        discretisation: Chapters 7-10.)

    - Jentzen, A., & Kloeden, P. E. (2009a). Overcoming the order
        barrier in the numerical approximation of stochastic partial
        differential equations with additive space-time noise.
        *Proceedings of the Royal Society A*, 465(2102), 649-667.
        https://doi.org/10.1098/rspa.2008.0325
        (Exponential integration with additive SPDE noise.)

    - Jentzen, A., & Kloeden, P. E. (2009b). The numerical approximation
        of stochastic partial differential equations.
        *Milan Journal of Mathematics*, 77(1), 205-244.
        https://doi.org/10.1007/s00032-009-0100-0
        (Review of convergence theory for numerical SPDEs.)

    - Hutzenthaler, M., Jentzen, A., & Kloeden, P. E. (2011). Strong and
        weak divergence in finite time of Euler's method for stochastic
        differential equations with non-globally Lipschitz continuous
        coefficients. *Proceedings of the Royal Society A*, 467(2130),
        1563-1576. https://doi.org/10.1098/rspa.2010.0348
        (Divergence of untamed Euler for super-linearly growing
        coefficients; motivation for taming.)

    - Hutzenthaler, M., & Jentzen, A. (2015). Numerical approximations
        of stochastic differential equations with non-globally Lipschitz
        continuous coefficients. *Memoirs of the American Mathematical
        Society*, 236(1112). https://doi.org/10.1090/memo/1112
        (Tamed Euler-Maruyama: almost-sure boundedness and convergence
        rates for super-linearly growing drift/diffusion coefficients.)

    - Cox, S. M., & Matthews, P. C. (2002). Exponential time differencing
        for stiff systems. *Journal of Computational Physics*, 176(2),
        430-455. https://doi.org/10.1006/jcph.2002.6995
        (ETD1 and ETDRK4 methods.)

    - Kassam, A.-K., & Trefethen, L. N. (2005). Fourth-order time-stepping
        for stiff PDEs. *SIAM Journal on Scientific Computing*, 26(4),
        1214-1233. https://doi.org/10.1137/S1064827502410633
        (Contour-integral method for stable ETD coefficient evaluation.)

    - Funaki, T. (1995). The scaling limit for a stochastic PDE and the
        separation of phases. *Probability Theory and Related Fields*,
        102(2), 221-288. https://doi.org/10.1007/BF01213390
        (Invariant measure and sharp-interface limit of stochastic
        Allen-Cahn; theoretical basis for the λ=0 validation tests.)
    """

    # ── Scalar hyperparameters ─────────────────────────────────────────
    diffusivity: float
    lambda_: float
    sigma: float
    noise_alpha: float
    use_taming: bool
    _dealiasing_fraction_stoch: float

    # Choose the noise scheme before tracing.
    noise_type: str = eqx.field(static=True)
    use_milstein: bool = eqx.field(static=True)

    _linear_symbol: Array
    _propagator: Array
    _increment_amplitude: Array
    _wiener_variance: Array | None

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dt: float,
        *,
        diffusivity: float = 0.01,
        lambda_: float = 1.0,
        noise_type: Literal["additive", "multiplicative"] = "additive",
        sigma: float = 0.1,
        noise_alpha: float = 1.0,
        use_taming: bool = True,
        use_milstein: bool = False,
        order: int = 1,
        dealiasing_fraction: float = 2.0 / 3.0,
    ) -> None:
        if order != 1:
            raise ValueError(
                "StochasticAllenCahn supports only order=1 (exponential "
                "Euler–Maruyama). In v0.1.0, order=2 silently ran order 1 and "
                "order>=3 integrated the nonlinearity with the ETDRK3/4 "
                "half-step coefficient (#5). Higher-order integrators are "
                "tracked in #14."
            )

        for name, value in (
            ("noise_alpha", noise_alpha),
            ("sigma", sigma),
            ("diffusivity", diffusivity),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

        if use_milstein and noise_alpha <= num_spatial_dims / 2:
            raise ValueError(
                "Milstein requires trace-class noise (noise_alpha > "
                "num_spatial_dims / 2). See #8 for other noise regularity conditions."
            )

        if (
            sigma != 0
            and diffusivity > 0
            and num_spatial_dims >= 2
            and noise_alpha <= num_spatial_dims / 2 - 1
        ):
            if noise_type == "additive":
                message = (
                    "Additive noise with noise_alpha <= d/2 - 1 in d >= 2 has "
                    "divergent continuum point variance. The finite-grid model "
                    "is available. Continuum Allen–Cahn in this regime requires "
                    "a separate renormalisation treatment (#8)."
                )
            else:
                message = (
                    "Multiplicative noise with noise_alpha <= d/2 - 1 in d >= 2 "
                    "is too rough for the classical continuum pointwise product. "
                    "This stepper simulates the finite-grid collocation model (#8)."
                )
            warnings.warn(message, NoiseRegularityWarning, stacklevel=2)

        # ── Store all fields before super().__init__ ───────────────────
        # BaseStepper.__init__ immediately calls _build_linear_operator
        # and _build_nonlinear_fun, so every attribute must exist first.
        self.diffusivity = diffusivity
        self.lambda_ = lambda_
        self.noise_type = noise_type
        self.sigma = sigma
        self.noise_alpha = noise_alpha
        self.use_taming = use_taming
        self.use_milstein = use_milstein
        self._dealiasing_fraction_stoch = dealiasing_fraction

        # ── Precompute noise spectral arrays ───────────────────────────
        # BaseStepper builds and discards derivative_operator locally;
        # we recompute it here via the same pure function.
        deriv_op = spectral.build_derivative_operator(
            num_spatial_dims, domain_extent, num_points
        )
        linear = self._build_linear_operator(deriv_op)
        self._linear_symbol = eqx.error_if(
            linear.real,
            jnp.any(linear.imag != 0),
            "The Allen–Cahn linear symbol must be real.",
        )
        self._propagator = jnp.exp(dt * self._linear_symbol)
        # Physical wavenumber squared: |k|²_phys = -∑ᵢ (∂/∂xᵢ)²_k (real part)
        k_sq = -jnp.sum(deriv_op**2, axis=0, keepdims=True).real  # (1, *wn)
        dx_d = (domain_extent / num_points) ** num_spatial_dims

        q = (1.0 + k_sq) ** (-noise_alpha)
        if noise_type == "additive":
            variance_factor = dt * phi1(2 * dt * self._linear_symbol)
            self._increment_amplitude = sigma * jnp.sqrt(q * variance_factor / dx_d)
            self._wiener_variance = None
        else:
            self._increment_amplitude = jnp.sqrt(q * dt / dx_d)
            # Only the last rFFT axis omits negative frequencies. DC and the
            # even-grid Nyquist plane occur once in the full spectrum.
            weights = jnp.full(q.shape[-1], 2, dtype=q.dtype).at[0].set(1)
            if num_points % 2 == 0:
                weights = weights.at[-1].set(1)
            self._wiener_variance = (
                dt / dx_d * jnp.sum(weights * q) / num_points**num_spatial_dims
            )

        # ── BaseStepper.__init__ (confirmed signature) ─────────────────
        # Signature: (num_spatial_dims, domain_extent, num_points, dt,
        #             *, num_channels, order)
        # No dealiasing_fraction argument at the BaseStepper level.
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dt=dt,
            num_channels=1,
            order=order,
        )

    # ── Abstract methods required by BaseStepper ──────────────────────

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        """Linear operator L_k = ν Δ_k + λ.

        Matches ``exponax.stepper.reaction.AllenCahn`` exactly:

        - ``first_order_coefficient = λ``  → ETD (analytic)
        - ``third_order_coefficient = -λ`` → 𝒩(u) = -λu³ (explicit)

        Their sum recovers ``ν Δu + λu - λu³ = ν Δu + λ(u - u³)``. ✓
        """
        laplacian = jnp.sum(derivative_operator**2, axis=0, keepdims=True)
        return self.diffusivity * laplacian + self.lambda_

    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> TamedPolynomialNonlinearFun:
        """Nonlinear function 𝒩(u) = -λu³  (cubic part only).

        The complementary linear part ``+λu`` is handled by the ETD operator.
        We construct the cubic polynomial coefficients `[0, 0, 0, -λ]` and
        forward them to the tamed polynomial helper which handles dealiasing
        and optional Hutzenthaler-Jentzen taming.
        """
        # Polynomial coefficients for -λ u³
        coeffs = [0.0, 0.0, 0.0, -float(self.lambda_)]
        return TamedPolynomialNonlinearFun(
            self.num_spatial_dims,
            self.num_points,
            dt=self.dt,
            coefficients=coeffs,
            dealiasing_fraction=self._dealiasing_fraction_stoch,
            use_taming=self.use_taming,
        )

    # ── Public interface ───────────────────────────────────────────────

    def step(self, u: Float[Array, "1 * N"]) -> Float[Array, "1 * N"]:
        """Disabled: stochastic steppers require a PRNG key.

        Raises
        ------
        NotImplementedError
            Always.  Use ``stepper(u, key=key)`` or
            ``stepper.step_fourier(u_hat, key=key)`` instead.
        """
        raise NotImplementedError(
            "StochasticAllenCahn.step() is not available because a PRNG key "
            "is required for the noise increment.\n"
            "Use stepper(u, key=key) or stepper.step_fourier(u_hat, key=key)."
        )

    def __call__(
        self,
        u: Float[Array, "1 * N"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "1 * N"]:
        """Perform one EEM step in physical space.

        **Arguments:**

        - `u`: The current physical-space field, shape ``(1, N)`` for
            1-D, ``(1, N, N)`` for 2-D, etc.
        - `key`: A JAX PRNGKey consumed to draw the noise increment.
            Two calls with the same ``u`` but different keys produce
            independent outcomes.

        **Returns:**

        - `u_next`: Updated field, same shape as ``u``.

        !!! note
            For batched operation use ``jax.vmap``::

                u_batch = jax.vmap(lambda k: stepper(u0, key=k))(keys)
        """
        expected = (self.num_channels,) + _spatial_shape(
            self.num_spatial_dims, self.num_points
        )
        if u.shape != expected:
            raise ValueError(
                f"Expected shape {expected}, got {u.shape}. "
                "Use jax.vmap for batched operation."
            )
        u_hat = fft(u, num_spatial_dims=self.num_spatial_dims)
        u_hat_next = self.step_fourier(u_hat, key=key)
        return ifft(
            u_hat_next,
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
        ).real

    def sample_increment(self, key: PRNGKeyArray) -> Complex[Array, "1 ... (N//2)+1"]:
        """Draw an additive convolution or a σ-free Wiener increment as raw rFFT."""
        amplitude = self._increment_amplitude
        shape = (self.num_channels,) + _spatial_shape(
            self.num_spatial_dims, self.num_points
        )
        white = jax.random.normal(key, shape, dtype=amplitude.dtype)
        return amplitude * fft(white, num_spatial_dims=self.num_spatial_dims)

    def step_with_increment(
        self,
        u_hat: Complex[Array, "1 ... (N//2)+1"],
        increment: Complex[Array, "1 ... (N//2)+1"],
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        """Apply ETDRK1 drift and the supplied increment."""
        expected = self._increment_amplitude.shape
        if increment.shape != expected:
            raise ValueError(
                f"Expected increment shape {expected}, got {increment.shape}."
            )
        drift = super().step_fourier(u_hat)
        if self.noise_type == "additive":
            return drift + increment

        d, n = self.num_spatial_dims, self.num_points
        u = ifft(u_hat, num_spatial_dims=d, num_points=n)
        dw = ifft(increment, num_spatial_dims=d, num_points=n)
        kick = self.sigma * u * dw
        if self.use_milstein:
            kick = kick + 0.5 * self.sigma**2 * u * (dw**2 - self._wiener_variance)
        return drift + self._propagator * fft(kick, num_spatial_dims=d)

    def coarsen_increments(
        self, increments: Complex[Array, "r 1 ... (N//2)+1"]
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        """Combine r consecutive fine increments into one increment of length r·dt.

        Weight additive convolutions by the propagator for the remaining fine
        steps. Sum plain multiplicative Wiener increments before applying the
        collocation product and optional Milstein correction at the coarse step.
        """
        expected = self._increment_amplitude.shape
        if increments.shape[1:] != expected or increments.shape[0] < 2:
            raise ValueError(
                f"Expected r >= 2 increments each of shape {expected}, "
                f"got {increments.shape}."
            )
        if self.noise_type != "additive":
            return jnp.sum(increments, axis=0)

        def propagate(total, increment):
            return self._propagator * total + increment, None

        total, _ = jax.lax.scan(propagate, increments[0], increments[1:])
        return total

    def step_fourier(
        self,
        u_hat: Complex[Array, "1 ... (N//2)+1"],
        *,
        key: PRNGKeyArray,
    ) -> Complex[Array, "1 ... (N//2)+1"]:
        """Perform an EEM step with Fourier-space input and output.

        The multiplicative kick is evaluated on the physical grid. Use this
        method when the caller already holds the Fourier representation.

        **Arguments:**

        - `u_hat`: The real-valued Fourier transform of the current field,
            shape ``(1, ..., (N//2)+1)``.
        - `key`: A JAX PRNGKey.

        **Returns:**

        - `u_next_hat`: Fourier transform of the updated field, same
            shape as ``u_hat``.
        """
        return self.step_with_increment(u_hat, self.sample_increment(key))
