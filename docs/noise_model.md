# Noise model and migration from v0.1.0

`StochasticAllenCahn` uses noise with physical amplitude $\sigma$
and spectral covariance $q_k=(1+|k|_{\rm phys}^2)^{-\alpha}$ on a periodic
box of side $\ell$. Write $h=\Delta t$, $P=N^d$, $a=dx^d=(\ell/N)^d$,
and $L_k=-\nu|k|_{\rm phys}^2+\lambda$.

## Fourier conventions

The forward FFT is unnormalised and the inverse divides by $P$. The
real FFT stores the nonnegative frequencies on its final axis. To sum over
the full spectrum, weight final-axis DC and even-$N$ Nyquist entries by
$w_k=1$, and all other entries by $w_k=2$. Each earlier axis already holds
both signs. With $m(k)$ the number of axes where $k$ is neither DC nor
Nyquist, exponax's `coef_extraction` divides the raw coefficient by
$D_k=P/2^{m(k)}$. Its `reconstruction` divisor is $R_k=P/w_k$.

For independent real $\xi(x)\sim N(0,1)$, $E|\widehat{\xi}_k|^2=P$ at every
stored mode. Boundary planes obey Hermitian symmetry. A mode whose indices
are all DC or Nyquist is purely real. The other modes have real and imaginary
variances $P/2$.

## Additive variance and the meaning of σ

The additive step draws one real white field and uses the increment
$$
\widehat O_k=\sigma\sqrt{q_k F_h(L_k)/a}\;\mathrm{rfft}(\xi)_k,
\qquad F_h(L)=h\varphi_1(2hL), \quad
\varphi_1(z)=\frac{\operatorname{expm1}(z)}{z},\quad \varphi_1(0)=1.
$$

$F_h$ is the integral of $e^{2Ls}$ from $0$ to $h$. The Itô isometry
therefore gives raw power $P\sigma^2q_kF_h(L_k)/a$. Define
$\mathrm{mean}_{\rm full}(f)=P^{-1}\sum_{\rm rfft}w_k f_k$.
The physical point variance is $\sigma^2\,\mathrm{mean}_{\rm full}(qF_h)/a$.
For a plain Wiener increment, use $F_h=h$. With white noise and negligible
drift this gives $\sigma^2h/a$ at each grid point.

$\sigma$ multiplies the continuum $Q$-Wiener forcing, with no time
step or grid correction required by the caller. The kernel stores real
$L_k$ and computes $F_h$ directly, including the large negative $L_k$
limit $1/(2|L_k|)$. The step applies the public order-1 deterministic
drift and adds $\widehat O$. The nonlinear dealiasing mask applies to
the drift and the additive draw covers the full resolved grid.

## Stationary spectrum

For $\lambda=0$ and $k\ne0$, divide the one-step raw power by
$1-e^{-2\nu |k|^2h}$ and by $D_k^2$. This gives the invariant spectrum
returned by `structure_factor` and `stochastic_structure_factor`:
$$
S_k=\frac{P\sigma^2q_k}{2\nu|k|^2 a D_k^2}
   =\frac{4^{m(k)}\sigma^2q_k}{2\nu|k|^2\ell^d}.
$$
It is independent of $h$. The spatial mean is Brownian with variance
$\sigma^2t/\ell^d$ from zero, so the stationary comparison excludes DC.

## Experimental increment API

- `sample_increment(key)` returns the additive convolution in raw rFFT
  units, including $\sigma$, or a σ-free multiplicative Wiener increment.
- `step_with_increment(u_hat, increment)` applies the deterministic step
  and applies the supplied increment according to the noise model. It
  validates the increment shape and takes no key.
- `fine.coarsen_increments(increments)` combines $r\ge2$ consecutive fine
  increments along the leading axis. With $E_h=e^{hL_k}$, it applies
  $\sum_{j=0}^{r-1} E_h^{r-1-j}\widehat O_j$ to additive increments.
  It sums the multiplicative Wiener increments before applying the
  collocation product and optional Milstein term. Match geometry,
  coefficients, dtype and covariance between the fine and coarse steppers.

The methods support JIT and batching over paths. Use `equinox.filter_jit`
or JIT a closure such as `jax.jit(lambda x: fine.coarsen_increments(x))`.
Sample the fine draws once and coarsen them for the coarse path to share
one Brownian path. With additive noise and linear drift, fine and coarse
steps agree to round-off. With nonlinear drift or multiplicative noise,
coupling preserves the correct marginal noise and exposes the strong
time-discretisation error. [Higham (2001)](https://doi.org/10.1137/S0036144500378302)

## Additive migration

The legacy draw counted $h$ both in its base variance and through $F_h$.
It also multiplied circular complex samples by $R_k$. The real inverse
projects the last-axis boundary planes, halving their expected power.
Let $g_k=1/2$ on these planes and $g_k=1$ elsewhere. At fixed $\sigma$,
$$
\frac{V_{\rm new}}{V_{\rm old}}
=\frac{P\sum w_k q_k F_h(L_k)}
       {h\sum w_k g_k R_k^2 q_k F_h(L_k)}.
$$

In 1-D with even $N$, white noise and negligible drift, the ratio is
$4/((N+2)h)$. A matching one-step intensity for an illustration uses
$\sigma_{\rm matched}=\sigma_{\rm old}\sqrt{(N+2)h/4}$ in that case.
For other settings divide by the square root of the weighted ratio.
One matched value preserves only the aggregate variance because interior
and boundary mode ratios differ. Several high-resolution 2-D and 3-D
examples received too much legacy noise, for example the 2-D $N=256$
and 3-D $N=32$ validation figures with $h=0.0005$. Keep physical $\sigma$
fixed in quantitative comparisons.

## Multiplicative noise and Milstein

The plain Wiener draw contains no $\sigma$:
$$
\Delta\widehat W_k=\sqrt{q_k h/a}\;\mathrm{rfft}(\xi)_k,
\qquad
v_W=E[\Delta W(x)^2]=\frac{h}{aP}\sum_{\rm rfft} w_k q_k.
$$
The final-axis weights account for all frequencies, including the odd and
even grid cases. Translation invariance makes $v_W$ the same at every
grid point.

Write $D_h(\widehat u)$ for the order-1 deterministic step. Multiply the
noise and the step-start state at the grid points (collocation). The drift
uses its usual dealiasing mask. The step is
$$
\widehat u_{n+1}
=D_h(\widehat u_n)
+ E_h\,\mathcal F\!\left[\sigma u_n\Delta W
   + \tfrac12\sigma^2 u_n(\Delta W^2-v_W)\right],
$$
with the quadratic term included only when `use_milstein=True`. The
propagated Euler kick follows the stochastic exponential Euler form.
[Lord & Tambue (2013)](https://doi.org/10.1093/imanum/drr059)

Milstein uses the commutative correction for the linear diffusion
$G(u)=\sigma u$. The centred square has zero expectation and variance
$2v_W^2$. Euler and quadratic kicks have zero cross covariance by
Gaussian symmetry. At $L=0$ the conditional variance is
$u_n^2(\sigma^2v_W+\tfrac12\sigma^4v_W^2)$ with Milstein and
$u_n^2\sigma^2v_W$ with Euler–Maruyama. Milstein requires
$\sum_k q_k<\infty$ in the continuum, equivalently $\alpha>d/2$.
Enabling it for $\alpha\le d/2$ raises `ValueError`.
[Jentzen & Röckner (2015)](https://doi.org/10.1007/s10208-015-9247-y)

### Verification by geometric Brownian motion

On the unit box, set $\lambda=0$, start from a uniform field and suppress
nonconstant noise modes with $\alpha=20$. In the DC-only limit the field
obeys $du=\sigma u\,dW$ with exact endpoint
$u(T)=u_0\exp(\sigma W_T-\sigma^2 T/2)$. Sum the same fine increments
into coarse steps. The strong RMS orders are $1/2$ for Euler–Maruyama
and $1$ for Milstein. The tests use paired bootstrap bounds as the
primary check and physverify as a host-side check on final RMS errors.
[Kloeden & Platen (1992)](https://doi.org/10.1007/978-3-662-12616-5)

### When Milstein improves on Euler–Maruyama

For smooth coefficients and test functions under the usual moment and
regularity assumptions, Euler–Maruyama and Milstein both have weak order
one. The Milstein correction increases the strong order from one half to
one in the commutative setting. [Kloeden & Platen (1992)](https://doi.org/10.1007/978-3-662-12616-5)

Resolving that gain requires commutative noise, which the diagonal
collocation diffusion supplies, and small $\sigma^2 v_W$ per step. The
drift error must be small enough to expose the stochastic error, and
the noise must be sufficiently smooth relative to the stiffness of
the resolved modes. [Jentzen & Röckner (2015)](https://doi.org/10.1007/s10208-015-9247-y),
[Lord & Tambue (2013)](https://doi.org/10.1093/imanum/drr059)

Weak errors and wall times describe the chosen observable and workload.
They cannot rank the schemes by strong accuracy. Compare terminal path
errors using the same Brownian increments across resolutions. The v0.1.0
comparisons used independent seeds and the former Milstein term.

### Multiplicative migration

The v0.1.0 draw counted time twice, used reconstruction scaling and
applied the linear-convolution factor to the noise before multiplication.
It also projected the state and stochastic product. Its Milstein term
subtracted a field in place of $v_W$ and carried an extra time-step
factor. The variance change depends on the grid, colour, step size and
state. Keep physical $\sigma$ fixed in quantitative work. For an
illustration, match the conditional one-step variance at its initial
field if the legacy noise level matters. Such a match preserves one
aggregate variance and can leave different modal power and mean.

Continuum regularity warnings and updated validation figures have
separate issues.
[Noise correction #4](https://github.com/GVourvachakis/exponax-spde/issues/4),
[regularity #8](https://github.com/GVourvachakis/exponax-spde/issues/8),
[figure regeneration #17](https://github.com/GVourvachakis/exponax-spde/issues/17)
