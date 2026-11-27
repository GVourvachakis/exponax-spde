# Additive noise and migration from v0.1.0

`StochasticAllenCahn` uses additive noise with physical amplitude $\sigma$
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

## Variance and the meaning of σ

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
  units, including $\sigma$.
- `step_with_increment(u_hat, increment)` applies the deterministic step
  and adds the supplied Fourier increment. It validates the increment
  shape and takes no key.
- `fine.coarsen_increments(increments)` combines $r\ge2$ consecutive fine
  increments along the leading axis. With $E_h=e^{hL_k}$, it applies
  $\sum_{j=0}^{r-1} E_h^{r-1-j}\widehat O_j$ for the step of length $rh$.
  Keep geometry, coefficients, dtype and covariance the same between
  the fine and coarse steppers.

The methods support JIT and batching over paths. JIT a closure over the
stepper, such as `jax.jit(lambda x: fine.coarsen_increments(x))`. Sample the fine
increments once and coarsen the same draws for the coarse path. Two fine
linear steps and the corresponding coarse step agree to round-off. For
nonlinear drift this coupling preserves the correct marginal noise
while exposing the time-discretisation error.

## Migration

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

The multiplicative update still uses the legacy discretisation. Its
normalisation, collocation product, Milstein correction and increment API
are pending #4b. Continuum regularity warnings are tracked separately.
[Noise correction #4](https://github.com/GVourvachakis/exponax-spde/issues/4),
[regularity #8](https://github.com/GVourvachakis/exponax-spde/issues/8),
[figure regeneration #17](https://github.com/GVourvachakis/exponax-spde/issues/17)
