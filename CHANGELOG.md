# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - Unreleased

### Changed

- Minimum supported versions are Python 3.10, JAX 0.6.2, Equinox 0.12.2,
  jaxtyping 0.2.33, exponax >=0.2.0,<0.3, and matplotlib 3.8.1. The test
  extra requires pytest 7.0.0 and pytest-cov 2.3.1 or later.

- **Breaking:** Additive noise uses the correctly normalised stochastic
  convolution. Multiplicative noise uses a correctly normalised Wiener increment
  and an unprojected pointwise product. A fixed `sigma` now has different
  variance. Use the [additive σ migration](docs/noise_model.md#additive-migration) and
  [multiplicative σ migration](docs/noise_model.md#multiplicative-migration)
  to preserve an intended noise scale. Reported by @rcgiri-physics.
  [#4](https://github.com/GVourvachakis/exponax-spde/issues/4),
  [PR #18](https://github.com/GVourvachakis/exponax-spde/pull/18),
  [PR #20](https://github.com/GVourvachakis/exponax-spde/pull/20).
- **Breaking:** Noise parameters must be non-negative. Milstein requires
  trace-class noise, with `noise_alpha > num_spatial_dims / 2`.
  [#8](https://github.com/GVourvachakis/exponax-spde/issues/8),
  [PR #25](https://github.com/GVourvachakis/exponax-spde/pull/25),
  [#4](https://github.com/GVourvachakis/exponax-spde/issues/4),
  [PR #20](https://github.com/GVourvachakis/exponax-spde/pull/20).
- Deterministic drift orders 1–4 run the complete ETDRK stages, with order 1 as
  the default. Construction warns when higher order is combined with taming.
  The stochastic order follows the noise treatment. Reported by @rcgiri-physics.
  [#5](https://github.com/GVourvachakis/exponax-spde/issues/5),
  [PR #15](https://github.com/GVourvachakis/exponax-spde/pull/15),
  [#31](https://github.com/GVourvachakis/exponax-spde/issues/31),
  [PR #35](https://github.com/GVourvachakis/exponax-spde/pull/35).
- Richardson weak extrapolation keeps only terminal states. Spectrum utilities
  use the public exponax spectral API.
  [#11](https://github.com/GVourvachakis/exponax-spde/issues/11),
  [PR #16](https://github.com/GVourvachakis/exponax-spde/pull/16).
- Development dependencies are locked. CI covers supported dependency floors
  and schedules slow tests.
  [PR #6](https://github.com/GVourvachakis/exponax-spde/pull/6),
  [#13](https://github.com/GVourvachakis/exponax-spde/issues/13),
  [PR #27](https://github.com/GVourvachakis/exponax-spde/pull/27),
  [PR #28](https://github.com/GVourvachakis/exponax-spde/pull/28),
  [PR #30](https://github.com/GVourvachakis/exponax-spde/pull/30).

### Added

- Ensemble rollouts accept `stride`. Higher-order drift with taming emits
  a public, selectively filterable `TamingOrderWarning`.
  [PR #38](https://github.com/GVourvachakis/exponax-spde/pull/38).

- Citation metadata in `CITATION.cff`.
- Release automation builds and smoke-tests distributions before deploying
  to TestPyPI and PyPI through Trusted Publishing.

- Experimental additive and multiplicative increment APIs support replay,
  coarsening and coupled paths.
  [#4](https://github.com/GVourvachakis/exponax-spde/issues/4),
  [PR #18](https://github.com/GVourvachakis/exponax-spde/pull/18),
  [PR #20](https://github.com/GVourvachakis/exponax-spde/pull/20).
- Richardson weak extrapolation accepts a weak order and a terminal observable,
  reports standard errors and offers opt-in paired paths.
  [#10](https://github.com/GVourvachakis/exponax-spde/issues/10),
  [PR #29](https://github.com/GVourvachakis/exponax-spde/pull/29).
- `stochastic_structure_factor` estimates the ensemble spectrum without storing
  trajectories.
  [#11](https://github.com/GVourvachakis/exponax-spde/issues/11),
  [PR #16](https://github.com/GVourvachakis/exponax-spde/pull/16).
- Ensemble utilities accept `batch_size` and stochastic rollouts accept a save
  `stride`. Added float32 CI coverage and accelerator guidance.
  [#21](https://github.com/GVourvachakis/exponax-spde/issues/21),
  [PR #24](https://github.com/GVourvachakis/exponax-spde/pull/24).
- `NoiseRegularityWarning` identifies continuum-unsafe noise for positive
  diffusivity at construction.
  [#8](https://github.com/GVourvachakis/exponax-spde/issues/8),
  [PR #25](https://github.com/GVourvachakis/exponax-spde/pull/25).
- The test extra includes physverify for host-side order checks.
  [#19](https://github.com/GVourvachakis/exponax-spde/issues/19),
  [PR #20](https://github.com/GVourvachakis/exponax-spde/pull/20).
- `StochasticAllenCahn` is available from the package root. Runnable
  documentation examples are tested in CI.
  [PR #34](https://github.com/GVourvachakis/exponax-spde/pull/34).

### Fixed

- Strongly damped additive modes stay finite under JIT when the deterministic
  propagator underflows. The multiplicative kick and scalar-centred Milstein
  correction use the same linear propagator. Reported by @rcgiri-physics.
  [#4](https://github.com/GVourvachakis/exponax-spde/issues/4),
  [PR #18](https://github.com/GVourvachakis/exponax-spde/pull/18),
  [PR #20](https://github.com/GVourvachakis/exponax-spde/pull/20).
- The hybrid Strang scaffold preserves conjugate low Fourier modes in
  1–3 dimensions and validates both SSA concentration increments.
  [#7](https://github.com/GVourvachakis/exponax-spde/issues/7),
  [PR #23](https://github.com/GVourvachakis/exponax-spde/pull/23).
- Corrected the hybrid SSA and SPDE Milstein citations.
  [#9](https://github.com/GVourvachakis/exponax-spde/issues/9),
  [PR #26](https://github.com/GVourvachakis/exponax-spde/pull/26).
- Corrected the README rollout, ensemble and gradient examples.
  [PR #34](https://github.com/GVourvachakis/exponax-spde/pull/34).

- Regenerated the notebook and four README figures with the corrected noise
  model and added a text-only validation summary. [#17](https://github.com/GVourvachakis/exponax-spde/issues/17),
  [PR #37](https://github.com/GVourvachakis/exponax-spde/pull/37).

### Deprecated

- The hybrid Strang scaffold’s `domain_extent` parameter is deprecated and has
  no effect.
  [#7](https://github.com/GVourvachakis/exponax-spde/issues/7),
  [PR #23](https://github.com/GVourvachakis/exponax-spde/pull/23).

[0.2.0]: https://github.com/GVourvachakis/exponax-spde/releases/tag/v0.2.0
