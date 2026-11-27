# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Experimental `sample_increment`, `step_with_increment` and
  `coarsen_increments` methods support additive and multiplicative increment
  reuse and coupled paths.
- The test extra includes physverify for host-side order checks.

- `stochastic_structure_factor` computes the ensemble spectrum during rollout without
  storing the trajectories. [#11](https://github.com/GVourvachakis/exponax-spde/issues/11)

### Changed

- Richardson weak extrapolation retains only its final states.
  [#11](https://github.com/GVourvachakis/exponax-spde/issues/11)
- The spectrum utilities use the public `exponax.spectral.get_fourier_coefficients`.

### Changed (breaking)

- Multiplicative noise uses correctly normalised Wiener increments and an
  unprojected pointwise product. Milstein raises for non-trace-class noise
  ($\alpha \le d/2$). See the [migration guide](docs/noise_model.md).
  Reported by @rcgiri-physics in
  [#4](https://github.com/GVourvachakis/exponax-spde/issues/4).

- Additive noise uses the exact, correctly normalised stochastic convolution.
  The variance changes for a fixed `sigma`. See the
  [migration guide](docs/noise_model.md). Reported by @rcgiri-physics
  in [#4](https://github.com/GVourvachakis/exponax-spde/issues/4).

- `StochasticAllenCahn` now raises `ValueError` for any `order` other than 1.
  In v0.1.0, order 2 silently ran exponential Euler, and orders 3 and 4 used
  the ETDRK half-step coefficient for the cubic term. Reported by
  @rcgiri-physics in [#5](https://github.com/GVourvachakis/exponax-spde/issues/5).
  Higher-order stochastic integration is tracked in
  [#14](https://github.com/GVourvachakis/exponax-spde/issues/14).

### Fixed

- The multiplicative kick and the scalar-centred Milstein correction use the
  same linear propagator.

- Strongly damped additive modes stay finite under JIT when their
  deterministic propagator underflows.

### Internal

- Delegate the Allen–Cahn drift to the public `BaseStepper.step_fourier`
  method. Physical-space and Fourier-space stepping share one update.
