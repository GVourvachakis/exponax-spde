# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Experimental additive `sample_increment`, `step_with_increment` and
  `coarsen_increments` methods support increment reuse and coupled paths.

- `stochastic_structure_factor` computes the ensemble spectrum during rollout without
  storing the trajectories. [#11](https://github.com/GVourvachakis/exponax-spde/issues/11)

### Changed

- Richardson weak extrapolation retains only its final states.
  [#11](https://github.com/GVourvachakis/exponax-spde/issues/11)
- The spectrum utilities use the public `exponax.spectral.get_fourier_coefficients`.

### Changed (breaking)

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

- The hybrid Strang scaffold now preserves conjugate low Fourier modes
  in 1–3 dimensions and validates both SSA concentration increments.
  Passing `domain_extent` is deprecated and has no effect.
  [#7](https://github.com/GVourvachakis/exponax-spde/issues/7)

- Strongly damped additive modes stay finite under JIT when their
  deterministic propagator underflows.

### Internal

- Delegate the Allen–Cahn drift to the public `BaseStepper.step_fourier`
  method. Physical-space and Fourier-space stepping share one update.
