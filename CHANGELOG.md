# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `stochastic_structure_factor` computes the ensemble spectrum during rollout without
  storing the trajectories. [#11](https://github.com/GVourvachakis/exponax-spde/issues/11)

### Changed

- Richardson weak extrapolation retains only its final states.
  [#11](https://github.com/GVourvachakis/exponax-spde/issues/11)
- The spectrum utilities use the public `exponax.spectral.get_fourier_coefficients`.

### Changed (breaking)

- `StochasticAllenCahn` now raises `ValueError` for any `order` other than 1.
  In v0.1.0, order 2 silently ran exponential Euler, and orders 3 and 4 used
  the ETDRK half-step coefficient for the cubic term. Reported by
  @rcgiri-physics in [#5](https://github.com/GVourvachakis/exponax-spde/issues/5).
  Higher-order stochastic integration is tracked in
  [#14](https://github.com/GVourvachakis/exponax-spde/issues/14).

### Internal

- Delegate the Allen–Cahn drift to the public `BaseStepper.step_fourier`
  method. Physical-space and Fourier-space stepping share one update.
