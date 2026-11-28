# Migrating to exponax-spde 0.3

Use `exponax` for deterministic functionality and `exponax_spde` for its
stochastic add-on. Installing the add-on installs exponax.

| Former name | Replacement |
| --- | --- |
| `exponax_spde.BaseStepper` | `exponax.BaseStepper` |
| `exponax_spde.ForcedStepper` | `exponax.ForcedStepper` |
| `exponax_spde.RepeatedStepper` | `exponax.RepeatedStepper` |
| `exponax_spde.derivative` | `exponax.derivative` |
| `exponax_spde.fft` | `exponax.fft` |
| `exponax_spde.ifft` | `exponax.ifft` |
| `exponax_spde.get_spectrum` | `exponax.get_spectrum` |
| `exponax_spde.make_grid` | `exponax.make_grid` |
| `exponax_spde.rollout` | `exponax.rollout` |
| `exponax_spde.repeat` | `exponax.repeat` |
| `exponax_spde.stack_sub_trajectories` | `exponax.stack_sub_trajectories` |
| `exponax_spde.build_ic_set` | `exponax.build_ic_set` |
| `exponax_spde.wrap_bc` | `exponax.wrap_bc` |
| `exponax_spde.metrics` | `exponax.metrics` |
| `exponax_spde.etdrk` | `exponax.etdrk` |
| `exponax_spde.ic` | `exponax.ic` |
| `exponax_spde.viz` | `exponax.viz` |
| `exponax_spde.spectral` | `exponax.spectral` |
| `exponax_spde.FourierInterpolator` | `exponax.FourierInterpolator` |
| `exponax_spde.map_between_resolutions` | `exponax.map_between_resolutions` |
| `exponax_spde.stepper.HeatEquation` | `exponax.stepper.Diffusion` |
| `exponax_spde.stepper.Burgers` | `exponax.stepper.Burgers` |
| `exponax_spde.stepper.generic` | `exponax.stepper.generic` |
| `exponax_spde.stepper.reaction` | `exponax.stepper.reaction` |
| Deterministic names under `exponax_spde.nonlin_fun` | Same names under `exponax.nonlin_fun` |
| Helpers under `exponax_spde._spectral` | Public helpers under `exponax.spectral` |
| `exponax_spde.stepper.stochastic.StochasticAllenCahn` | `exponax_spde.StochasticAllenCahn` or `exponax_spde.stepper.StochasticAllenCahn` |
| Warnings under `exponax_spde.stepper.stochastic` | Same warning classes at the `exponax_spde` root |

`TamedPolynomialNonlinearFun` is available at the root and under
`exponax_spde.nonlin_fun`. Old deterministic attribute accesses on the root
and `stepper` raise errors identifying the replacement in 0.3. The migration
messages are scheduled for removal in 0.4. Private vendored modules were removed.

## Initial conditions

The required exponax 0.2.1 produces a single output channel for
`Discontinuities` even on a multidimensional grid. Update shape expectations
from `(d, *N)` to `(1, *N)` where older code observed repeated masks.
`SineWaves1d` accepts traced offsets. Concrete offsets still follow its
normalisation constraints.

## Extras

Use `pip install "exponax[vape4d]"` for optional volume rendering in the
validation notebook. The `docs` extra was removed. The documentation is in
Markdown, and the add-on retains its `test` and `dev` extras.
