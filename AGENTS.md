# AGENTS.md — exponax-spde

## Purpose

exponax-spde adds stochastic PDE (SPDE) steppers on top of `exponax` (Felix Koehler, MIT
licence). The main stepper, `StochasticAllenCahn`, uses exponential Euler–Maruyama with
additive or multiplicative Q-Wiener noise and optional taming and Milstein correction.
The utilities cover rollouts, ensembles, the structure factor, Richardson extrapolation
and a Strang-split hybrid scaffold. Deterministic behaviour must agree with `exponax`.

## Layout

- `exponax_spde/stepper/stochastic/_stochastic_allen_cahn.py` contains the SPDE stepper.
- `exponax_spde/_stochastic_utils.py` contains the stochastic utilities.
- `exponax_spde/nonlin_fun/_tamed_polynomial.py` contains the tamed polynomial nonlinearity.
- Issue #3 tracks the vendored exponax modules under `exponax_spde/`. Do not extend them.
- `tests/test_stochastic/` contains the SPDE tests.
- `tests/test_builtin_solvers.py` contains the vendored-solver tests.
- `validation/validate_stochastic_allen_cahn.ipynb` is the validation notebook. Strip
  outputs and execution counts before committing.
- `docs/` contains documentation and images.
- `scratch/` is the local, git-ignored working area. Never commit it.

## Commands

- Build the in-project `.venv` with
  `POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --all-extras`.
  Use `.venv/bin/python` in VS Code and select the Jupyter kernel
  "Python (exponax-spde)". CI installs `.[test]` with pip.
- Install hooks with `poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg`.
- Lint and format with `poetry run pre-commit run --all-files`.
- Run fast tests with `JAX_ENABLE_X64=1 poetry run pytest -m "not slow" -q`.
- Run one pytest or JAX-heavy job at a time. Finish environment builds before running tests.
- Run the full suite in a memory-capped systemd user scope with swap disabled.
  Set `cap` to about 60% of the machine's total RAM, then run
  `systemd-run --user --scope -p "MemoryMax=$cap" -p MemorySwapMax=0 -- env JAX_ENABLE_X64=1 poetry run pytest -q`.
  Slow tests are marked `@pytest.mark.slow`.
- Run heavy jobs one at a time under `systemd-run --user --scope -p MemoryMax=8400M -p MemorySwapMax=0`.
- Keep each new or changed fast test within 10 s and 1.5 GiB peak RSS. Keep each slow
  test within 120 s and 4 GiB peak RSS. Measure with `pytest --durations` and
  `/usr/bin/time -v` on targeted selections under the cap.
- Chunk Monte Carlo sampling and accumulate statistics without materialising
  an array of shape `(M, T, ...)`. Verify JAX features on the 0.6.2 CI floor.

## Git and GitHub

- Use one branch per task, cut from an up-to-date main. Use `fix/issue-<n>-<slug>`,
  `chore/<slug>`, `docs/<slug>` or `feat/<slug>`.
- Use Conventional Commits. The commit-msg hook rejects other formats.
- Do not push, open pull requests, comment on issues, merge, tag or release unless the
  current task explicitly allows it. Never force-push main or rewrite published history.
- Never commit `scratch/`, notebook outputs, credentials or large binaries.

## Code conventions

- Convert both JAX array operands with `np.asarray` for `pytest.approx` assertions.
  Leave Python scalar operands as scalars.
- Keep traced JAX code free of Python side effects and global RNG state. Thread
  `jax.random` keys explicitly.
- Steppers are `equinox.Module`s. Declare configuration that drives Python branching
  static and precompute spectral arrays in `__init__`.
- Use quoted jaxtyping shape annotations such as `Float[Array, "1 *N"]`. Ruff UP037 is
  ignored globally and F821 is ignored per file for this reason. Keep the quotes and
  these ignores. Add new modules that use shape strings to the per-file ignores.
- Library code must avoid private exponax attributes such as `_integrator._exp_term`
  and `_coef_1`. Prefer exponax's public API. Import from `exponax._spectral` only when
  there is no public equivalent, and explain the choice in a comment.
- Store the linear operator and use `expm1`-based phi functions with gradient-safe
  `jnp.where`. Never recover the operator with `log(exp(L*dt))`.

## Numerical-correctness rules (mandatory)

- Keep physverify in the test extra (`>=0.9.1,<0.10`). Run solvers under JIT in
  float64 and pass only host NumPy arrays of final errors or statistics to physverify.
  Keep physverify out of runtime and traced JAX code.
- Derive verification references from the physics and keep standard-error-based checks
  primary. Verification tests use only the public `exponax_spde` API and installed
  `exponax`, so they are independent of the vendored modules (#3).

- Back every variance, scaling or convergence-order claim with a statistical test
  against an analytical reference. Include the derivation in the test docstring.
- Use two-sided acceptance bands for convergence orders. For Monte Carlo slopes,
  require the paired-bootstrap confidence interval to lie inside the prescribed band.
- Test changes to stochastic update formulas against an independent NumPy oracle
  using the same supplied increment and an independent deterministic reference.
- Derive every Δt and N factor in reference values. Never calibrate a reference to
  observed output.
- Use fixed PRNG keys and enough samples for a relative standard error below about 3%.
  Justify tolerances from that standard error.
- Couple Brownian paths for strong-convergence tests by combining fine increments into
  coarse increments with the scheme's propagation weights.
- With `sigma=0` and taming off, the stepper must match
  `exponax.stepper.reaction.AllenCahn` at the same `order`. Pass `order` explicitly.
  The exponax default is 2.
- Document each feature's noise regularity requirements. Milstein needs trace-class
  noise (`alpha > d/2`). The continuum Allen–Cahn model with additive white noise in
  `d >= 2` requires renormalisation.
- Cite literature with verified DOIs. Never invent or guess a reference.

## Code Review Rules

- Flag any Δt in a noise amplitude that is later multiplied by a time-integrated variance.
- Flag tests with Δt or N factors absent from the derivation in the test docstring.
- Flag uncoupled convergence tests and new uses of private exponax attributes.
- Flag citations without a verified DOI.
