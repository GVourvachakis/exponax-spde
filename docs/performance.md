# Performance and accelerator guidance

## Choose a device and precision

Run `jax.devices()` to confirm that JAX sees the accelerator. Set
`JAX_PLATFORMS=cpu` before importing JAX to reproduce a CPU run, or use
`jax.default_device(jax.devices("gpu")[0])` to place a local calculation.
Follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

Use `JAX_ENABLE_X64=1` and float64 on CPU for reference comparisons,
especially when checking convergence near round-off. Exact-equality
checks target CPU float64. Use float32 for production workloads once the
statistics and precision meet your requirements. TPUs lack native
float64 arithmetic.

On an AMD Ryzen 7 5800H, the float32 CPU throughput was about 1.5× the
float64 throughput for 1-D N=128 and 1.8–1.9× for 2-D N=64. These
measurements used 256 paths and 2,000 compiled steps, with
`block_until_ready()` and compilation outside the timer. Benchmark on the
target accelerator to measure its throughput.

For GPU memory pressure, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
before importing JAX to allocate on demand, or limit the pool with
`XLA_CLIENT_MEM_FRACTION`. The older fraction setting is
`XLA_PYTHON_CLIENT_MEM_FRACTION`.
[JAX GPU memory settings](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

## Reduce working memory

- `batch_size` limits how many paths a mapped kernel processes together in
  `stochastic_ensemble_rollout`, `stochastic_structure_factor` and
  `richardson_weak_extrapolation`. `None` uses the full vmap batch. Compare
  peak memory and throughput on your workload. An ensemble rollout still
  returns all retained states, and the streamed spectrum still carries
  current fields for all paths.
- `stochastic_rollout(stepper, num_steps, stride=k)` saves steps
  `k, 2*k, …`. It saves step zero if `include_init=True` and omits
  the final partial block. The saved states use the same keys as those
  steps in a stride-one rollout. Upstream exponax's `rollout` has no
  stride argument.
- Use compiled scans for time loops. The Strang scaffold calls Python
  for each SSA half-step and returns to the host on accelerators.
- Prefer FFT sizes with small prime factors and benchmark nearby sizes
  that meet your resolution requirements. The corrected noise model in
  #20 traces 5 FFTs per additive step and 8 for either multiplicative
  scheme via the physical-space interface.
  [FFT performance issue #22](https://github.com/GVourvachakis/exponax-spde/issues/22)
  · [Exponax Performance Hints](https://fkoehler.site/exponax/examples/performance_hints/)

In the CPU runs with the corrected multiplicative implementation, Milstein
cost about 0–7% more than Euler–Maruyama. At 256 steps the coupled fixed-grid
SPDE test had about 14× lower strong RMS error with Milstein. Prefer it
for pathwise accuracy when `alpha > d/2` and the commutative, smooth-noise
regime resolves the stochastic error. Both methods have weak order one
under the usual smoothness assumptions.
[Noise model](noise_model.md) · [Corrected implementation and evidence #20](https://github.com/GVourvachakis/exponax-spde/pull/20)

## Distribute the key batch

This single-host example shards independent keys and their initial fields.
Spatial axes stay local. Sharding both inputs gives the scan a per-device
state from its first step. The path count is a multiple of the device count.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from exponax_spde import stochastic_rollout
from exponax_spde.stepper.stochastic import StochasticAllenCahn

devices = np.asarray(jax.local_devices())
mesh = Mesh(devices, ("path",))
paths = PartitionSpec("path")
sharding = NamedSharding(mesh, paths)
keys = jax.device_put(
    jax.random.split(jax.random.key(0, impl="threefry2x32"), 4 * devices.size),
    sharding,
)
stepper = StochasticAllenCahn(1, 1.0, 128, 0.001)
dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
initial_fields = jax.device_put(
    jnp.zeros((keys.shape[0], 1, 128), dtype=dtype), sharding
)
rollout = stochastic_rollout(stepper, 40, include_init=False, stride=4)


def local_ensemble(fields, local_keys):
    return jax.vmap(rollout)(fields, local_keys)


run = jax.jit(
    jax.shard_map(local_ensemble, mesh=mesh, in_specs=(paths, paths), out_specs=paths)
)
states = run(initial_fields, keys)
states.block_until_ready()
```

Use typed Threefry keys when comparing across chunk sizes and shardings.
Record the seed, RNG implementation, JAX version and dtype with results.
Experimental `rbg` and `unsafe_rbg` can be faster on some devices, but their
`vmap` batching can generate a full batch from its first key, changing
which draws each path receives. [JAX PRNG configuration](https://docs.jax.dev/en/latest/jax.random.html)

## Reuse compilations

Configure a persistent cache before the first compilation. For example,
set `JAX_COMPILATION_CACHE_DIR` to a user-owned directory. The
`jax_enable_compilation_cache` setting enables caching, and
`jax_persistent_cache_min_compile_time_secs` and
`jax_persistent_cache_min_entry_size_bytes` control which executables
are written. Fix shapes, dtypes and rollout parameters where practical.
[JAX persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)

## Run the smoke checks on a Colab GPU

1. In Colab, choose **Runtime → Change runtime type** and select a GPU
   such as T4 if a free GPU is available.
2. In a fresh notebook cell, install the CUDA build of JAX and the
   project from main. Restart the runtime after installation if JAX
   has already been imported.

   ```ipython
   %pip install -U "jax[cuda12]" "git+https://github.com/GVourvachakis/exponax-spde.git@main"
   ```

3. Download and run the script in a fresh float32 process.

   ```ipython
   !curl -fsSL https://raw.githubusercontent.com/GVourvachakis/exponax-spde/main/scripts/gpu_smoke.py -o /tmp/exponax_gpu_smoke.py
   !env -u JAX_ENABLE_X64 python /tmp/exponax_gpu_smoke.py
   ```

4. Confirm the printed device is a GPU and paste the device list,
   analytical checks and median throughput into
   [issue #21](https://github.com/GVourvachakis/exponax-spde/issues/21).
