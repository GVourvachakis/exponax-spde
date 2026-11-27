"""Exercise the standalone smoke checks on a GPU when one is available."""

import jax
import pytest

from scripts.gpu_smoke import run_smoke


@pytest.mark.gpu
def test_gpu_smoke():
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        pytest.skip("JAX has no GPU device.")
    original = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with jax.default_device(devices[0]):
            run_smoke()
    finally:
        jax.config.update("jax_enable_x64", original)
