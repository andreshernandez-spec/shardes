"""Shared pytest configuration.

Sets the simulated-device flag before jax is imported so it cannot be forgotten on the
command line (docs/conventions.md, "Tests"). The gpu marker and its default deselection
live in pyproject.toml.

Budget: the whole suite runs on CPU, no GPU, no network, under two minutes.
"""

import os
import sys

N_SIMULATED_DEVICES = 8

# This has to happen before the first `import jax` in the process. conftest is imported
# before test modules, so it is the only place it reliably works. If jax is already loaded
# the flag is too late and every sharding test silently becomes a 1-device test that
# passes for the wrong reason, so fail loudly instead.
if "jax" in sys.modules:
    raise RuntimeError(
        "jax was imported before conftest ran, so the simulated-device flag is too late. "
        "Look for a jax import that happens at collection time outside tests/."
    )

# A CUDA jaxlib is installed for the Phase 0 sweep, so jax would otherwise default to the
# GPU, report one device, and quietly discard the simulated mesh. Tests are CPU-only by
# convention, so pin the platform instead of depending on what happens to be installed.
# setdefault, so `JAX_PLATFORMS=cuda pytest -m gpu tests/gpu/` still works.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _flags:
    os.environ["XLA_FLAGS"] = (
        f"{_flags} --xla_force_host_platform_device_count={N_SIMULATED_DEVICES}"
    ).strip()

import jax  # noqa: E402


def pytest_report_header(config):
    """Put the device count in the header so a 1-device run is visible, not inferred."""
    return (
        f"jax {jax.__version__}, {jax.device_count()} device(s), "
        f"platform {jax.devices()[0].platform}, x64 {jax.config.jax_enable_x64}"
    )
