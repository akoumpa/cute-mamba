from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from functools import cache

import torch


@cache
def _has_usable_nvcc() -> bool:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return False
    try:
        return (
            subprocess.run(
                [nvcc, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@cache
def _host_supports_cute_dsl() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    if not _has_usable_nvcc():
        return False
    try:
        return importlib.util.find_spec("cutlass.cute") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def is_cute_dsl_available() -> bool:
    """Return whether CuTe DSL dispatch is enabled and usable on this host."""
    if os.environ.get("MAMBA_DISABLE_BACKEND_DISPATCH") == "1":
        return False
    if os.environ.get("MAMBA_CUTE_DSL", "1") == "0":
        return False
    return _host_supports_cute_dsl()


__all__ = ["is_cute_dsl_available"]
