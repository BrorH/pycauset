"""One-time GPU-install hint.

PyCauset ships CPU-only by default. Enabling the GPU backend needs two pieces:

1. the compiled CUDA plugin (``pycauset_cuda``), built from source with
   ``-DENABLE_CUDA=ON``, and
2. the ~500 MB of NVIDIA runtime it links against, installed via
   ``pip install pycauset[gpu]``.

To make the GPU backend *impossible to miss* without being annoying, we detect an
NVIDIA GPU through the *driver* API (``libcuda``/``nvcuda``, present with any
NVIDIA GPU, no CUDA runtime required) and, when a GPU is present but the backend
is not active, emit a single clear hint pointing at both steps.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Callable


def _driver_has_nvidia_gpu() -> bool:
    """True if an NVIDIA GPU is present, probed via the driver API (no CUDA runtime needed)."""
    try:
        if sys.platform == "win32":
            lib = ctypes.WinDLL("nvcuda.dll")
        else:
            # Linux (macOS has no NVIDIA CUDA).
            lib = ctypes.CDLL("libcuda.so.1")
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        if lib.cuInit(0) != 0:
            return False
        count = ctypes.c_int(0)
        lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuDeviceGetCount.restype = ctypes.c_int
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
        return count.value > 0
    except Exception:
        return False


def emit_gpu_install_hint(*, backend_available: Callable[[], bool]) -> None:
    """Emit a single install hint when a GPU is present but the backend is not installed."""
    if os.environ.get("PYCAUSET_GPU_HINT", "1").strip() in ("0", "false", "False", "no"):
        return

    try:
        if backend_available():
            return  # already installed, nothing to say
        if not _driver_has_nvidia_gpu():
            return  # no NVIDIA GPU, stay quiet (CPU user)
    except Exception:
        return

    print(
        "[PyCauset] NVIDIA GPU detected, but the GPU backend is not active "
        "(running on CPU).\n"
        "  Enable it with:  CMAKE_ARGS=\"-DENABLE_CUDA=ON\" pip install .\n"
        "  (builds the compiled plugin) plus  pip install \"pycauset[gpu]\"\n"
        "  (adds ~500 MB of CUDA runtime; set PYCAUSET_GPU_HINT=0 to silence this message)",
        file=sys.stderr,
    )
