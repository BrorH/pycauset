"""Import-time native configuration, kept out of ``__init__.py``.

These are one-shot side effects run after the native extension loads: the
OpenBLAS thread default and the optional GPU install hint. They are not public
API; ``__init__.py`` calls them once and forgets about them.
"""

from __future__ import annotations

import ctypes
import os


def configure_openblas_threads() -> None:
    """Set a sensible OpenBLAS thread default (best-effort).

    Too many threads add SMP-server overhead to small LAPACK factorizations
    (invert/determinant/eigh) while barely helping GEMM at the parity-benchmark
    size (n=1024). 8 balances both; users override with OPENBLAS_NUM_THREADS.
    The library is already loaded as a pycauset_core dependency, so a name-based
    ctypes lookup reuses the existing handle. The name differs per platform
    (libopenblas.dll / libopenblas.dylib / libopenblas.so); probe each and fall
    back to ctypes.util.find_library.
    """
    try:
        threads = int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or "0")
        if threads <= 0:
            # Cap to the visible CPU count so small runners (macOS CI exposes 3
            # vCPUs) are not asked to run more threads than they have cores.
            threads = min(8, os.cpu_count() or 8)

        dll = None
        for name in ("libopenblas.dll", "libopenblas.dylib", "libopenblas.so.0", "libopenblas.so"):
            try:
                dll = ctypes.CDLL(name)
                break
            except Exception:
                continue
        if dll is None:
            try:
                import ctypes.util as _ctypes_util

                found = _ctypes_util.find_library("openblas")
                if found:
                    dll = ctypes.CDLL(found)
            except Exception:
                dll = None

        if dll is not None:
            setter = getattr(dll, "openblas_set_num_threads", None)
            if setter is not None:
                setter(threads)
    except Exception:
        pass


def emit_gpu_install_hint_once() -> None:
    """Emit the one-time GPU install hint when appropriate (best-effort)."""
    try:
        from .gpu_hint import emit_gpu_install_hint

        def backend_available() -> bool:
            try:
                from pycauset import _pycauset_cuda as mod  # noqa: PLC0415

                return bool(mod.is_available())
            except Exception:
                return False

        emit_gpu_install_hint(backend_available=backend_available)
    except Exception:
        pass
