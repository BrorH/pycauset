"""Python facade for the native CUDA control surface.

`pycauset.cuda` is the native `_pycauset.cuda` submodule (`is_available`, `enable`,
`disable`, `force_backend`, `benchmark`, `current_device`, `set_pinning_budget`).
This module re-exports it so `import pycauset.cuda` works on a CPU-only install too:
the submodule always exists and its controls are safe no-ops when no CUDA device is
present.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def _load_native() -> ModuleType | None:
    try:
        from . import _pycauset

        return getattr(_pycauset, "cuda", None)
    except ImportError:
        return None


_native = _load_native()


def is_available() -> bool:
    if _native is None:
        return False
    try:
        return bool(_native.is_available())
    except Exception:
        return False


def __getattr__(name: str) -> Any:
    if _native is None:
        raise AttributeError(name)
    return getattr(_native, name)


def __dir__() -> list[str]:
    if _native is None:
        return ["is_available"]
    return sorted(set(dir(_native)) | {"is_available"})


__all__ = [n for n in __dir__() if not n.startswith("_")]
