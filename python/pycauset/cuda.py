"""Python facade for the native cuda submodule.

This makes `import pycauset.cuda` work even when the native CUDA plugin
is unavailable, so tests can skip cleanly.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any

import pycauset as _pc


def _load_native() -> ModuleType | None:
    try:
        return getattr(_pc, "cuda")
    except Exception:
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
