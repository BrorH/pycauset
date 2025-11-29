"""Proxy package that loads the actual implementation from python/pycauset."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent / "python" / "pycauset"
_pkg_init = _pkg_root / "__init__.py"

if not _pkg_init.exists():  # pragma: no cover - developer misconfiguration
    raise ImportError("Missing python/pycauset package directory")

_spec = importlib.util.spec_from_file_location(
    __name__,
    _pkg_init,
    submodule_search_locations=[str(_pkg_root)],
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
assert _spec.loader is not None  # For type checkers
_spec.loader.exec_module(_module)