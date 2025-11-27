"""Python package wrapper for the native pycauset extension."""
from importlib import import_module as _import_module

_native = _import_module(".pycauset", package=__name__)

# Re-export everything from the native module
def __getattr__(name):
    return getattr(_native, name)

__all__ = [name for name in dir(_native) if not name.startswith("__")]
