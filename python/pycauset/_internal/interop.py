from __future__ import annotations
from typing import Any
import numpy as np

def _array_ufunc(self: Any, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
    """
    NumPy ufunc protocol implementation for PyCauset matrices.
    Allows using numpy functions like np.sin(A) on PyCauset matrices.
    """
    if method == '__call__':
        # Lazy import to avoid circular dependency
        import pycauset
        
        # Unary operations
        if len(inputs) == 1:
            if ufunc == np.sin:
                return pycauset.sin(inputs[0])
            if ufunc == np.cos:
                return pycauset.cos(inputs[0])
            if ufunc == np.exp:
                return pycauset.exp(inputs[0])
            if ufunc == np.log:
                return pycauset.log(inputs[0])
            if ufunc == np.negative:
                return -inputs[0]
            if ufunc == np.invert:
                return ~inputs[0]
        
        # Binary operations
        if len(inputs) == 2:
            if ufunc == np.add:
                return inputs[0] + inputs[1]
            if ufunc == np.subtract:
                return inputs[0] - inputs[1]
            if ufunc == np.multiply:
                return inputs[0] * inputs[1]
            if ufunc == np.divide:
                return inputs[0] / inputs[1]
                
    return NotImplemented

def patch_interop(cls: Any) -> None:
    """Patch __array_ufunc__ onto the given class."""
    cls.__array_ufunc__ = _array_ufunc
