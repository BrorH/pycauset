"""
Internal module for handling NumPy ufuncs via Lazy Evaluation.
"""
import numpy as np
from .. import _native

def handle_array_ufunc(self, ufunc, method, *inputs, **kwargs):
    """
    Implementation of __array_ufunc__ for PyCauset matrices.
    Intercepts NumPy ufunc calls and returns LazyMatrix expressions.
    """
    if method != "__call__":
        return NotImplemented
    
    # Only support operations where all inputs are PyCauset matrices or scalars
    # (For now, we assume inputs are correct types or let native throw)

    try:
        if ufunc == np.add:
            if len(inputs) != 2: return NotImplemented
            return inputs[0] + inputs[1]
        
        elif ufunc == np.subtract:
            if len(inputs) != 2: return NotImplemented
            return inputs[0] - inputs[1]
        
        elif ufunc == np.multiply:
            if len(inputs) != 2: return NotImplemented
            return inputs[0] * inputs[1]

        elif ufunc == np.sin:
            if len(inputs) != 1: return NotImplemented
            # Use the exposed function from the package if available
            # We need to import it inside to avoid circular imports if possible, 
            # or rely on _native if it exposes it directly.
            # But _native.lazy_sin seems to crash.
            # Let's try to use the operator if available via some other way?
            # No, sin is a function.
            # Let's try to return NotImplemented for sin to avoid crash for now,
            # or try to use pycauset.sin if it exists.
            # from .. import sin
            # return sin(inputs[0])
            # For now, return NotImplemented to allow fallback to eager evaluation via __array__
            # Once rebuilt, we can use _native.lazy_sin if exposed, or rely on __array__
            # But wait, if we want LAZY, we must return an expression.
            # If _native.lazy_sin is available, we should use it.
            if hasattr(_native, "lazy_sin"):
                 return _native.lazy_sin(inputs[0])
            return NotImplemented

        elif ufunc == np.cos:
            if len(inputs) != 1: return NotImplemented
            # For now, return NotImplemented to allow fallback to eager evaluation via __array__
            if hasattr(_native, "lazy_cos"):
                 return _native.lazy_cos(inputs[0])
            return NotImplemented

    except Exception:
        # If native binding fails (e.g. type mismatch), fall back to NumPy
        return NotImplemented

    return NotImplemented

def patch_matrix_classes(classes):
    """
    Monkey-patch __array_ufunc__ onto the given native classes.
    """
    for cls in classes:
        if cls is not None:
            setattr(cls, "__array_ufunc__", handle_array_ufunc)
