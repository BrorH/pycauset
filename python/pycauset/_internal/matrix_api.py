"""Matrix construction (`pycauset.matrix(...)`) and display hooks for the native classes.

There is one native class per structure (DenseMatrix, TriangularMatrix, ...) and the dtype
is an attribute. Construction is deterministic: the dtype comes from `dtype=` or from the
input data (NumPy rules, Python ints -> int32); nothing depends on the matrix size, and the
structure of the result never depends on the values (a list of lists is always dense, like
a NumPy array).
"""

from __future__ import annotations

from typing import Any

from . import export_guard
from . import dtypes as _dtypes
from .dtypes import infer_sequence_dtype as _infer_sequence_dtype
from .dtypes import normalize_dtype as _normalize_dtype
from .formatting import MatrixMixin
from .native_threshold import temporary_native_memory_threshold

_np: Any | None = None
_native: Any | None = None

# PyCauset dtype name -> NumPy dtype name (the one map, dtypes.py).
_NUMPY_DTYPE = _dtypes.NUMPY_EXPORT


def configure(*, np_module: Any | None, native: Any) -> None:
    global _np, _native
    _np = np_module
    _native = native


def dtype_or_raise(dtype: Any) -> str | None:
    """The internal dtype name; None only for `dtype=None`. Anything unparsable raises (a typo
    became float64 in `vector` / `zeros`, review 3.8 C-3)."""
    import numpy as np

    target = _normalize_dtype(dtype, np_module=np)
    if dtype is not None and target is None:
        raise TypeError(f"unsupported dtype: {dtype!r}")
    return target


def check_storage(storage: Any) -> None:
    if storage not in (None, "ram", "disk"):
        raise ValueError(
            f"storage must be 'ram' or 'disk', not {storage!r} (RAM-first is the default; "
            "'disk' places the object in a file)."
        )


def _placement(storage: Any, max_in_ram_bytes: int | None, nbytes: int | None) -> Any:
    """Where a new object goes: `storage="disk"` forces a file; `max_in_ram_bytes` sends an
    object larger than it to a file. Otherwise the global threshold decides."""
    from contextlib import nullcontext

    if storage == "disk":
        return temporary_native_memory_threshold(_native, 0)
    if max_in_ram_bytes is not None and nbytes is not None and nbytes > max_in_ram_bytes:
        return temporary_native_memory_threshold(_native, int(max_in_ram_bytes))
    return nullcontext()


def allocate(shape: tuple[int, ...], dtype: Any, *, storage: Any = None,
             max_in_ram_bytes: int | None = None) -> Any:
    """A zero-filled dense vector ((n,)) or matrix ((rows, cols)) of `dtype` (default float64)."""
    from types import SimpleNamespace

    target = dtype_or_raise(dtype) or "float64"
    check_storage(storage)
    nbytes = export_guard.estimate_materialized_bytes(SimpleNamespace(shape=shape, dtype=target))
    with _placement(storage, max_in_ram_bytes, nbytes):
        if len(shape) == 1:
            return _native.DenseVector(int(shape[0]), dtype=target)
        return _native.DenseMatrix(int(shape[0]), int(shape[1]), dtype=target)


def from_array(data: Any, target: str | None, *, storage: Any = None,
               max_in_ram_bytes: int | None = None) -> Any:
    """A 1-D or 2-D ndarray as a native vector / matrix of dtype `target` (None: the array's).
    The one import path of `matrix` and `vector`."""
    import numpy as np

    if target == "complex_float16":  # no NumPy dtype: filled element-wise from complex64
        out = allocate(data.shape, "complex_float16", storage=storage, max_in_ram_bytes=max_in_ram_bytes)
        for idx, z in np.ndenumerate(data.astype(np.complex64)):
            if z != 0:
                if data.ndim == 1:
                    out[idx[0]] = complex(z)
                else:
                    out.set(idx[0], idx[1], complex(z))
        return out
    if target is not None and data.dtype != np.dtype(_NUMPY_DTYPE[target]):
        data = data.astype(_NUMPY_DTYPE[target])
    data = np.ascontiguousarray(data)
    with _placement(storage, max_in_ram_bytes, int(data.nbytes)):
        return _native._asarray(data)


def Matrix(size_or_data: Any, dtype: Any = None, **kwargs: Any) -> Any:
    """Build a dense matrix from a size, a NumPy array or nested sequences."""
    if _native is None:
        raise RuntimeError("pycauset matrix construction is not configured")
    import numpy as np

    target = dtype_or_raise(dtype)
    max_in_ram_bytes = kwargs.pop("max_in_ram_bytes", None)
    storage = kwargs.pop("storage", None)
    check_storage(storage)
    if "backing_file" in kwargs:
        raise TypeError(
            "matrix() does not accept 'backing_file'. For persistent storage use save()/load(); "
            "for placement in a file use storage='disk' or set_memory_threshold()."
        )
    if kwargs:
        raise TypeError(f"matrix() got unexpected keyword arguments: {sorted(kwargs)}")

    # 1) A size: an n x n dense matrix of zeros.
    if isinstance(size_or_data, (int, float)) and not isinstance(size_or_data, bool) and (
        isinstance(size_or_data, int) or float(size_or_data).is_integer()
    ):
        n = int(size_or_data)
        return allocate((n, n), target, storage=storage, max_in_ram_bytes=max_in_ram_bytes)

    # 2) Nested sequences: convert with the dtype rules above, then the array path.
    data = size_or_data
    if not isinstance(data, np.ndarray):
        if not all(isinstance(r, (list, tuple, np.ndarray)) for r in data):
            # NumPy's error for [[1], 2] (it was "TypeError: 'int' object is not iterable")
            raise ValueError("matrix(): the requested array has an inhomogeneous shape (rows of different kinds)")
        rows = [list(r) for r in data]
        if rows and len({len(r) for r in rows}) != 1:
            raise ValueError("matrix() requires rows of equal length")
        inferred = target or _infer_sequence_dtype([v for row in rows for v in row], np_module=np)
        np_dtype = _NUMPY_DTYPE.get(inferred, "complex64")
        data = np.array(rows, dtype=np_dtype).reshape(len(rows), len(rows[0]) if rows else 0)
        target = inferred
    elif data.ndim != 2:
        raise ValueError(f"matrix() expects 2-D input, got {data.ndim}-D (use vector() for 1-D)")
    return from_array(data, target, storage=storage, max_in_ram_bytes=max_in_ram_bytes)


def register_native_matrices(*classes: Any) -> None:
    """Install the shared __str__/__repr__ on the native matrix classes."""
    for cls in classes:
        if cls is not None:
            cls.__str__ = MatrixMixin.__str__
            cls.__repr__ = MatrixMixin.__repr__
