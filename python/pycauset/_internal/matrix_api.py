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
from .dtypes import infer_sequence_dtype as _infer_sequence_dtype
from .dtypes import normalize_dtype as _normalize_dtype
from .formatting import MatrixMixin
from .native_threshold import temporary_native_memory_threshold

_np: Any | None = None
_native: Any | None = None

# Internal dtype name -> NumPy dtype name (complex_float16 has no NumPy equivalent).
_NUMPY_DTYPE = {
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "complex_float32": "complex64",
    "complex_float64": "complex128",
}


def configure(*, np_module: Any | None, native: Any) -> None:
    global _np, _native
    _np = np_module
    _native = native


def Matrix(size_or_data: Any, dtype: Any = None, **kwargs: Any) -> Any:
    """Build a dense matrix from a size, a NumPy array or nested sequences."""
    if _native is None:
        raise RuntimeError("pycauset matrix construction is not configured")
    import numpy as np

    native = _native
    target = _normalize_dtype(dtype, np_module=np)
    if dtype is not None and target is None:
        raise TypeError(f"unsupported dtype: {dtype!r}")
    max_in_ram_bytes = kwargs.pop("max_in_ram_bytes", None)

    # storage='disk' forces spill-to-disk (RAM-first is the default).
    storage = kwargs.pop("storage", None)
    if storage not in (None, "ram", "disk"):
        raise ValueError(
            f"Matrix storage must be 'ram' or 'disk', not {storage!r}. "
            "RAM-first is the default; 'disk' forces spill-to-disk."
        )
    force_disk = storage == "disk"

    # 1) A size: an n x n dense matrix of zeros.
    if isinstance(size_or_data, (int, float)) and not isinstance(size_or_data, bool) and (
        isinstance(size_or_data, int) or float(size_or_data).is_integer()
    ):
        if force_disk:
            raise TypeError(
                "storage='disk' requires NumPy-array input. Use set_memory_threshold(0) "
                "before allocation for typed construction, or save()/load() for explicit persistence."
            )
        return native.DenseMatrix(int(size_or_data), dtype=target or "float64", **kwargs)

    if "backing_file" in kwargs:
        raise TypeError(
            "matrix() does not accept 'backing_file' for data input. For persistent storage "
            "use save()/load(); for automatic spill-to-disk use set_memory_threshold()."
        )
    if kwargs:
        raise TypeError(f"matrix() got unexpected keyword arguments: {sorted(kwargs)}")

    # 2) Nested sequences: convert with the dtype rules above, then the array path.
    data = size_or_data
    if not isinstance(data, np.ndarray):
        rows = [list(r) for r in data]
        if rows and len({len(r) for r in rows}) != 1:
            raise ValueError("matrix() requires rows of equal length")
        inferred = target or _infer_sequence_dtype([v for row in rows for v in row], np_module=np)
        np_dtype = _NUMPY_DTYPE.get(inferred, "complex64")
        data = np.array(rows, dtype=np_dtype).reshape(len(rows), len(rows[0]) if rows else 0)
        target = inferred
    elif data.ndim != 2:
        raise ValueError(f"matrix() expects 2-D input, got {data.ndim}-D (use vector() for 1-D)")

    # 3) Convert to the requested dtype; complex_float16 has no NumPy dtype and is filled
    #    element-wise from complex64.
    if target == "complex_float16":
        out = native.DenseMatrix(int(data.shape[0]), int(data.shape[1]), dtype="complex_float16")
        for (i, j), z in np.ndenumerate(data.astype(np.complex64)):
            if z != 0:
                out.set(i, j, complex(z))
        return out
    if target is not None and data.dtype != np.dtype(_NUMPY_DTYPE[target]):
        data = data.astype(_NUMPY_DTYPE[target])
    data = np.ascontiguousarray(data)

    if force_disk:
        with temporary_native_memory_threshold(native, 0):
            return native.asarray(data)
    if max_in_ram_bytes is not None:
        est = export_guard.estimate_materialized_bytes(data)
        if est is not None and est > max_in_ram_bytes:
            with temporary_native_memory_threshold(native, int(max_in_ram_bytes)):
                return native.asarray(data)
    return native.asarray(data)


def register_native_matrices(*classes: Any) -> None:
    """Install the shared __str__/__repr__ on the native matrix classes."""
    for cls in classes:
        if cls is not None:
            cls.__str__ = MatrixMixin.__str__
            cls.__repr__ = MatrixMixin.__repr__
