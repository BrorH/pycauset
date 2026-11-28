from __future__ import annotations

import math
from typing import Any

from .coercion import safe_rows_cols

_EXPORT_MAX_BYTES: int | None = None


def set_max_bytes(limit: int | None) -> None:
    """Set global export ceiling in bytes (None disables size check)."""
    global _EXPORT_MAX_BYTES
    _EXPORT_MAX_BYTES = limit


def get_max_bytes() -> int | None:
    return _EXPORT_MAX_BYTES


def _safe_shape(obj: Any) -> tuple[int, int] | None:
    shape = safe_rows_cols(obj)
    if shape is not None:
        return shape
    shape = getattr(obj, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 1:
        return int(shape[0]), 1
    return None


_DTYPE_SIZE_BYTES = {
    # NumPy materializes bit matrices as bool arrays.
    "bit": 1,
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    # NumPy lacks a true complex16; we promote to complex64 (8 bytes).
    "complex_float16": 8,
    "complex_float32": 8,
    "complex_float64": 16,
}


def _normalize_dtype_token(token: Any) -> str | None:
    if token is None:
        return None
    s = str(token).lower()
    for suffix in ("matrix", "vector"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    alias = {
        "bool_": "bool",
        "complex64": "complex_float32",
        "complex128": "complex_float64",
        "float_": "float64",
        "float": "float64",
        "int_": "int32",
        "int": "int32",
        "uint": "uint32",
        "complexfloat16": "complex_float16",
        "complexfloat32": "complex_float32",
        "complexfloat64": "complex_float64",
    }
    return alias.get(s, s)


def _resolve_numpy_dtype(token: Any, np_module: Any) -> Any:
    if token is None:
        return None

    mapping = {
        "bit": "bool",
        "bool": "bool",
        "bool_": "bool",
        "integer": "int32",
        # NumPy lacks a true complex16; promote to complex64 for safety.
        "complex_float16": "complex64",
        "complexfloat16": "complex64",
        "complex_float32": "complex64",
        "complexfloat32": "complex64",
        "complex_float64": "complex128",
        "complexfloat64": "complex128",
    }

    return np_module.dtype(mapping.get(token, token))


def estimate_materialized_bytes(obj: Any) -> int | None:
    blocks = getattr(obj, "_blocks", None)
    if isinstance(blocks, list):  # a BlockMatrix: its leaves (it has no single dtype)
        sizes = [estimate_materialized_bytes(b) for row in blocks for b in row]
        return None if any(s is None for s in sizes) else int(sum(sizes))
    shape = _safe_shape(obj)
    if shape is None:
        return None

    dtype = _normalize_dtype_token(getattr(obj, "dtype", None))

    size_per = _DTYPE_SIZE_BYTES.get(dtype)
    if size_per is None:
        return None

    rows, cols = shape
    total_elems = rows * cols
    total_bytes = total_elems * size_per
    return int(math.ceil(total_bytes))


def ensure_export_allowed(obj: Any, *, allow_huge: bool, ceiling_bytes: int | None) -> None:
    if allow_huge:
        return

    # By size only (F-016), as the engine's guard (numpy_export_guard.hpp): RAM or file backing
    # does not matter. With a ceiling, above it raises; without one, not fitting in free RAM.
    est = estimate_materialized_bytes(obj)
    if est is None:
        return
    _check_size(est, ceiling_bytes)


def _check_size(est: int, ceiling_bytes: int | None) -> None:
    hint = "; pass allow_huge=True via pycauset.to_numpy(...) to override."
    if ceiling_bytes is not None:
        if est > ceiling_bytes:
            raise RuntimeError(f"Export to NumPy ({est} bytes) exceeds the set_export_max_bytes ceiling{hint}")
        return
    from .. import _pycauset  # loaded with the package

    if not _pycauset.MemoryGovernor.instance().can_fit_in_ram(est):
        raise RuntimeError(f"Export to NumPy ({est} bytes) would not fit in free RAM{hint}")


# Number of exports since import. The operation atlas (tests/atlas) reads it around an
# operation: an operation that exports its input to NumPy is a "numpy-fallback" path.
export_count = 0


def export_to_numpy(obj: Any, *, allow_huge: bool, dtype: Any = None, copy: bool = True) -> Any:
    """One copy into a new NumPy array, natively (every class, dtype and view: the engine's
    exporter, bind_matrix.cpp export_numpy). `dtype` converts afterwards. No fallback: an
    engine error is the answer. Objects that are not engine matrices/vectors (BlockMatrix,
    submatrix views) export through their own __array__."""
    global export_count
    export_count += 1
    import numpy as np  # type: ignore

    if copy is False:
        # As np.asarray(..., copy=False): every export is a new array (review 3.9 C-6; it
        # returned a writable copy silently, or NumPy's error with dtype=).
        raise ValueError("a PyCauset object cannot be exported without a copy; use copy=True")
    target = _resolve_numpy_dtype(dtype, np) if dtype is not None else None
    if target is not None and not allow_huge:
        # The converted array exists next to the exported one: both count (only the export did).
        shape = _safe_shape(obj)
        if shape is not None:
            converted = shape[0] * shape[1] * target.itemsize
            _check_size((estimate_materialized_bytes(obj) or 0) + converted, _EXPORT_MAX_BYTES)

    fast_export = getattr(obj, "_to_numpy_fast", None)
    if callable(fast_export):
        arr = fast_export(allow_huge=allow_huge)  # the engine applies the guard (numpy_export_guard.hpp)
    elif isinstance(getattr(obj, "_blocks", None), list):
        # A BlockMatrix: one check for the whole matrix, then its leaves without their own
        # (they ignored allow_huge and were checked one by one against the ceiling).
        ensure_export_allowed(obj, allow_huge=allow_huge, ceiling_bytes=_EXPORT_MAX_BYTES)
        arr = np.block([[export_to_numpy(b, allow_huge=True) for b in row] for row in obj._blocks])
        if obj.dtype != "mixed":
            arr = arr.astype(obj._numpy_dtype(), copy=False)  # the grid's (engine) dtype
    else:
        ensure_export_allowed(obj, allow_huge=allow_huge, ceiling_bytes=_EXPORT_MAX_BYTES)
        arr = np.asarray(obj)
    if target is not None:
        return arr.astype(target, copy=False)
    return arr
