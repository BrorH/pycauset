from __future__ import annotations

import math
from typing import Any

from . import dtypes as _dtypes
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


def estimate_materialized_bytes(obj: Any) -> int | None:
    blocks = getattr(obj, "_blocks", None)
    if isinstance(blocks, list) and getattr(obj, "dtype", "mixed") == "mixed":
        # A grid of leaves without dtypes (duck-typed matrices): the sum of its leaves.
        sizes = [estimate_materialized_bytes(b) for row in blocks for b in row]
        return None if any(s is None for s in sizes) else int(sum(sizes))
    shape = _safe_shape(obj)
    if shape is None:
        return None

    import numpy as np

    try:
        dtype = _dtypes.normalize_dtype(getattr(obj, "dtype", None), np_module=np)  # (the one parser)
    except Exception:
        return None
    size_per = _dtypes.itemsize(dtype)
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


def export_part(obj: Any) -> Any:
    """A part of an export already checked as a whole (a grid's leaf, an evaluated tile, a
    window): no size check and not counted as an export of its own."""
    fast_export = getattr(obj, "_to_numpy_fast", None)
    if callable(fast_export):
        return fast_export(allow_huge=True)
    if callable(getattr(obj, "_export_numpy", None)):
        return obj._export_numpy()
    import numpy as np  # type: ignore

    return np.asarray(obj)


def export_to_numpy(obj: Any, *, allow_huge: bool, dtype: Any = None, copy: bool = True) -> Any:
    """One copy into a new NumPy array, natively (every class, dtype and view: the engine's
    exporter, bind_matrix.cpp export_numpy). `dtype` converts afterwards. No fallback: an
    engine error is the answer. The Python classes (BlockMatrix, ThunkBlock, SubmatrixView)
    give their values through `_export_numpy()` after one size check for the whole object;
    their `__array__` comes here, so `allow_huge`, `copy` and `dtype` mean the same for every
    object (review 3.10 C-6: `to_numpy(product, allow_huge=True)` still raised, `copy=False`
    returned a copy, `dtype=` was not counted)."""
    global export_count
    export_count += 1
    import numpy as np  # type: ignore

    if copy is False:
        # As np.asarray(..., copy=False): every export is a new array (review 3.9 C-6; it
        # returned a writable copy silently, or NumPy's error with dtype=).
        raise ValueError("a PyCauset object cannot be exported without a copy; use copy=True")
    target = None
    if dtype is not None:
        # The constructors' parser, then the export map: to_numpy(dtype="int") was int64 while
        # zeros(dtype="int") is int32, and "complex16" raised (review 3.8 C-17).
        name = _dtypes.normalize_dtype(dtype, np_module=np)
        if name is None:
            raise TypeError(f"unsupported dtype: {dtype!r}")
        target = np.dtype(_dtypes.NUMPY_EXPORT[name])
    if target is not None and not allow_huge:
        # The converted array exists next to the exported one: both count (only the export did).
        shape = _safe_shape(obj)
        if shape is not None:
            converted = shape[0] * shape[1] * target.itemsize
            _check_size((estimate_materialized_bytes(obj) or 0) + converted, _EXPORT_MAX_BYTES)

    fast_export = getattr(obj, "_to_numpy_fast", None)
    if callable(fast_export):
        arr = fast_export(allow_huge=allow_huge)  # the engine applies the guard (numpy_export_guard.hpp)
    elif callable(getattr(obj, "_export_numpy", None)):
        # One check for the whole object, then its parts without their own (a grid's leaves
        # ignored allow_huge and were checked one by one against the ceiling).
        ensure_export_allowed(obj, allow_huge=allow_huge, ceiling_bytes=_EXPORT_MAX_BYTES)
        arr = obj._export_numpy()
    else:
        ensure_export_allowed(obj, allow_huge=allow_huge, ceiling_bytes=_EXPORT_MAX_BYTES)
        arr = np.asarray(obj)
    if target is not None:
        return arr.astype(target, copy=False)
    return arr
