from __future__ import annotations

import math
from typing import Any


_EXPORT_MAX_BYTES: int | None = None
_DEFAULT_FILEBACKED_LIMIT_BYTES = 128 * 1024 * 1024


def set_max_bytes(limit: int | None) -> None:
    """Set global export ceiling in bytes (None disables size check)."""
    global _EXPORT_MAX_BYTES
    _EXPORT_MAX_BYTES = limit


def get_max_bytes() -> int | None:
    return _EXPORT_MAX_BYTES


def _safe_shape(obj: Any) -> tuple[int, int] | None:
    try:
        return int(obj.rows()), int(obj.cols())
    except Exception:
        pass
    try:
        shape = getattr(obj, "shape", None)
        if isinstance(shape, tuple) and len(shape) == 2:
            return int(shape[0]), int(shape[1])
        if isinstance(shape, tuple) and len(shape) == 1:
            return int(shape[0]), 1
    except Exception:
        pass
    try:
        n = len(obj)
        return int(n), 1
    except Exception:
        pass
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
    try:
        s = str(token)
    except Exception:
        return None
    s = s.lower()
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
        # Triangular float matrices are stored as float64.
        "triangularfloat": "float64",
        "int_": "int32",
        "int": "int32",
        "uint": "uint32",
        "densebit": "bit",
        "triangularbit": "bit",
        "integer": "int32",
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

    target = mapping.get(token, token)
    try:
        return np_module.dtype(target)
    except Exception:
        try:
            return np_module.dtype(str(target))
        except Exception:
            return target


def _infer_numpy_dtype(obj: Any, requested: Any, np_module: Any) -> Any:
    if requested is not None:
        return _resolve_numpy_dtype(requested, np_module)

    token = _normalize_dtype_token(getattr(obj, "dtype", None))
    if token is None:
        name = type(obj).__name__
        token = _normalize_dtype_token(name.replace("Matrix", "").replace("Vector", ""))
    token = "bool" if token == "bit" else token
    return _resolve_numpy_dtype(token, np_module)


def estimate_materialized_bytes(obj: Any) -> int | None:
    shape = _safe_shape(obj)
    if shape is None:
        return None

    dtype = _normalize_dtype_token(getattr(obj, "dtype", None))
    if dtype is None:
        dtype = _normalize_dtype_token(type(obj).__name__.replace("Matrix", "").replace("Vector", ""))

    size_per = _DTYPE_SIZE_BYTES.get(dtype)
    if size_per is None:
        return None

    rows, cols = shape
    total_elems = rows * cols
    total_bytes = total_elems * size_per
    return int(math.ceil(total_bytes))


def backing_kind(obj: Any) -> tuple[str | None, str | None]:
    for attr in ("get_backing_file", "backing_file"):
        try:
            val = getattr(obj, attr)
            path = val() if callable(val) else val
            if path:
                path_str = str(path)
                # Explicit in-memory sentinel used by native matrices.
                if path_str == ":memory:":
                    return path_str, None
                if path_str.endswith(".pycauset"):
                    return path_str, "snapshot"
                if path_str.endswith(".tmp") or path_str.endswith(".raw_tmp"):
                    return path_str, "temp"
                return path_str, "unknown"
        except Exception:
            continue
    return None, None


def ensure_export_allowed(obj: Any, *, allow_huge: bool, ceiling_bytes: int | None) -> None:
    if allow_huge:
        return

    _, kind = backing_kind(obj)
    est = estimate_materialized_bytes(obj)

    if kind and kind != "snapshot":
        if not allow_huge:
            raise RuntimeError(
                "Export to NumPy is blocked for file-backed/out-of-core objects; pass allow_huge=True via pycauset.to_numpy(...) to override."
            )
        limit = ceiling_bytes if ceiling_bytes is not None else _DEFAULT_FILEBACKED_LIMIT_BYTES
        if limit is not None and est is not None and est > limit:
            raise RuntimeError(
                "Export to NumPy exceeds configured materialization limit; lower the size or raise the ceiling."
            )

    if ceiling_bytes is None:
        return

    if est is not None and est > ceiling_bytes:
        raise RuntimeError(
            "Export to NumPy exceeds configured materialization limit; pass allow_huge=True via pycauset.to_numpy(...) to override."
        )


def _materialize_via_memoryview(obj: Any, dtype: Any, np_module: Any, *, copy: bool) -> Any:
    try:
        mv = memoryview(obj)
        resolved = _resolve_numpy_dtype(dtype, np_module)
        if copy:
            return np_module.array(mv, copy=True, dtype=resolved)
        # Prefer a zero-copy view when possible.
        return np_module.asarray(mv, dtype=resolved)
    except Exception:
        return None


def export_to_numpy(obj: Any, *, allow_huge: bool, dtype: Any = None, copy: bool = True) -> Any:
    ensure_export_allowed(obj, allow_huge=allow_huge, ceiling_bytes=_EXPORT_MAX_BYTES)

    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover - numpy optional at runtime
        raise RuntimeError("NumPy is required for export") from e

    # Try fast native export if available
    fast_export = getattr(obj, "_to_numpy_fast", None)
    if callable(fast_export):
        try:
            arr = fast_export(allow_huge=allow_huge)
            if arr is not None:
                if dtype is not None:
                    return np.array(arr, dtype=dtype, copy=copy)
                return arr
        except Exception:
            pass

    target_dtype = _infer_numpy_dtype(obj, dtype, np)

    # When allow_huge=True, the caller explicitly wants to bypass the safety policy.
    # Avoid the memoryview/buffer fast-path because native buffer exports cannot see
    # the allow_huge flag and may enforce the global ceiling.
    if not allow_huge:
        arr = _materialize_via_memoryview(obj, target_dtype, np, copy=copy)
        if arr is not None:
            return arr

    shape = _safe_shape(obj)
    if shape is None:
        # Fallback to NumPy's own coercion (may be slower but avoids infinite recursion).
        return np.array(obj, dtype=dtype, copy=copy)

    rows, cols = shape
    get_fn = getattr(obj, "get", None)

    is_vector_like = False
    if callable(get_fn) and not hasattr(obj, "cols"):
        # Vector types may present a 2D shape when transposed (e.g. (1, N)), but their
        # get(...) accessor still accepts a single index. Detect that and export correctly.
        if rows > 0 and cols > 0:
            try:
                get_fn(0, 0)
            except TypeError:
                is_vector_like = True
            except Exception:
                # Any non-TypeError indicates the signature accepts (i, j).
                is_vector_like = False
        else:
            is_vector_like = True

    if callable(get_fn) and is_vector_like:
        if cols == 1:
            out = np.empty((rows,), dtype=target_dtype)
            for i in range(rows):
                out[i] = get_fn(i)
            return out
        if rows == 1:
            out = np.empty((1, cols), dtype=target_dtype)
            for j in range(cols):
                out[0, j] = get_fn(j)
            return out

    is_vector = cols == 1 and not hasattr(obj, "cols")
    out_shape = (rows,) if is_vector else (rows, cols)
    out = np.empty(out_shape, dtype=target_dtype)

    if callable(get_fn):
        for i in range(rows):
            if is_vector:
                out[i] = get_fn(i)
            else:
                for j in range(cols):
                    out[i, j] = get_fn(i, j)
        return out

    if is_vector:
        for i in range(rows):
            out[i] = obj[i]
        return out

    # Last resort
    return np.array(obj, dtype=target_dtype, copy=copy)
