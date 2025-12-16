from __future__ import annotations

from typing import Any


def normalize_dtype(dtype: Any, *, np_module: Any | None) -> str | None:
    """Normalize user-provided dtype tokens into internal strings.

    Returns one of:
        {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
         "float16", "float32", "float64", "bool",
         "complex_float16", "complex_float32", "complex_float64"}
    or None.

    Accepted inputs include:
    - Case-insensitive strings: "int16", "INT16", "f32", "bool_", "bit", ...
    - Python builtins: int, float, bool
    - NumPy dtypes/scalars: np.int16, np.dtype("int16"), np.float32, ...
    """

    if dtype is None:
        return None

    if dtype is int:
        return "int32"
    if dtype is float:
        return "float64"
    if dtype is bool:
        return "bool"

    if isinstance(dtype, str):
        s = dtype.strip().lower()
        if s in ("int8", "i8"):
            return "int8"
        if s in ("int16", "i16"):
            return "int16"
        if s in ("int32", "i32", "int"):
            return "int32"
        if s in ("int64", "i64"):
            return "int64"
        if s in ("uint8", "u8"):
            return "uint8"
        if s in ("uint16", "u16"):
            return "uint16"
        if s in ("uint32", "u32", "uint"):
            return "uint32"
        if s in ("uint64", "u64"):
            return "uint64"
        if s in ("float16", "f16", "half"):
            return "float16"
        if s in ("float32", "f32", "single"):
            return "float32"
        if s in ("float", "float64", "f64", "double"):
            return "float64"
        if s in ("bool", "bool_", "bit"):
            return "bool"
        if s in ("complex_float16", "complex16"):
            return "complex_float16"
        if s in ("complex_float32", "complex64"):
            return "complex_float32"
        if s in ("complex_float64", "complex128", "complex"):
            return "complex_float64"
        return None

    if np_module is None:
        return None

    # NumPy dtype/scalar types
    try:
        np_dtype = np_module.dtype(dtype)
    except Exception:
        np_dtype = None

    if np_dtype is not None:
        try:
            if np_dtype == np_module.dtype("int8"):
                return "int8"
            if np_dtype == np_module.dtype("int16"):
                return "int16"
            if np_dtype == np_module.dtype("int32"):
                return "int32"
            if np_dtype == np_module.dtype("int64"):
                return "int64"
            if np_dtype == np_module.dtype("uint8"):
                return "uint8"
            if np_dtype == np_module.dtype("uint16"):
                return "uint16"
            if np_dtype == np_module.dtype("uint32"):
                return "uint32"
            if np_dtype == np_module.dtype("uint64"):
                return "uint64"
        except Exception:
            pass

        # Any other integer kind: default to int32 unless explicitly sized.
        if getattr(np_dtype, "kind", None) in ("i", "u"):
            return "int32"

        # Exact float32
        try:
            if np_dtype == np_module.dtype("float32"):
                return "float32"
        except Exception:
            pass

        # Exact float16
        try:
            if np_dtype == np_module.dtype("float16"):
                return "float16"
        except Exception:
            pass

        # Any other float kind -> float64-backed FloatMatrix/Vector
        if getattr(np_dtype, "kind", None) == "f":
            return "float64"

        if getattr(np_dtype, "kind", None) == "c":
            # complex64/complex128
            try:
                if np_dtype == np_module.dtype("complex64"):
                    return "complex_float32"
            except Exception:
                pass
            return "complex_float64"

        if getattr(np_dtype, "kind", None) == "b":
            return "bool"

    # Handle scalar type objects like np.integer/np.floating
    try:
        if dtype is getattr(np_module, "int8", object()):
            return "int8"
        if dtype is np_module.int16:
            return "int16"
        if dtype is np_module.int32:
            return "int32"
        if dtype is np_module.int64:
            return "int64"
        if dtype is getattr(np_module, "uint8", object()):
            return "uint8"
        if dtype is getattr(np_module, "uint16", object()):
            return "uint16"
        if dtype is getattr(np_module, "uint32", object()):
            return "uint32"
        if dtype is getattr(np_module, "uint64", object()):
            return "uint64"
        if dtype is np_module.integer:
            return "int32"
        if dtype is getattr(np_module, "float16", None):
            return "float16"
        if dtype is np_module.float32:
            return "float32"
        if dtype in (np_module.float64, np_module.floating):
            return "float64"
        if dtype in (getattr(np_module, "complex64", object()),):
            return "complex_float32"
        if dtype in (
            getattr(np_module, "complex128", object()),
            getattr(np_module, "complexfloating", object()),
        ):
            return "complex_float64"
        if dtype in (np_module.bool_, getattr(np_module, "bool", np_module.bool_)):
            return "bool"
    except Exception:
        pass

    return None
