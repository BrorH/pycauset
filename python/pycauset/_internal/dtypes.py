from __future__ import annotations

from typing import Any


_NUMPY_NAMES = {
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "float16": "float16", "float32": "float32", "float64": "float64",
    "complex64": "complex_float32", "complex128": "complex_float64", "bool": "bool",
}


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

    # Abstract NumPy scalar types have no dtype of their own: np.dtype() rejects them.
    abstract = {
        np_module.integer: "int32",
        np_module.signedinteger: "int32",
        np_module.floating: "float64",
        np_module.complexfloating: "complex_float64",
    }
    if isinstance(dtype, type) and dtype in abstract:
        return abstract[dtype]
    try:
        np_dtype = np_module.dtype(dtype)
    except TypeError:
        return None  # not a dtype token
    name = _NUMPY_NAMES.get(np_dtype.name)
    if name is not None:
        return name
    # Wider kinds than PyCauset stores (longdouble, clongdouble): the widest float / complex.
    return {"f": "float64", "c": "complex_float64"}.get(np_dtype.kind)

