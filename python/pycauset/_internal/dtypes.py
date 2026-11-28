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
        # PyCauset's own names and aliases. NumPy's type codes ("i8" is int64, "f4" float32,
        # "<u2" ...) go to np.dtype below: short forms here like "i8" -> int8 contradicted
        # them (review 3.3 C-9).
        named = {
            "int": "int32", "uint": "uint32",  # PyCauset's default integer (F-103)
            "float": "float64", "half": "float16", "single": "float32", "double": "float64",
            "bool": "bool", "bool_": "bool", "bit": "bool",
            "complex": "complex_float64", "complex16": "complex_float16",
            "complex_float16": "complex_float16", "complex_float32": "complex_float32",
            "complex_float64": "complex_float64",
        }
        if s in named:
            return named[s]
        if s in _NUMPY_NAMES.values():
            return s
        dtype = dtype.strip()  # NumPy codes are case-sensitive ("f" vs "F")

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
    # None for what PyCauset does not store (longdouble, clongdouble, strings, objects): the
    # caller raises. longdouble was narrowed to float64 silently.
    return _NUMPY_NAMES.get(np_dtype.name)



def infer_sequence_dtype(values: list[Any], *, np_module: Any) -> str:
    """The dtype of a matrix or vector built from Python values (every element, flattened).

    NumPy decides when any element is a NumPy scalar (`np.array(values).dtype`: NumPy is the
    oracle; `np.complex64(1j)` lost its imaginary part and `np.int64` values were rounded
    through float64, review 3.3 C-5). Plain Python values follow the documented rules: bool,
    then Python ints -> int32 (F-103), any complex -> complex_float64, else float64. Empty
    input is float64, as in NumPy.
    """
    if not values:
        return "float64"
    if any(isinstance(v, np_module.generic) for v in values):
        inferred = normalize_dtype(np_module.array(values).dtype, np_module=np_module)
        if inferred is None:
            raise TypeError("matrix/vector elements must be numbers")
        return inferred
    if all(isinstance(v, bool) for v in values):
        return "bool"
    if any(isinstance(v, complex) for v in values):
        return "complex_float64"
    if all(isinstance(v, int) for v in values):
        return "int32"
    return "float64"
