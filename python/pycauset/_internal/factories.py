from __future__ import annotations

import warnings
from typing import Any, Callable

from .dtypes import normalize_dtype as _normalize_dtype


def vector_factory(
    size_or_data: Any,
    *,
    dtype: Any,
    np_module: Any | None,
    Float16Vector: Any | None,
    Float32Vector: Any | None,
    FloatVector: Any | None,
    ComplexFloat16Vector: Any | None,
    ComplexFloat32Vector: Any | None,
    ComplexFloat64Vector: Any | None,
    Int8Vector: Any | None,
    Int64Vector: Any | None,
    UInt8Vector: Any | None,
    UInt16Vector: Any | None,
    UInt32Vector: Any | None,
    UInt64Vector: Any | None,
    IntegerVector: Any | None,
    Int16Vector: Any | None,
    BitVector: Any | None,
    kwargs: dict[str, Any],
) -> Any:
    dtype_norm = _normalize_dtype(dtype, np_module=np_module)

    # 1) Creation by size
    if isinstance(size_or_data, (int, float)) and (
        isinstance(size_or_data, int) or size_or_data.is_integer()
    ):
        n = int(size_or_data)
        if dtype_norm == "int8" and Int8Vector:
            return Int8Vector(n, **kwargs)
        if dtype_norm == "int16" and Int16Vector:
            return Int16Vector(n, **kwargs)
        if dtype_norm == "int32" and IntegerVector:
            return IntegerVector(n, **kwargs)
        if dtype_norm == "int64" and Int64Vector:
            return Int64Vector(n, **kwargs)
        if dtype_norm == "uint8" and UInt8Vector:
            return UInt8Vector(n, **kwargs)
        if dtype_norm == "uint16" and UInt16Vector:
            return UInt16Vector(n, **kwargs)
        if dtype_norm == "uint32" and UInt32Vector:
            return UInt32Vector(n, **kwargs)
        if dtype_norm == "uint64" and UInt64Vector:
            return UInt64Vector(n, **kwargs)
        if dtype_norm == "bool" and BitVector:
            return BitVector(n, **kwargs)
        if dtype_norm == "float16" and Float16Vector:
            return Float16Vector(n, **kwargs)
        if dtype_norm == "float32" and Float32Vector:
            return Float32Vector(n, **kwargs)
        if dtype_norm == "float64" and FloatVector:
            return FloatVector(n, **kwargs)
        if dtype_norm == "complex_float16" and ComplexFloat16Vector:
            return ComplexFloat16Vector(n, **kwargs)
        if dtype_norm == "complex_float32" and ComplexFloat32Vector:
            return ComplexFloat32Vector(n, **kwargs)
        if dtype_norm == "complex_float64" and ComplexFloat64Vector:
            return ComplexFloat64Vector(n, **kwargs)
        raise ImportError("Vector classes not available in native extension.")

    data: Any = size_or_data

    # 2) Normalize NumPy
    if np_module is not None and isinstance(data, np_module.ndarray):
        if dtype_norm is None:
            dtype_norm = _normalize_dtype(data.dtype, np_module=np_module)
        data = data.tolist()

    # 3) Infer dtype
    if dtype_norm is None:
        dtype_norm = "float64"
        if any(isinstance(x, complex) for x in data):
            dtype_norm = "complex_float64"
        elif all(isinstance(x, (int, bool)) for x in data):
            dtype_norm = "int32"
        if all(isinstance(x, bool) for x in data):
            dtype_norm = "bool"

    n = len(data)

    if dtype_norm == "int8" and Int8Vector:
        vec = Int8Vector(n, **kwargs)
    elif dtype_norm == "int16" and Int16Vector:
        vec = Int16Vector(n, **kwargs)
    elif dtype_norm == "int32" and IntegerVector:
        vec = IntegerVector(n, **kwargs)
    elif dtype_norm == "int64" and Int64Vector:
        vec = Int64Vector(n, **kwargs)
    elif dtype_norm == "uint8" and UInt8Vector:
        vec = UInt8Vector(n, **kwargs)
    elif dtype_norm == "uint16" and UInt16Vector:
        vec = UInt16Vector(n, **kwargs)
    elif dtype_norm == "uint32" and UInt32Vector:
        vec = UInt32Vector(n, **kwargs)
    elif dtype_norm == "uint64" and UInt64Vector:
        vec = UInt64Vector(n, **kwargs)
    elif dtype_norm == "bool" and BitVector:
        vec = BitVector(n, **kwargs)
    elif dtype_norm == "float16" and Float16Vector:
        vec = Float16Vector(n, **kwargs)
    elif dtype_norm == "float32" and Float32Vector:
        vec = Float32Vector(n, **kwargs)
    elif dtype_norm == "float64" and FloatVector:
        vec = FloatVector(n, **kwargs)
    elif dtype_norm == "complex_float16" and ComplexFloat16Vector:
        vec = ComplexFloat16Vector(n, **kwargs)
    elif dtype_norm == "complex_float32" and ComplexFloat32Vector:
        vec = ComplexFloat32Vector(n, **kwargs)
    elif dtype_norm == "complex_float64" and ComplexFloat64Vector:
        vec = ComplexFloat64Vector(n, **kwargs)
    else:
        raise ImportError("Vector classes not available in native extension.")

    for i, val in enumerate(data):
        vec[i] = val

    return vec


def causal_matrix_factory(
    source: Any,
    *,
    populate: bool,
    np_module: Any | None,
    TriangularBitMatrix: Any,
    coerce_general_matrix: Callable[[Any], tuple[int, list[list[Any]]]],
    kwargs: dict[str, Any],
) -> Any:
    source_obj: Any = source

    # Case 1: integer size
    if isinstance(source_obj, (int, float)) and (
        isinstance(source_obj, int) or source_obj.is_integer()
    ):
        n = int(source_obj)
        if populate:
            return TriangularBitMatrix.random(n, p=0.5, **kwargs)
        return TriangularBitMatrix(n, **kwargs)

    # Case 2: numpy fast-path
    if np_module is not None and isinstance(source_obj, np_module.ndarray):
        if source_obj.dtype != bool:  # type: ignore[attr-defined]
            if not np_module.all(np_module.isin(source_obj, [0, 1])):
                warnings.warn(
                    "Input data contains non-binary values. They will be converted to boolean (True/False).",
                    UserWarning,
                    stacklevel=2,
                )

        if np_module.any(np_module.tril(source_obj) != 0):
            warnings.warn(
                "Input data contains non-zero values in the lower triangle or diagonal. "
                "CausalMatrix is strictly upper triangular; these values will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        try:
            if source_obj.dtype != bool:  # type: ignore[attr-defined]
                source_obj = source_obj.astype(bool)  # type: ignore[attr-defined]
            return TriangularBitMatrix(source_obj, **kwargs)
        except Exception:
            pass

    # Case 3: generic coercion
    size, rows = coerce_general_matrix(source_obj)
    matrix = TriangularBitMatrix(size, **kwargs)

    has_non_binary = False
    has_lower_triangular = False

    for i in range(size):
        for j in range(size):
            val = rows[i][j]

            if not has_non_binary:
                if val not in (0, 1, False, True, 0.0, 1.0):
                    has_non_binary = True

            if j <= i:
                if val:
                    has_lower_triangular = True
                continue

            if val:
                matrix.set(i, j, True)

    if has_non_binary:
        warnings.warn(
            "Input data contains non-binary values. They will be converted to boolean (True/False).",
            UserWarning,
            stacklevel=2,
        )

    if has_lower_triangular:
        warnings.warn(
            "Input data contains non-zero values in the lower triangle or diagonal. "
            "CausalMatrix is strictly upper triangular; these values will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    return matrix
