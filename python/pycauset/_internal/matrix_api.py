from __future__ import annotations

import abc
from typing import Any, Callable

from . import formatting as _formatting
from .dtypes import normalize_dtype as _normalize_dtype
from . import export_guard
from .native_threshold import temporary_native_memory_threshold


_np: Any | None = None
_warnings: Any | None = None
_native: Any | None = None
_track_matrix: Callable[[Any], None] | None = None
_coerce_general_matrix: Callable[[Any], tuple[int, list[list[Any]]]] | None = None

_IntegerMatrix: Any | None = None
_Int8Matrix: Any | None = None
_Int16Matrix: Any | None = None
_Int64Matrix: Any | None = None
_UInt8Matrix: Any | None = None
_UInt16Matrix: Any | None = None
_UInt32Matrix: Any | None = None
_UInt64Matrix: Any | None = None
_FloatMatrix: Any | None = None
_Float32Matrix: Any | None = None
_Float16Matrix: Any | None = None
_ComplexFloat16Matrix: Any | None = None
_ComplexFloat32Matrix: Any | None = None
_ComplexFloat64Matrix: Any | None = None
_TriangularFloatMatrix: Any | None = None
_TriangularIntegerMatrix: Any | None = None
_DenseBitMatrix: Any | None = None
_TriangularBitMatrix: Any | None = None


def configure(
    *,
    np_module: Any | None,
    warnings_module: Any,
    native: Any,
    track_matrix: Callable[[Any], None],
    coerce_general_matrix: Callable[[Any], tuple[int, list[list[Any]]]],
    IntegerMatrix: Any | None,
    Int8Matrix: Any | None,
    Int16Matrix: Any | None,
    Int64Matrix: Any | None,
    UInt8Matrix: Any | None,
    UInt16Matrix: Any | None,
    UInt32Matrix: Any | None,
    UInt64Matrix: Any | None,
    FloatMatrix: Any | None,
    Float16Matrix: Any | None,
    Float32Matrix: Any | None,
    ComplexFloat16Matrix: Any | None,
    ComplexFloat32Matrix: Any | None,
    ComplexFloat64Matrix: Any | None,
    TriangularFloatMatrix: Any | None,
    TriangularIntegerMatrix: Any | None,
    DenseBitMatrix: Any | None,
    TriangularBitMatrix: Any | None,
) -> None:
    global _np, _warnings, _native, _track_matrix, _coerce_general_matrix
    global _IntegerMatrix, _Int8Matrix, _Int16Matrix, _Int64Matrix
    global _UInt8Matrix, _UInt16Matrix, _UInt32Matrix, _UInt64Matrix
    global _FloatMatrix, _Float16Matrix, _Float32Matrix, _TriangularFloatMatrix
    global _TriangularIntegerMatrix, _DenseBitMatrix, _TriangularBitMatrix
    global _ComplexFloat16Matrix, _ComplexFloat32Matrix, _ComplexFloat64Matrix

    _np = np_module
    _warnings = warnings_module
    _native = native
    _track_matrix = track_matrix
    _coerce_general_matrix = coerce_general_matrix

    _IntegerMatrix = IntegerMatrix
    _Int8Matrix = Int8Matrix
    _Int16Matrix = Int16Matrix
    _Int64Matrix = Int64Matrix
    _UInt8Matrix = UInt8Matrix
    _UInt16Matrix = UInt16Matrix
    _UInt32Matrix = UInt32Matrix
    _UInt64Matrix = UInt64Matrix
    _FloatMatrix = FloatMatrix
    _Float16Matrix = Float16Matrix
    _Float32Matrix = Float32Matrix
    _ComplexFloat16Matrix = ComplexFloat16Matrix
    _ComplexFloat32Matrix = ComplexFloat32Matrix
    _ComplexFloat64Matrix = ComplexFloat64Matrix
    _TriangularFloatMatrix = TriangularFloatMatrix
    _TriangularIntegerMatrix = TriangularIntegerMatrix
    _DenseBitMatrix = DenseBitMatrix
    _TriangularBitMatrix = TriangularBitMatrix


MatrixMixin = _formatting.MatrixMixin


class Matrix(MatrixMixin, metaclass=abc.ABCMeta):
    """Matrix factory shim.

    Actual implementation is configured at import-time by the package facade.
    """

    def __new__(cls, size_or_data: Any, dtype: Any = None, **kwargs: Any):
        if cls is not Matrix:
            return super().__new__(cls)

        if _warnings is None or _native is None or _coerce_general_matrix is None:
            raise RuntimeError("pycauset Matrix API not configured")

        warnings_module = _warnings
        native = _native
        coerce_general_matrix = _coerce_general_matrix

        target_dtype = _normalize_dtype(dtype, np_module=_np)

        max_in_ram_bytes = kwargs.pop("max_in_ram_bytes", None)

        if isinstance(size_or_data, (int, float)) and (
            isinstance(size_or_data, int) or size_or_data.is_integer()
        ):
            n = int(size_or_data)
            if target_dtype == "complex_float16":
                if _ComplexFloat16Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _ComplexFloat16Matrix(n, **kwargs)
            if target_dtype == "complex_float32":
                if _ComplexFloat32Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _ComplexFloat32Matrix(n, **kwargs)
            if target_dtype == "complex_float64":
                if _ComplexFloat64Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _ComplexFloat64Matrix(n, **kwargs)
            if target_dtype == "int8":
                if _Int8Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _Int8Matrix(n, **kwargs)
            if target_dtype == "int16":
                if _Int16Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _Int16Matrix(n, **kwargs)
            if target_dtype == "int32":
                if _IntegerMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _IntegerMatrix(n, **kwargs)
            if target_dtype == "int64":
                if _Int64Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _Int64Matrix(n, **kwargs)
            if target_dtype == "uint8":
                if _UInt8Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _UInt8Matrix(n, **kwargs)
            if target_dtype == "uint16":
                if _UInt16Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _UInt16Matrix(n, **kwargs)
            if target_dtype == "uint32":
                if _UInt32Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _UInt32Matrix(n, **kwargs)
            if target_dtype == "uint64":
                if _UInt64Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _UInt64Matrix(n, **kwargs)
            if target_dtype == "bool":
                if _DenseBitMatrix is not None:
                    return _DenseBitMatrix(n, **kwargs)
                if _IntegerMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _IntegerMatrix(n, **kwargs)
            if target_dtype == "float16":
                if _Float16Matrix is not None:
                    return _Float16Matrix(n, **kwargs)
                if _Float32Matrix is not None:
                    return _Float32Matrix(n, **kwargs)
                if _FloatMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _FloatMatrix(n, **kwargs)
            if target_dtype == "float32":
                if _Float32Matrix is not None:
                    return _Float32Matrix(n, **kwargs)
                if _FloatMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _FloatMatrix(n, **kwargs)

            if target_dtype == "float64":
                if _FloatMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _FloatMatrix(n, **kwargs)

            force = kwargs.pop("force_precision", None)

            if force == "double" or force == "float64":
                if _FloatMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _FloatMatrix(n, **kwargs)
            if force == "single" or force == "float32":
                if _Float32Matrix is not None:
                    return _Float32Matrix(n, **kwargs)

            if n >= 10000 and _Float32Matrix is not None:
                warnings_module.warn(
                    f"Matrix size {n} >= 10,000. Enforcing Float32 precision for storage efficiency. "
                    "Use force_precision='double' to override."
                )
                return _Float32Matrix(n, **kwargs)

            if _FloatMatrix is None:
                return super(Matrix, cls).__new__(cls)
            return _FloatMatrix(n, **kwargs)

        data: Any = size_or_data

        try:
            import numpy as np

            if isinstance(data, np.ndarray):
                if data.dtype in (
                    getattr(np, "int8", object()),
                    np.int16,
                    np.int32,
                    np.int64,
                    getattr(np, "uint8", object()),
                    getattr(np, "uint16", object()),
                    getattr(np, "uint32", object()),
                    getattr(np, "uint64", object()),
                    getattr(np, "bool_", object()),
                    getattr(np, "float16", object()),
                    np.float32,
                    np.float64,
                    getattr(np, "complex64", object()),
                    getattr(np, "complex128", object()),
                ) and hasattr(native, "asarray"):
                    if max_in_ram_bytes is not None:
                        est = export_guard.estimate_materialized_bytes(data)
                        if est is not None and est > max_in_ram_bytes:
                            # Force native allocation policy for this import (best-effort).
                            with temporary_native_memory_threshold(native, int(max_in_ram_bytes)):
                                return native.asarray(data)
                    return native.asarray(data)

                if dtype is None:
                    inferred = _normalize_dtype(data.dtype, np_module=np)
                    if inferred is not None:
                        target_dtype = inferred
                data = data.tolist()
        except ImportError:
            pass

        try:
            is_integer = True
            is_triangular = True
            rows = len(data)

            if rows == 0:
                if _FloatMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                return _FloatMatrix(0, **kwargs)

            cols: int | None = None
            for row in data:
                if cols is None:
                    cols = len(row)
                elif len(row) != cols:
                    return super(Matrix, cls).__new__(cls)

            if cols is None:
                return super(Matrix, cls).__new__(cls)

            is_square = rows == cols
            if not is_square:
                is_triangular = False

            for i, row in enumerate(data):
                for j, val in enumerate(row):
                    if target_dtype is None:
                        if is_integer and not isinstance(val, (int, bool)):
                            is_integer = False

                    if is_triangular and is_square and j <= i and val != 0:
                        is_triangular = False

            def create_and_fill(cls_type: Any, data_source: Any):
                size, rows2 = coerce_general_matrix(data_source)
                mat = cls_type(size, **kwargs)
                for i in range(size):
                    for j in range(size):
                        val = rows2[i][j]
                        if val != 0:
                            mat.set(i, j, val)
                return mat

            def create_and_fill_rectangular(cls_type: Any, data_source: Any, *, r: int, c: int):
                mat = cls_type(r, c, **kwargs)
                for i in range(r):
                    row2 = data_source[i]
                    for j in range(c):
                        val = row2[j]
                        if val != 0:
                            mat.set(i, j, val)
                return mat

            if target_dtype == "int8":
                if _Int8Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_Int8Matrix, data, r=rows, c=cols)
                return create_and_fill(_Int8Matrix, data)

            if target_dtype == "int32":
                if is_triangular and _TriangularIntegerMatrix is not None:
                    return create_and_fill(_TriangularIntegerMatrix, data)
                if _IntegerMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_IntegerMatrix, data, r=rows, c=cols)
                return create_and_fill(_IntegerMatrix, data)

            if target_dtype == "int64":
                if _Int64Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_Int64Matrix, data, r=rows, c=cols)
                return create_and_fill(_Int64Matrix, data)

            if target_dtype == "uint8":
                if _UInt8Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_UInt8Matrix, data, r=rows, c=cols)
                return create_and_fill(_UInt8Matrix, data)

            if target_dtype == "uint16":
                if _UInt16Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_UInt16Matrix, data, r=rows, c=cols)
                return create_and_fill(_UInt16Matrix, data)

            if target_dtype == "uint32":
                if _UInt32Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_UInt32Matrix, data, r=rows, c=cols)
                return create_and_fill(_UInt32Matrix, data)

            if target_dtype == "uint64":
                if _UInt64Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_UInt64Matrix, data, r=rows, c=cols)
                return create_and_fill(_UInt64Matrix, data)

            if target_dtype == "int16":
                if _Int16Matrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_Int16Matrix, data, r=rows, c=cols)
                return create_and_fill(_Int16Matrix, data)

            if target_dtype == "bool":
                if is_triangular and _TriangularBitMatrix is not None:
                    return create_and_fill(_TriangularBitMatrix, data)
                if _DenseBitMatrix is not None:
                    if not is_square:
                        return create_and_fill_rectangular(_DenseBitMatrix, data, r=rows, c=cols)
                    return create_and_fill(_DenseBitMatrix, data)
                if _IntegerMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_IntegerMatrix, data, r=rows, c=cols)
                return create_and_fill(_IntegerMatrix, data)

            if target_dtype == "float32":
                if _Float32Matrix is not None:
                    if not is_square:
                        return create_and_fill_rectangular(_Float32Matrix, data, r=rows, c=cols)
                    return create_and_fill(_Float32Matrix, data)
                if _FloatMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_FloatMatrix, data, r=rows, c=cols)
                return create_and_fill(_FloatMatrix, data)

            if target_dtype == "float64":
                if is_triangular and _TriangularFloatMatrix is not None:
                    return create_and_fill(_TriangularFloatMatrix, data)
                if _FloatMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_FloatMatrix, data, r=rows, c=cols)
                return create_and_fill(_FloatMatrix, data)

            if is_integer:
                if is_triangular and _TriangularIntegerMatrix is not None:
                    return create_and_fill(_TriangularIntegerMatrix, data)
                if _IntegerMatrix is None:
                    return super(Matrix, cls).__new__(cls)
                if not is_square:
                    return create_and_fill_rectangular(_IntegerMatrix, data, r=rows, c=cols)
                return create_and_fill(_IntegerMatrix, data)

            if is_triangular and _TriangularFloatMatrix is not None:
                return create_and_fill(_TriangularFloatMatrix, data)
            if _FloatMatrix is None:
                return super(Matrix, cls).__new__(cls)
            if not is_square:
                return create_and_fill_rectangular(_FloatMatrix, data, r=rows, c=cols)
            return create_and_fill(_FloatMatrix, data)

        except Exception:
            return super(Matrix, cls).__new__(cls)

    def __init__(self, size_or_data: Any):
        if _warnings is None or _track_matrix is None or _coerce_general_matrix is None:
            raise RuntimeError("pycauset Matrix API not configured")

        if isinstance(size_or_data, float):
            size_or_data = int(size_or_data)

        if isinstance(size_or_data, int):
            if size_or_data < 0:
                raise ValueError("Matrix dimension must be non-negative.")
            self._size = size_or_data
            self._data: list[list[Any]] = [
                [0 for _ in range(size_or_data)] for _ in range(size_or_data)
            ]
        else:
            size, rows = _coerce_general_matrix(size_or_data)
            self._size = size
            self._data = rows

        _track_matrix(self)

    def size(self) -> int:
        return self._size

    @property
    def shape(self) -> tuple[int, int]:
        return (self._size, self._size)

    def __len__(self) -> int:
        return self._size

    def _validate_indices(self, i: int, j: int) -> None:
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError("Matrix indices must be integers.")
        if i < 0 or j < 0 or i >= self._size or j >= self._size:
            raise IndexError("Matrix indices out of range.")

    def get(self, i: int, j: int) -> Any:
        self._validate_indices(i, j)
        return self._data[i][j]

    def set(self, i: int, j: int, value: Any) -> None:
        self._validate_indices(i, j)
        self._data[i][j] = value

    def __getitem__(self, key: Any) -> Any:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("matrix indices must be provided as [row, col].")
        i, j = key
        return self.get(i, j)

    def __setitem__(self, key: Any, value: Any) -> None:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("matrix indices must be provided as [row, col].")
        i, j = key
        self.set(i, j, value)

    def close(self) -> None:
        self._data = []
        self._size = 0


class TriangularMatrix(Matrix):
    """Base class for triangular matrices."""

    pass


def _supports_matrix_protocol(candidate: Any) -> bool:
    return callable(getattr(candidate, "size", None)) and callable(getattr(candidate, "get", None))


def register_native_matrices(
    *,
    TriangularBitMatrix: Any | None,
    IntegerMatrix: Any | None,
    Int8Matrix: Any | None,
    Int16Matrix: Any | None,
    Int64Matrix: Any | None,
    UInt8Matrix: Any | None,
    UInt16Matrix: Any | None,
    UInt32Matrix: Any | None,
    UInt64Matrix: Any | None,
    FloatMatrix: Any | None,
    Float16Matrix: Any | None,
    Float32Matrix: Any | None,
    ComplexFloat16Matrix: Any | None,
    ComplexFloat32Matrix: Any | None,
    ComplexFloat64Matrix: Any | None,
    TriangularFloatMatrix: Any | None,
    TriangularIntegerMatrix: Any | None,
    DenseBitMatrix: Any | None,
) -> None:
    for cls in (
        TriangularBitMatrix,
        IntegerMatrix,
        Int8Matrix,
        Int16Matrix,
        Int64Matrix,
        UInt8Matrix,
        UInt16Matrix,
        UInt32Matrix,
        UInt64Matrix,
        FloatMatrix,
        Float16Matrix,
        Float32Matrix,
        ComplexFloat16Matrix,
        ComplexFloat32Matrix,
        ComplexFloat64Matrix,
        TriangularFloatMatrix,
    ):
        if cls and _supports_matrix_protocol(cls):
            cls.__str__ = MatrixMixin.__str__
            cls.__repr__ = MatrixMixin.__repr__
            Matrix.register(cls)

    if TriangularIntegerMatrix and _supports_matrix_protocol(TriangularIntegerMatrix):
        TriangularIntegerMatrix.__str__ = MatrixMixin.__str__
        TriangularIntegerMatrix.__repr__ = MatrixMixin.__repr__
        Matrix.register(TriangularIntegerMatrix)

    if DenseBitMatrix and _supports_matrix_protocol(DenseBitMatrix):
        DenseBitMatrix.__str__ = MatrixMixin.__str__
        DenseBitMatrix.__repr__ = MatrixMixin.__repr__
        Matrix.register(DenseBitMatrix)

    if TriangularBitMatrix:
        TriangularMatrix.register(TriangularBitMatrix)
    if TriangularFloatMatrix:
        TriangularMatrix.register(TriangularFloatMatrix)
    if TriangularIntegerMatrix:
        TriangularMatrix.register(TriangularIntegerMatrix)
