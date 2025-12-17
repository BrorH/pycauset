"""Python package wrapper that augments the native pycauset extension."""
from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import os
import warnings
from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Any
from importlib import import_module as _import_module
from ._internal import persistence as _persistence
from ._internal import linalg_cache as _linalg_cache
from ._internal import runtime as _runtime_mod
from ._internal import formatting as _formatting
from ._internal import patching as _patching
from ._internal import coercion as _coercion
from ._internal import factories as _factories
from ._internal.dtypes import normalize_dtype as _normalize_dtype
from ._internal import ops as _ops
from ._internal import native as _native_mod
from ._internal import matrix_api as _matrix_api
from ._internal.warnings import (
    PyCausetWarning,
    PyCausetDTypeWarning,
    PyCausetOverflowRiskWarning,
    PyCausetPerformanceWarning,
)

try:  # NumPy is optional at runtime
    import numpy as _np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    _np = None

# Public dtype tokens (NumPy-like). These are simple sentinels accepted by
# Matrix/Vector factories and native interop.
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
int_ = "int32"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
uint64 = "uint64"
uint = "uint32"
float16 = "float16"
float32 = "float32"
float64 = "float64"
float_ = "float64"
bool_ = "bool"
bit = "bit"

# Complex float dtype tokens (Phase 3)
complex_float16 = "complex_float16"
complex_float32 = "complex_float32"
complex_float64 = "complex_float64"
complex64 = "complex_float32"
complex128 = "complex_float64"

_native_mod.configure_windows_dll_search_paths(package_dir=os.path.dirname(__file__))
_native = _native_mod.import_native_extension(package=__name__)

# Import modules that depend on the native extension after it is loaded.
from ._storage import cleanup_storage, set_temporary_file
from .causet import CausalSet
from . import spacetime
from . import field


def _debug_resolve_promotion(op: str, a_dtype: str, b_dtype: str) -> dict[str, object]:
    """Internal/test helper: return promotion resolver decision without running kernels."""
    fn = getattr(_native, "_debug_resolve_promotion", None)
    if fn is None:
        raise RuntimeError("Native helper _debug_resolve_promotion is not available")
    return fn(op, a_dtype, b_dtype)


def _debug_last_kernel_trace() -> str:
    """Internal/test helper: last dispatched kernel tag (thread-local)."""
    fn = getattr(_native, "_debug_last_kernel_trace", None)
    if fn is None:
        raise RuntimeError("Native helper _debug_last_kernel_trace is not available")
    return str(fn())


def _debug_clear_kernel_trace() -> None:
    """Internal/test helper: clear last dispatched kernel tag (thread-local)."""
    fn = getattr(_native, "_debug_clear_kernel_trace", None)
    if fn is None:
        raise RuntimeError("Native helper _debug_clear_kernel_trace is not available")
    fn()

# Helper to safely get attributes
def _safe_get(name):
    return _native_mod.safe_get(_native, name)

_TriangularBitMatrix = _safe_get("TriangularBitMatrix")
if _TriangularBitMatrix:
    _original_triangular_bit_matrix_init = getattr(_TriangularBitMatrix, "__init__", None)
    _original_triangular_bit_matrix_random = getattr(_TriangularBitMatrix, "random", None)
else:
    class _TriangularBitMatrix:  # pragma: no cover
        pass
    _original_triangular_bit_matrix_init = None
    _original_triangular_bit_matrix_random = None

# Private aliases for native classes
_IntegerMatrix = _safe_get("IntegerMatrix")
_Int8Matrix = _safe_get("Int8Matrix")
_Int16Matrix = _safe_get("Int16Matrix")
_Int64Matrix = _safe_get("Int64Matrix")
_UInt8Matrix = _safe_get("UInt8Matrix")
_UInt16Matrix = _safe_get("UInt16Matrix")
_UInt32Matrix = _safe_get("UInt32Matrix")
_UInt64Matrix = _safe_get("UInt64Matrix")
_FloatMatrix = _safe_get("FloatMatrix")
_Float16Matrix = _safe_get("Float16Matrix")
_Float32Matrix = _safe_get("Float32Matrix")
_ComplexFloat16Matrix = _safe_get("ComplexFloat16Matrix")
_ComplexFloat32Matrix = _safe_get("ComplexFloat32Matrix")
_ComplexFloat64Matrix = _safe_get("ComplexFloat64Matrix")
_TriangularFloatMatrix = _safe_get("TriangularFloatMatrix")
_TriangularIntegerMatrix = _safe_get("TriangularIntegerMatrix")
_DenseBitMatrix = _safe_get("DenseBitMatrix")
_UnitVector = _safe_get("UnitVector")
_SymmetricMatrix = getattr(_native, "SymmetricMatrix", None)
_AntiSymmetricMatrix = getattr(_native, "AntiSymmetricMatrix", None)

# Public exports
IntegerMatrix = _IntegerMatrix
Int8Matrix = _Int8Matrix
Int16Matrix = _Int16Matrix
Int64Matrix = _Int64Matrix
UInt8Matrix = _UInt8Matrix
UInt16Matrix = _UInt16Matrix
UInt32Matrix = _UInt32Matrix
UInt64Matrix = _UInt64Matrix
FloatMatrix = _FloatMatrix
Float16Matrix = _Float16Matrix
Float32Matrix = _Float32Matrix
ComplexFloat16Matrix = _ComplexFloat16Matrix
ComplexFloat32Matrix = _ComplexFloat32Matrix
ComplexFloat64Matrix = _ComplexFloat64Matrix
TriangularFloatMatrix = _TriangularFloatMatrix
TriangularIntegerMatrix = _TriangularIntegerMatrix
DenseBitMatrix = _DenseBitMatrix
SymmetricMatrix = _SymmetricMatrix
AntiSymmetricMatrix = _AntiSymmetricMatrix
SymmetricFloat64Matrix = _SymmetricMatrix
AntiSymmetricFloat64Matrix = _AntiSymmetricMatrix

# Vector classes
_FloatVector = getattr(_native, "FloatVector", None)
_Float32Vector = getattr(_native, "Float32Vector", None)
_Float16Vector = getattr(_native, "Float16Vector", None)
_ComplexFloat16Vector = getattr(_native, "ComplexFloat16Vector", None)
_ComplexFloat32Vector = getattr(_native, "ComplexFloat32Vector", None)
_ComplexFloat64Vector = getattr(_native, "ComplexFloat64Vector", None)
_Int8Vector = getattr(_native, "Int8Vector", None)
_IntegerVector = getattr(_native, "IntegerVector", None)
_Int16Vector = getattr(_native, "Int16Vector", None)
_Int64Vector = getattr(_native, "Int64Vector", None)
_UInt8Vector = getattr(_native, "UInt8Vector", None)
_UInt16Vector = getattr(_native, "UInt16Vector", None)
_UInt32Vector = getattr(_native, "UInt32Vector", None)
_UInt64Vector = getattr(_native, "UInt64Vector", None)
_BitVector = getattr(_native, "BitVector", None)

FloatVector = _FloatVector
Float32Vector = _Float32Vector
Float16Vector = _Float16Vector
ComplexFloat16Vector = _ComplexFloat16Vector
ComplexFloat32Vector = _ComplexFloat32Vector
ComplexFloat64Vector = _ComplexFloat64Vector
Int8Vector = _Int8Vector
IntegerVector = _IntegerVector
Int16Vector = _Int16Vector
Int64Vector = _Int64Vector
UInt8Vector = _UInt8Vector
UInt16Vector = _UInt16Vector
UInt32Vector = _UInt32Vector
UInt64Vector = _UInt64Vector
BitVector = _BitVector


def _vector_fill(self: Any, value: Any) -> Any:
    for i in range(len(self)):
        self[i] = value
    return self


def _install_vector_fill(*classes: Any) -> None:
    for cls in classes:
        if cls is None:
            continue
        if hasattr(cls, "fill"):
            continue
        cls.fill = _vector_fill  # type: ignore[attr-defined]


_install_vector_fill(
    FloatVector,
    Float32Vector,
    Float16Vector,
    ComplexFloat16Vector,
    ComplexFloat32Vector,
    ComplexFloat64Vector,
    Int8Vector,
    IntegerVector,
    Int16Vector,
    Int64Vector,
    UInt8Vector,
    UInt16Vector,
    UInt32Vector,
    UInt64Vector,
    BitVector,
)

_runtime = _runtime_mod.Runtime(
    cleanup_storage=cleanup_storage,
    set_temporary_file=set_temporary_file,
)

keep_temp_files: bool = False
seed: int | None = None


def _storage_root() -> Path:
    return _runtime.storage_root()


# Perform initial cleanup of temporary files from previous runs
_runtime.initial_cleanup()


def _track_matrix(instance: Any) -> None:
    _runtime.track_matrix(instance)


def _release_tracked_matrices() -> None:
    _runtime.release_tracked_matrices()


def _register_cleanup() -> None:
    _runtime.register_cleanup(keep_temp_files_getter=lambda: keep_temp_files)


_register_cleanup()

_PERSISTENCE_DEPS = _SimpleNamespace(
    CausalSet=CausalSet,
    TriangularBitMatrix=_TriangularBitMatrix,
    DenseBitMatrix=_DenseBitMatrix,
    FloatMatrix=_FloatMatrix,
    Float16Matrix=_Float16Matrix,
    Float32Matrix=_Float32Matrix,
    ComplexFloat16Matrix=_ComplexFloat16Matrix,
    ComplexFloat32Matrix=_ComplexFloat32Matrix,
    ComplexFloat64Matrix=_ComplexFloat64Matrix,
    IntegerMatrix=_IntegerMatrix,
    Int8Matrix=_Int8Matrix,
    Int16Matrix=_Int16Matrix,
    Int64Matrix=_Int64Matrix,
    UInt8Matrix=_UInt8Matrix,
    UInt16Matrix=_UInt16Matrix,
    UInt32Matrix=_UInt32Matrix,
    UInt64Matrix=_UInt64Matrix,
    TriangularFloatMatrix=_TriangularFloatMatrix,
    TriangularIntegerMatrix=_TriangularIntegerMatrix,
    FloatVector=_FloatVector,
    Float32Vector=_Float32Vector,
    Float16Vector=_Float16Vector,
    ComplexFloat16Vector=_ComplexFloat16Vector,
    ComplexFloat32Vector=_ComplexFloat32Vector,
    ComplexFloat64Vector=_ComplexFloat64Vector,
    Int8Vector=_Int8Vector,
    IntegerVector=_IntegerVector,
    Int16Vector=_Int16Vector,
    Int64Vector=_Int64Vector,
    UInt8Vector=_UInt8Vector,
    UInt16Vector=_UInt16Vector,
    UInt32Vector=_UInt32Vector,
    UInt64Vector=_UInt64Vector,
    BitVector=_BitVector,
    UnitVector=_UnitVector,
    IdentityMatrix=getattr(_native, "IdentityMatrix", None),
    native=_native,
)


def save(obj: Any, path: str | Path) -> None:
    return _persistence.save(obj, path, deps=_PERSISTENCE_DEPS)


def load(path: str | Path) -> Any:
    return _persistence.load(path, deps=_PERSISTENCE_DEPS)


_linalg_cache.patch_matrixbase_save(_native, save)
if _FloatMatrix is not None:
    _linalg_cache.patch_inverse(
        FloatMatrix=_FloatMatrix,
        classes=[
            _FloatMatrix,
            _IntegerMatrix,
            _TriangularFloatMatrix,
            _TriangularIntegerMatrix,
            _DenseBitMatrix,
            _TriangularBitMatrix,
        ],
    )

_formatting.configure(np_module=_np, edge_items=4)
if _TriangularBitMatrix and callable(getattr(_TriangularBitMatrix, "size", None)) and callable(
    getattr(_TriangularBitMatrix, "get", None)
):
    _TriangularBitMatrix.__str__ = _formatting.matrix_str


def _mark_temporary_if_auto(matrix: Any) -> None:
    _runtime.mark_temporary_if_auto(matrix)


def _coerce_general_matrix(candidate: Any) -> tuple[int, list[list[Any]]]:
    return _coercion.coerce_general_matrix(candidate, np_module=_np)

_matrix_api.configure(
    np_module=_np,
    warnings_module=warnings,
    native=_native,
    track_matrix=_track_matrix,
    coerce_general_matrix=_coerce_general_matrix,
    IntegerMatrix=_IntegerMatrix,
    Int8Matrix=_Int8Matrix,
    Int16Matrix=_Int16Matrix,
    Int64Matrix=_Int64Matrix,
    UInt8Matrix=_UInt8Matrix,
    UInt16Matrix=_UInt16Matrix,
    UInt32Matrix=_UInt32Matrix,
    UInt64Matrix=_UInt64Matrix,
    FloatMatrix=_FloatMatrix,
    Float16Matrix=_Float16Matrix,
    Float32Matrix=_Float32Matrix,
    ComplexFloat16Matrix=_ComplexFloat16Matrix,
    ComplexFloat32Matrix=_ComplexFloat32Matrix,
    ComplexFloat64Matrix=_ComplexFloat64Matrix,
    TriangularFloatMatrix=_TriangularFloatMatrix,
    TriangularIntegerMatrix=_TriangularIntegerMatrix,
    DenseBitMatrix=_DenseBitMatrix,
    TriangularBitMatrix=_TriangularBitMatrix,
)

TriangularMatrix = _matrix_api.TriangularMatrix

_matrix_api.register_native_matrices(
    TriangularBitMatrix=_TriangularBitMatrix,
    IntegerMatrix=_IntegerMatrix,
    Int8Matrix=_Int8Matrix,
    Int16Matrix=_Int16Matrix,
    Int64Matrix=_Int64Matrix,
    UInt8Matrix=_UInt8Matrix,
    UInt16Matrix=_UInt16Matrix,
    UInt32Matrix=_UInt32Matrix,
    UInt64Matrix=_UInt64Matrix,
    FloatMatrix=_FloatMatrix,
    Float16Matrix=_Float16Matrix,
    Float32Matrix=_Float32Matrix,
    ComplexFloat16Matrix=_ComplexFloat16Matrix,
    ComplexFloat32Matrix=_ComplexFloat32Matrix,
    ComplexFloat64Matrix=_ComplexFloat64Matrix,
    TriangularFloatMatrix=_TriangularFloatMatrix,
    TriangularIntegerMatrix=_TriangularIntegerMatrix,
    DenseBitMatrix=_DenseBitMatrix,
)

_patching.apply_native_storage_patches(
    matrix_classes=[
        (_IntegerMatrix, "backing_file"),
        (_Int8Matrix, "backing_file"),
        (_Int16Matrix, "backing_file"),
        (_Int64Matrix, "backing_file"),
        (_UInt8Matrix, "backing_file"),
        (_UInt16Matrix, "backing_file"),
        (_UInt32Matrix, "backing_file"),
        (_UInt64Matrix, "backing_file"),
        (_FloatMatrix, "backing_file"),
        (_Float16Matrix, "backing_file"),
        (_Float32Matrix, "backing_file"),
        (_ComplexFloat16Matrix, "backing_file"),
        (_ComplexFloat32Matrix, "backing_file"),
        (_ComplexFloat64Matrix, "backing_file"),
        (_TriangularFloatMatrix, "backing_file"),
        (_TriangularIntegerMatrix, "backing_file"),
        (_DenseBitMatrix, "backing_file"),
    ],
    vector_classes=[
        (_FloatVector, "backing_file"),
        (_Float32Vector, "backing_file"),
        (_Float16Vector, "backing_file"),
        (_ComplexFloat16Vector, "backing_file"),
        (_ComplexFloat32Vector, "backing_file"),
        (_ComplexFloat64Vector, "backing_file"),
        (_Int8Vector, "backing_file"),
        (_IntegerVector, "backing_file"),
        (_Int16Vector, "backing_file"),
        (_Int64Vector, "backing_file"),
        (_UInt8Vector, "backing_file"),
        (_UInt16Vector, "backing_file"),
        (_UInt32Vector, "backing_file"),
        (_UInt64Vector, "backing_file"),
        (_BitVector, "backing_file"),
    ],
    track_matrix=_track_matrix,
    mark_temporary_if_auto=_mark_temporary_if_auto,
)

TriangularBitMatrix = _TriangularBitMatrix


def _is_scalar_0d(value: Any) -> bool:
    # Scalar/0D inputs are explicitly unsupported for matrix/vector factories.
    # (Use zeros/ones/empty for allocation.)
    if isinstance(value, (bool, int, float, complex)):
        return True
    if _np is not None:
        if isinstance(value, _np.generic):
            return True
        if isinstance(value, _np.ndarray) and value.ndim == 0:
            return True
    return False


def _is_sequence_like(value: Any) -> bool:
    return _coercion.is_sequence_like(value)


def vector(source: Any, dtype: Any = None, **kwargs: Any) -> Any:
    """Create a 1D vector from vector-like input.

    NumPy alignment: this is a data constructor. Use `zeros/ones/empty` for allocation.
    """
    if _is_scalar_0d(source):
        raise TypeError(
            "vector(...) constructs from data; shape allocation uses zeros/ones/empty. "
            "Scalars/0D are not supported."
        )

    _VectorBase = getattr(_native, "VectorBase", None)
    if _VectorBase is not None and isinstance(source, _VectorBase):
        if dtype is not None or kwargs:
            raise TypeError(
                "vector(...) does not accept dtype/kwargs when source is already a vector. "
                "Pass data instead."
            )
        return source

    _MatrixBase = getattr(_native, "MatrixBase", None)
    if _MatrixBase is not None and isinstance(source, _MatrixBase):
        raise TypeError("vector(...) expects 1D input; use matrix(...) for 2D")

    if _np is not None and isinstance(source, _np.ndarray) and source.ndim != 1:
        raise TypeError("vector(...) expects 1D input; use matrix(...) for 2D")

    if _is_sequence_like(source) and len(source) > 0 and _is_sequence_like(source[0]):
        raise TypeError("vector(...) expects 1D input; use matrix(...) for 2D")
    return _factories.vector_factory(
        source,
        dtype=dtype,
        np_module=_np,
        Float16Vector=_Float16Vector,
        Float32Vector=_Float32Vector,
        FloatVector=_FloatVector,
        ComplexFloat16Vector=_ComplexFloat16Vector,
        ComplexFloat32Vector=_ComplexFloat32Vector,
        ComplexFloat64Vector=_ComplexFloat64Vector,
        Int8Vector=_Int8Vector,
        Int64Vector=_Int64Vector,
        UInt8Vector=_UInt8Vector,
        UInt16Vector=_UInt16Vector,
        UInt32Vector=_UInt32Vector,
        UInt64Vector=_UInt64Vector,
        IntegerVector=_IntegerVector,
        Int16Vector=_Int16Vector,
        BitVector=_BitVector,
        kwargs=kwargs,
    )


def matrix(source: Any, dtype: Any = None, **kwargs: Any) -> Any:
    """Create a vector or matrix from data.

    - 1D input -> vector
    - 2D input -> matrix

    NumPy alignment: this is a data constructor. Use `zeros/ones/empty` for allocation.
    """
    if _is_scalar_0d(source):
        raise TypeError(
            "matrix(...) constructs from data; shape allocation uses zeros/ones/empty. "
            "Scalars/0D are not supported."
        )

    _MatrixBase = getattr(_native, "MatrixBase", None)
    if _MatrixBase is not None and isinstance(source, _MatrixBase):
        if dtype is not None or kwargs:
            raise TypeError(
                "matrix(...) does not accept dtype/kwargs when source is already a matrix. "
                "Pass data instead."
            )
        return source

    _VectorBase = getattr(_native, "VectorBase", None)
    if _VectorBase is not None and isinstance(source, _VectorBase):
        if dtype is not None or kwargs:
            raise TypeError(
                "matrix(...) does not accept dtype/kwargs when source is already a vector. "
                "Pass data instead."
            )
        return source

    if _np is not None and isinstance(source, _np.ndarray):
        if source.ndim == 2:
            if dtype is None:
                return _matrix_api.Matrix(source, **kwargs)
            return _matrix_api.Matrix(source, dtype=dtype, **kwargs)
        if source.ndim == 1:
            return vector(source, dtype=dtype, **kwargs)
        raise TypeError("matrix(...) expects 1D or 2D input")

    if _is_sequence_like(source):
        if len(source) == 0:
            raise ValueError("matrix/vector input must not be empty")
        first = source[0]
        if _is_sequence_like(first):
            # Reject higher-rank nested sequences deterministically.
            if len(first) > 0 and _is_sequence_like(first[0]):
                raise TypeError("matrix(...) expects 1D or 2D input")
            if dtype is None:
                return _matrix_api.Matrix(source, **kwargs)
            return _matrix_api.Matrix(source, dtype=dtype, **kwargs)
        return vector(source, dtype=dtype, **kwargs)

    raise TypeError("matrix(...) expects 1D or 2D input")


def _require_dtype(dtype: Any) -> Any:
    if dtype is None:
        raise TypeError("dtype is required for zeros/ones/empty")
    return dtype


def _allocate_dense_matrix_by_shape(rows: int, cols: int, *, dtype: Any, kwargs: dict[str, Any]) -> Any:
    dtype_norm = _normalize_dtype(dtype, np_module=_np)
    if dtype_norm is None:
        raise TypeError("Unsupported dtype")

    if dtype_norm == "int8":
        if _Int8Matrix is None:
            raise ImportError("Int8Matrix is not available in the native extension")
        return _Int8Matrix(rows, cols, **kwargs)
    if dtype_norm == "int16":
        if _Int16Matrix is None:
            raise ImportError("Int16Matrix is not available in the native extension")
        return _Int16Matrix(rows, cols, **kwargs)
    if dtype_norm == "int32":
        if _IntegerMatrix is None:
            raise ImportError("IntegerMatrix is not available in the native extension")
        return _IntegerMatrix(rows, cols, **kwargs)
    if dtype_norm == "int64":
        if _Int64Matrix is None:
            raise ImportError("Int64Matrix is not available in the native extension")
        return _Int64Matrix(rows, cols, **kwargs)
    if dtype_norm == "uint8":
        if _UInt8Matrix is None:
            raise ImportError("UInt8Matrix is not available in the native extension")
        return _UInt8Matrix(rows, cols, **kwargs)
    if dtype_norm == "uint16":
        if _UInt16Matrix is None:
            raise ImportError("UInt16Matrix is not available in the native extension")
        return _UInt16Matrix(rows, cols, **kwargs)
    if dtype_norm == "uint32":
        if _UInt32Matrix is None:
            raise ImportError("UInt32Matrix is not available in the native extension")
        return _UInt32Matrix(rows, cols, **kwargs)
    if dtype_norm == "uint64":
        if _UInt64Matrix is None:
            raise ImportError("UInt64Matrix is not available in the native extension")
        return _UInt64Matrix(rows, cols, **kwargs)

    if dtype_norm == "bool":
        if _DenseBitMatrix is not None:
            return _DenseBitMatrix(rows, cols, **kwargs)
        if _IntegerMatrix is None:
            raise ImportError("DenseBitMatrix/IntegerMatrix are not available in the native extension")
        return _IntegerMatrix(rows, cols, **kwargs)

    if dtype_norm == "float16":
        if _Float16Matrix is not None:
            return _Float16Matrix(rows, cols, **kwargs)
        if _Float32Matrix is not None:
            return _Float32Matrix(rows, cols, **kwargs)
        if _FloatMatrix is None:
            raise ImportError("Float matrix classes are not available in the native extension")
        return _FloatMatrix(rows, cols, **kwargs)

    if dtype_norm == "float32":
        if _Float32Matrix is not None:
            return _Float32Matrix(rows, cols, **kwargs)
        if _FloatMatrix is None:
            raise ImportError("Float matrix classes are not available in the native extension")
        return _FloatMatrix(rows, cols, **kwargs)

    if dtype_norm == "float64":
        if _FloatMatrix is None:
            raise ImportError("FloatMatrix is not available in the native extension")
        return _FloatMatrix(rows, cols, **kwargs)

    if dtype_norm == "complex_float16":
        if _ComplexFloat16Matrix is None:
            raise ImportError("ComplexFloat16Matrix is not available in the native extension")
        return _ComplexFloat16Matrix(rows, cols, **kwargs)
    if dtype_norm == "complex_float32":
        if _ComplexFloat32Matrix is None:
            raise ImportError("ComplexFloat32Matrix is not available in the native extension")
        return _ComplexFloat32Matrix(rows, cols, **kwargs)
    if dtype_norm == "complex_float64":
        if _ComplexFloat64Matrix is None:
            raise ImportError("ComplexFloat64Matrix is not available in the native extension")
        return _ComplexFloat64Matrix(rows, cols, **kwargs)

    raise TypeError("Unsupported dtype")


def zeros(shape: Any, *, dtype: Any, **kwargs: Any) -> Any:
    """Allocate a vector/matrix filled with zeros. Requires explicit dtype."""
    dtype = _require_dtype(dtype)
    if isinstance(shape, int):
        vec = _factories.vector_factory(
            int(shape),
            dtype=dtype,
            np_module=_np,
            Float16Vector=_Float16Vector,
            Float32Vector=_Float32Vector,
            FloatVector=_FloatVector,
            ComplexFloat16Vector=_ComplexFloat16Vector,
            ComplexFloat32Vector=_ComplexFloat32Vector,
            ComplexFloat64Vector=_ComplexFloat64Vector,
            Int8Vector=_Int8Vector,
            Int64Vector=_Int64Vector,
            UInt8Vector=_UInt8Vector,
            UInt16Vector=_UInt16Vector,
            UInt32Vector=_UInt32Vector,
            UInt64Vector=_UInt64Vector,
            IntegerVector=_IntegerVector,
            Int16Vector=_Int16Vector,
            BitVector=_BitVector,
            kwargs=kwargs,
        )
        vec.fill(0)
        return vec

    if not _is_sequence_like(shape):
        raise TypeError("shape must be an int or a tuple")
    if len(shape) == 1:
        return zeros(int(shape[0]), dtype=dtype, **kwargs)
    if len(shape) != 2:
        raise ValueError("shape must be 1D or 2D")
    rows, cols = int(shape[0]), int(shape[1])
    if rows != cols:
        mat = _allocate_dense_matrix_by_shape(rows, cols, dtype=dtype, kwargs=kwargs)
    else:
        mat = _matrix_api.Matrix(rows, dtype=dtype, **kwargs)
    if hasattr(mat, "fill"):
        mat.fill(0)
    return mat


def ones(shape: Any, *, dtype: Any, **kwargs: Any) -> Any:
    """Allocate a vector/matrix filled with ones. Requires explicit dtype."""
    dtype = _require_dtype(dtype)
    obj = zeros(shape, dtype=dtype, **kwargs)
    obj.fill(1)
    return obj


def empty(shape: Any, *, dtype: Any, **kwargs: Any) -> Any:
    """Allocate a vector/matrix without guaranteeing initialization.

    Note: for some backends this may still be zero-initialized.
    Requires explicit dtype.
    """
    dtype = _require_dtype(dtype)
    if isinstance(shape, int):
        return _factories.vector_factory(
            int(shape),
            dtype=dtype,
            np_module=_np,
            Float16Vector=_Float16Vector,
            Float32Vector=_Float32Vector,
            FloatVector=_FloatVector,
            ComplexFloat16Vector=_ComplexFloat16Vector,
            ComplexFloat32Vector=_ComplexFloat32Vector,
            ComplexFloat64Vector=_ComplexFloat64Vector,
            Int8Vector=_Int8Vector,
            Int64Vector=_Int64Vector,
            UInt8Vector=_UInt8Vector,
            UInt16Vector=_UInt16Vector,
            UInt32Vector=_UInt32Vector,
            UInt64Vector=_UInt64Vector,
            IntegerVector=_IntegerVector,
            Int16Vector=_Int16Vector,
            BitVector=_BitVector,
            kwargs=kwargs,
        )

    if not _is_sequence_like(shape):
        raise TypeError("shape must be an int or a tuple")
    if len(shape) == 1:
        return empty(int(shape[0]), dtype=dtype, **kwargs)
    if len(shape) != 2:
        raise ValueError("shape must be 1D or 2D")
    rows, cols = int(shape[0]), int(shape[1])
    if rows != cols:
        return _allocate_dense_matrix_by_shape(rows, cols, dtype=dtype, kwargs=kwargs)
    return _matrix_api.Matrix(rows, dtype=dtype, **kwargs)


def causal_matrix(source: Any, populate: bool = True, **kwargs: Any) -> Any:
    """Factory function for creating TriangularBitMatrix instances (lower-case API)."""
    if _original_triangular_bit_matrix_init is None:
        raise ImportError(
            "TriangularBitMatrix is not available in the native extension. "
            "Rebuild the extension with causal-matrix support enabled."
        )
    if populate and getattr(TriangularBitMatrix, "random", None) is None:
        raise ImportError(
            "TriangularBitMatrix.random is not available in the native extension. "
            "Rebuild the extension with random generation support enabled."
        )
    return _factories.causal_matrix_factory(
        source,
        populate=populate,
        np_module=_np,
        TriangularBitMatrix=TriangularBitMatrix,
        coerce_general_matrix=_coerce_general_matrix,
        kwargs=kwargs,
    )


if getattr(TriangularBitMatrix, "random", None) is not None:
    causal_matrix.random = TriangularBitMatrix.random


def causet(
    *,
    n: int = None,
    density: float = None,
    spacetime=None,
    seed: int | str | None = None,
    matrix=None,
) -> CausalSet:
    """Lower-case convenience factory that returns a CausalSet."""
    return CausalSet(n=n, density=density, spacetime=spacetime, seed=seed, matrix=matrix)

_OPS_DEPS = _ops.OpsDeps(
    native=_native,
    np_module=_np,
    Matrix=_matrix_api.Matrix,
    TriangularBitMatrix=_TriangularBitMatrix,
    track_matrix=_track_matrix,
    mark_temporary_if_auto=_mark_temporary_if_auto,
    warnings_module=warnings,
)

def matmul(a: Any, b: Any) -> Any:
    """
    Perform matrix multiplication.
    
    If both inputs are TriangularBitMatrices, uses the optimized C++ implementation.
    Otherwise, performs generic multiplication (slow).
    """
    return _ops.matmul(a, b, deps=_OPS_DEPS)


def divide(a: Any, b: Any) -> Any:
    """Elementwise division with NumPy-style 2D broadcasting.

    This is a small convenience wrapper around the `/` operator.
    When the result is a native PyCauset matrix/vector, it is tracked
    for temporary-file lifecycle management.
    """
    result = a / b
    native_matrix_base = getattr(_native, "MatrixBase", None)
    native_vector_base = getattr(_native, "VectorBase", None)
    if native_matrix_base is not None and isinstance(result, native_matrix_base):
        _track_matrix(result)
        _mark_temporary_if_auto(result)
    elif native_vector_base is not None and isinstance(result, native_vector_base):
        _track_matrix(result)
        _mark_temporary_if_auto(result)
    return result


def norm(x: Any) -> float:
    """Return the norm of a vector or matrix.

    - For vectors: $\\ell_2$ norm.
    - For matrices: Frobenius norm.
    """
    fn = getattr(_native, "norm", None)
    if fn is None:
        raise RuntimeError("Native function norm is not available")
    return float(fn(x))

def compute_k(matrix: TriangularBitMatrix, a: float):
    """
    Compute K = C(aI + C)^-1.
    
    Args:
        matrix: The TriangularBitMatrix C.
        a: The scalar a.
        
    Returns:
        A TriangularFloatMatrix representing K.
    """
    return _ops.compute_k(matrix, a, deps=_OPS_DEPS)


def bitwise_not(matrix: Any) -> Any:
    """
    Compute the bitwise inversion (NOT) of a matrix.
    
    Args:
        matrix: The matrix to invert. Must be a TriangularBitMatrix or IntegerMatrix.
        
    Returns:
        A new matrix with inverted bits.
    """
    return _ops.bitwise_not(matrix, deps=_OPS_DEPS)


def invert(matrix: Any) -> Any:
    """
    Compute the linear algebra inverse of a matrix.
    
    Args:
        matrix: The matrix to invert.
        
    Returns:
        The inverse matrix.
        
    Raises:
        RuntimeError: If the matrix is singular (e.g. strictly upper triangular).
    """
    return _ops.invert(matrix, deps=_OPS_DEPS)


def identity(x: Any) -> Any:
    """Create an identity-like matrix.

    Accepted inputs:
    - int N: returns an N×N identity matrix.
    - sequence [rows, cols]: returns a rows×cols rectangular identity (ones on the diagonal, zeros elsewhere).
    - Matrix/Vector: returns an identity-like matrix sized to the input (matrix: rows×cols, vector: N×N).
    """
    I_cls = getattr(_native, "IdentityMatrix", None)
    if I_cls is None:
        raise ImportError("IdentityMatrix is not available in the native extension")

    try:
        return I_cls(x)
    except Exception as e:
        raise TypeError("identity(x) expects an int, [rows, cols], or a matrix/vector") from e


# Alias for IdentityMatrix
I = getattr(_native, "IdentityMatrix", None)

# Python-level field API
from .field import ScalarField


def __getattr__(name):
    return getattr(_native, name)


__all__ = [name for name in dir(_native) if not name.startswith("_")]

# Add pure-Python facade symbols (and any optional native symbols) to __all__.
# IMPORTANT: Only include names that actually resolve; otherwise
# `from pycauset import *` can fail if a symbol is not exported by the native module.
_extra_exports = [
    "save",
    "keep_temp_files",
    "seed",
    "matrix",
    "vector",
    "zeros",
    "ones",
    "empty",
    "causal_matrix",
    "TriangularBitMatrix",
    "matmul",
    "divide",
    "norm",
    "compute_k",
    "bitwise_not",
    "invert",
    "identity",
    "I",
    "causet",
    "MemoryHint",
    "AccessPattern",
    "ScalarField",
]

for _name in _extra_exports:
    if _name in __all__:
        continue
    if _name in globals() or getattr(_native, _name, None) is not None:
        __all__.append(_name)

