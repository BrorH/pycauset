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
from ._internal import coercion as _coercion
from ._internal import formatting as _formatting
from ._internal import patching as _patching
from ._internal import factories as _factories
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

# Alias for CausalSet
Causet = CausalSet


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

Matrix = _matrix_api.Matrix
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

# Aliases (Only CausalMatrix and Matrix are public now)
TriangularBitMatrix = _TriangularBitMatrix

def Vector(size_or_data: Any, dtype: Any = None, **kwargs) -> Any:
    """
    Factory function for creating Vector instances.
    
    Args:
        size_or_data: Size of the vector (int) or data (list/array).
        dtype: Dtype token such as pc.int16, np.int16, "int16" (case-insensitive),
               or builtins like int/float/bool. If None, inferred from data.
        **kwargs: Additional arguments (ignored for now).
    """
    return _factories.vector_factory(
        size_or_data,
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

def CausalMatrix(source: Any, populate: bool = True, **kwargs):
    """
    Factory function for creating TriangularBitMatrix instances.
    
    Args:
        source: The size of the matrix (int) OR a source matrix (list of lists, numpy array).
        populate: If True (default) and source is an int, fills the matrix with random bits (p=0.5).
                  If False and source is an int, returns an empty (all-zeros) matrix.
                  Ignored if source is a list/array.
        **kwargs: Additional arguments passed to TriangularBitMatrix constructor.
    """
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
    CausalMatrix.random = TriangularBitMatrix.random

_OPS_DEPS = _ops.OpsDeps(
    native=_native,
    np_module=_np,
    Matrix=Matrix,
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


# Alias for IdentityMatrix
I = getattr(_native, "IdentityMatrix", None)

# Python-level field API
from .field import ScalarField


def __getattr__(name):
    return getattr(_native, name)


__all__ = [name for name in dir(_native) if not name.startswith("__")]
__all__.extend(["save", "keep_temp_files", "seed", "Matrix", "Vector", "TriangularMatrix", "CausalMatrix", "TriangularBitMatrix", "matmul", "compute_k", "bitwise_not", "invert", "I", "CausalSet", "Causet", "MemoryHint", "AccessPattern", "ScalarField"])

