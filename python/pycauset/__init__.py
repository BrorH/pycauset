"""Python package wrapper that augments the native pycauset extension."""
from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Any

from ._internal import coercion as _coercion
from ._internal import export_guard as _export_guard
from ._internal import factories as _factories
from ._internal import formatting as _formatting
from ._internal import io_observability as _io_observability
from ._internal import lazy_allocation as _lazy_allocation
from ._internal import linalg_cache as _linalg_cache
from ._internal import matrix_api as _matrix_api
from ._internal import native as _native_mod
from ._internal import ops as _ops
from ._internal import patching as _patching
from ._internal import persistence as _persistence
from ._internal import properties as _properties
from ._internal import runtime as _runtime_mod
from ._internal import streaming_manager as _streaming_manager
from ._internal.dtypes import normalize_dtype as _normalize_dtype
from ._internal.warnings import (
    PyCausetDTypeWarning as PyCausetDTypeWarning,
)
from ._internal.warnings import (
    PyCausetOverflowRiskWarning as PyCausetOverflowRiskWarning,
)
from ._internal.warnings import (
    PyCausetPerformanceWarning as PyCausetPerformanceWarning,
)
from ._internal.warnings import (
    PyCausetStorageWarning as PyCausetStorageWarning,
)
from ._internal.warnings import (
    PyCausetWarning as PyCausetWarning,
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

# One-shot native configuration (OpenBLAS thread default + GPU install hint).
from ._internal.native_setup import configure_openblas_threads, emit_gpu_install_hint_once

configure_openblas_threads()
emit_gpu_install_hint_once()

# Import modules that depend on the native extension after it is loaded.
from . import field as field
from . import spacetime as spacetime
from . import synthetic as synthetic
from ._storage import cleanup_storage, set_temporary_file
from .causet import CausalSet
from .spacetime import MinkowskiBox as MinkowskiBox
from .spacetime import MinkowskiCylinder as MinkowskiCylinder
from .spacetime import MinkowskiDiamond as MinkowskiDiamond


def _debug_resolve_promotion(
    op: str,
    a_dtype: str,
    b_dtype: str,
    precision_mode: str | None = None,
) -> dict[str, object]:
    """Internal/test helper: return promotion resolver decision without running kernels."""
    fn = getattr(_native, "_debug_resolve_promotion", None)
    if fn is None:
        raise RuntimeError("Native helper _debug_resolve_promotion is not available")
    return fn(op, a_dtype, b_dtype, precision_mode)


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
_LazyMatrix = _safe_get("LazyMatrix")
_UnitVector = _safe_get("UnitVector")
_SymmetricMatrix = getattr(_native, "SymmetricMatrix", None)
_AntiSymmetricMatrix = getattr(_native, "AntiSymmetricMatrix", None)
_DiagonalMatrix = getattr(_native, "DiagonalMatrix", None)

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
_EXPORT_MAX_BYTES: int | None = None
_DEFAULT_MEMORY_THRESHOLD = getattr(_native, "get_memory_threshold", lambda: None)()
_IO_OBS = _io_observability.IOObservability(memory_threshold_bytes=_DEFAULT_MEMORY_THRESHOLD)

_STREAMING_MANAGER = _streaming_manager.StreamingManager(io_observer=_IO_OBS)
for _desc in (
    _streaming_manager.StreamingDescriptor(
        op="matmul",
        access_pattern="blocked_rowcol",
        tile_budget_fn=_streaming_manager._matmul_tile_budget,
        queue_depth_fn=_streaming_manager._matmul_queue_depth,
        guard=_streaming_manager._matmul_guard,
    ),
    _streaming_manager.StreamingDescriptor(
        op="invert",
        access_pattern="invert_dense",
        tile_budget_fn=_streaming_manager._square_tile_budget,
        queue_depth_fn=_streaming_manager._unary_queue_depth,
        guard=_streaming_manager._square_guard,
    ),
    _streaming_manager.StreamingDescriptor(
        op="eigvalsh",
        access_pattern="symmetric_eigvals",
        tile_budget_fn=_streaming_manager._square_tile_budget,
        queue_depth_fn=_streaming_manager._unary_queue_depth,
        guard=_streaming_manager._square_guard,
    ),
    _streaming_manager.StreamingDescriptor(
        op="eigh",
        access_pattern="symmetric_eigh",
        tile_budget_fn=_streaming_manager._square_tile_budget,
        queue_depth_fn=_streaming_manager._unary_queue_depth,
        guard=_streaming_manager._square_guard,
    ),
    _streaming_manager.StreamingDescriptor(
        op="eigvals_arnoldi",
        access_pattern="arnoldi_topk",
        tile_budget_fn=_streaming_manager._square_tile_budget,
        queue_depth_fn=_streaming_manager._unary_queue_depth,
        guard=_streaming_manager._square_guard,
    ),
):
    _STREAMING_MANAGER.register(_desc)


def _storage_root() -> Path:
    return _runtime.storage_root()


def set_backing_dir(path: str | Path) -> Path:
    """Set the directory used for auto-created backing files.

    This is the recommended way to choose where PyCauset places temporary
    disk-backed payloads (the files behind large matrices).

    Guidance:
    - Call once, right after importing PyCauset, before creating large matrices.
    - Repeatedly switching this directory mid-session is allowed, but not
      guaranteed to be stable for already-created objects.
    """
    import warnings

    from ._internal.warnings import PyCausetStorageWarning

    new_root = Path(path).expanduser().resolve()

    try:
        old_root = _runtime.storage_root().resolve()
    except Exception:
        old_root = None

    # Native code allocates backing files; configure it directly.
    set_native_root = getattr(_native, "set_storage_root", None)
    if callable(set_native_root):
        set_native_root(str(new_root))

    if old_root is not None and new_root != old_root and _runtime.has_live_matrices():
        warnings.warn(
            "Changing the backing directory after matrices have been created can leave "
            "some objects backed by the old directory and may complicate cleanup. "
            "Prefer calling pycauset.set_backing_dir(...) immediately after import.",
            PyCausetStorageWarning,
            stacklevel=2,
        )

    return _runtime.set_storage_root(new_root)


def get_memory_threshold() -> int | None:
    getter = getattr(_native, "get_memory_threshold", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except Exception:
        return None


def set_memory_threshold(limit: int | None) -> int | None:
    setter = getattr(_native, "set_memory_threshold", None)
    getter = getattr(_native, "get_memory_threshold", None)
    if setter is None:
        raise RuntimeError("Native memory threshold control is unavailable")

    target = limit
    if target is None:
        target = _DEFAULT_MEMORY_THRESHOLD

    if target is None:
        raise RuntimeError("Native memory threshold default is unavailable")

    setter(int(target))
    return int(getter()) if getter is not None else int(target)


def set_io_streaming_threshold(limit: int | None) -> int | None:
    """Set the routing threshold for IO observability heuristics (bytes)."""

    return _IO_OBS.set_memory_threshold(limit)


def get_io_streaming_threshold() -> int | None:
    """Return the current IO routing threshold (bytes)."""

    return _IO_OBS.get_memory_threshold()


def last_io_trace(op: str | None = None) -> dict[str, Any] | None:
    """Return the most recent IO trace (optionally filtered by op name)."""

    return _IO_OBS.last(op)


def clear_io_traces() -> None:
    """Clear recorded IO traces (observability/debug only)."""

    _IO_OBS.clear()


# Perform initial cleanup of temporary files from previous runs
_runtime.initial_cleanup()


def _track_matrix(instance: Any) -> None:
    _runtime.track_matrix(instance)


def _release_tracked_matrices() -> None:
    _runtime.release_tracked_matrices()


def _register_cleanup() -> None:
    _runtime.register_cleanup(keep_temp_files_getter=lambda: keep_temp_files)


_register_cleanup()

# NumPy export safety hooks for native matrices/vectors
_MatrixBaseType = getattr(_native, "MatrixBase", None)
_VectorBaseType = getattr(_native, "VectorBase", None)


def _guarded_array_export(self: Any, dtype: Any = None, copy: Any = None) -> Any:
    # NumPy 2.x may pass copy=... to __array__; honor when provided.
    copy_flag = True if copy is None else copy
    if not copy_flag:
        return _export_guard.export_to_numpy(self, allow_huge=False, dtype=dtype, copy=copy_flag)
    try:
        arr = _export_guard.export_to_numpy(self, allow_huge=False, dtype=dtype, copy=False)
    except Exception:
        arr = None
    if arr is not None:
        import numpy as _np_local  # type: ignore

        return _np_local.array(arr, dtype=dtype, copy=True)
    return _export_guard.export_to_numpy(self, allow_huge=False, dtype=dtype, copy=copy_flag)


def _install_array_export_hook(native_type: Any) -> None:
    if native_type is None:
        return
    if getattr(native_type, "__pycauset_array_wrapped__", False):
        return
    try:
        native_type.__array__ = _guarded_array_export
        # Disallow ufuncs so standard ops take precedence (forcing __radd__ over NumPy's buffer path)
        native_type.__array_ufunc__ = None
        native_type.__array_priority__ = 1e6
        setattr(native_type, "__pycauset_array_wrapped__", True)
    except Exception:
        return


def _coerce_scalar_for_native(value: Any) -> Any:
    """Downcast NumPy scalar types to plain Python scalars for pybind setters."""
    try:
        import numpy as _np_local  # type: ignore
    except Exception:
        _np_local = None

    if _np_local is not None and isinstance(value, _np_local.generic):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _wrap_setitem_for_scalars(base_type: Any) -> None:
    if base_type is None:
        return
    if getattr(base_type, "__pycauset_setitem_wrapped__", False):
        return
    original = getattr(base_type, "__setitem__", None)
    set_method = getattr(base_type, "set", None)
    if not callable(original):
        return

    def _wrapped(self: Any, key: Any, value: Any, _orig=original) -> Any:
        coerced = _coerce_scalar_for_native(value)
        if callable(set_method):
            try:
                # Matrices use (i, j); vectors use i.
                if isinstance(key, tuple) and len(key) == 2:
                    return set_method(self, int(key[0]), int(key[1]), coerced)
                return set_method(self, int(key), coerced)
            except Exception:
                # Fall back to the original setter on any failure (e.g., slicing).
                pass
        try:
            return _orig(self, key, coerced)
        except RuntimeError as exc:
            msg = str(exc)
            if "Unable to cast" in msg:
                raise TypeError(msg) from exc
            raise

    base_type.__setitem__ = _wrapped  # type: ignore[attr-defined]
    setattr(base_type, "__pycauset_setitem_wrapped__", True)


try:
    if _MatrixBaseType is not None:
        _install_array_export_hook(_MatrixBaseType)
        _wrap_setitem_for_scalars(_MatrixBaseType)
    if _VectorBaseType is not None:
        _install_array_export_hook(_VectorBaseType)
        _wrap_setitem_for_scalars(_VectorBaseType)
except Exception:
    pass

# Ensure NumPy scalar coercion also applies to concrete native classes that
# may override __setitem__.
for _cls in (
    _Float16Matrix,
    _Float32Matrix,
    _FloatMatrix,
    _IntegerMatrix,
    _Int8Matrix,
    _Int16Matrix,
    _Int64Matrix,
    _UInt8Matrix,
    _UInt16Matrix,
    _UInt32Matrix,
    _UInt64Matrix,
    _ComplexFloat16Matrix,
    _ComplexFloat32Matrix,
    _ComplexFloat64Matrix,
    _TriangularFloatMatrix,
    _TriangularIntegerMatrix,
    _DenseBitMatrix,
    _Float16Vector,
    _Float32Vector,
    _FloatVector,
    _IntegerVector,
    _Int8Vector,
    _Int16Vector,
    _Int64Vector,
    _UInt8Vector,
    _UInt16Vector,
    _UInt32Vector,
    _UInt64Vector,
    _ComplexFloat16Vector,
    _ComplexFloat32Vector,
    _ComplexFloat64Vector,
    _BitVector,
):
    _install_array_export_hook(_cls)
    _wrap_setitem_for_scalars(_cls)


def set_export_max_bytes(limit: int | None) -> None:
    """Set the materialization ceiling (bytes) for NumPy exports.

    None disables the size ceiling (file-backed objects still hard-error).
    """

    global _EXPORT_MAX_BYTES
    _EXPORT_MAX_BYTES = limit
    _export_guard.set_max_bytes(limit)
    try:
        setter = getattr(_native, "_set_numpy_export_max_bytes", None)
        if callable(setter):
            setter(limit)
    except Exception:
        pass


def to_numpy(obj: Any, *, allow_huge: bool = False, dtype: Any = None, copy: bool = True) -> Any:
    """Convert a PyCauset object to NumPy, enforcing out-of-core safety.

    - By default, file-backed or over-ceiling exports hard-error.
    - Pass allow_huge=True to intentionally materialize.
    - Ceiling is controlled via set_export_max_bytes(...).

    Lazy dtype-deferred allocations (zeros/ones/empty without dtype) are
    materialized first; a still-typeless empty() raises.
    """
    if isinstance(obj, _lazy_allocation.LazyAllocated):
        obj = obj._materialize()

    return _export_guard.export_to_numpy(obj, allow_huge=allow_huge, dtype=dtype, copy=copy)

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
    DiagonalMatrix=_DiagonalMatrix,
    SymmetricMatrix=_SymmetricMatrix,
    AntiSymmetricMatrix=_AntiSymmetricMatrix,
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


def _array_to_pycauset(arr: Any) -> Any:
    if _np is None:
        raise RuntimeError("NumPy is required for NumPy-based imports")

    if not isinstance(arr, _np.ndarray):
        arr = _np.array(arr)

    if arr.ndim == 1:
        return vector(arr)
    if arr.ndim == 2:
        return matrix(arr)
    raise ValueError("Only 1D or 2D arrays can be imported")


def load_npy(path: str | Path) -> Any:
    if _np is None:
        raise RuntimeError("NumPy is required for .npy import")
    arr = _np.load(path)
    return _array_to_pycauset(arr)


def load_npz(path: str | Path, *, key: str | None = None) -> Any:
    if _np is None:
        raise RuntimeError("NumPy is required for .npz import")

    data = _np.load(path)
    keys = list(data.keys())
    if not keys:
        raise ValueError(".npz archive is empty")

    use_key = key if key is not None else sorted(keys)[0]
    if use_key not in data:
        raise KeyError(f"Key '{use_key}' not found in npz archive")
    arr = data[use_key]
    return _array_to_pycauset(arr)


def save_npy(obj: Any, path: str | Path, *, allow_huge: bool = False, dtype: Any = None) -> Path:
    if _np is None:
        raise RuntimeError("NumPy is required for .npy export")
    arr = to_numpy(obj, allow_huge=allow_huge, dtype=dtype)
    _np.save(path, arr)
    return Path(path)


def save_npz(
    obj: Any,
    path: str | Path,
    *,
    allow_huge: bool = False,
    dtype: Any = None,
    key: str = "array",
) -> Path:
    if _np is None:
        raise RuntimeError("NumPy is required for .npz export")
    arr = to_numpy(obj, allow_huge=allow_huge, dtype=dtype)
    _np.savez(path, **{key: arr})
    return Path(path)


def convert_file(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    dst_format: str | None = None,
    allow_huge: bool = False,
    dtype: Any = None,
    npz_key: str | None = None,
) -> Path:
    """Convert between .pycauset and NumPy formats (.npy/.npz).

    - dst_format defaults from dst_path suffix when not provided.
    - npz imports default to the first key unless npz_key is provided.
    - Exports respect the NumPy materialization guard via allow_huge.
    """

    src = Path(src_path)
    dst = Path(dst_path)

    def _infer_format(p: Path, explicit: str | None) -> str:
        if explicit:
            return explicit.lower()
        suf = p.suffix.lower().lstrip(".")
        if suf in {"pycauset", "npy", "npz"}:
            return suf
        raise ValueError("Could not infer format from path; provide dst_format")

    src_fmt = _infer_format(src, None)
    dst_fmt = _infer_format(dst, dst_format)

    obj = None
    try:
        if src_fmt == "pycauset":
            obj = load(src)
        elif src_fmt == "npy":
            obj = load_npy(src)
        elif src_fmt == "npz":
            obj = load_npz(src, key=npz_key)
        else:
            raise ValueError(f"Unsupported source format: {src_fmt}")

        if dst_fmt == "pycauset":
            save(obj, dst)
        elif dst_fmt == "npy":
            save_npy(obj, dst, allow_huge=allow_huge, dtype=dtype)
        elif dst_fmt == "npz":
            save_npz(obj, dst, allow_huge=allow_huge, dtype=dtype, key=npz_key or "array")
        else:
            raise ValueError(f"Unsupported destination format: {dst_fmt}")
    finally:
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    return dst


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

_properties.apply_properties_patches(
    classes=[
        _LazyMatrix,
        _IntegerMatrix,
        _Int8Matrix,
        _Int16Matrix,
        _Int64Matrix,
        _UInt8Matrix,
        _UInt16Matrix,
        _UInt32Matrix,
        _UInt64Matrix,
        _FloatMatrix,
        _Float16Matrix,
        _Float32Matrix,
        _ComplexFloat16Matrix,
        _ComplexFloat32Matrix,
        _ComplexFloat64Matrix,
        _TriangularFloatMatrix,
        _TriangularIntegerMatrix,
        _DenseBitMatrix,
        _TriangularBitMatrix,
        _FloatVector,
        _Float32Vector,
        _Float16Vector,
        _ComplexFloat16Vector,
        _ComplexFloat32Vector,
        _ComplexFloat64Vector,
        _Int8Vector,
        _IntegerVector,
        _Int16Vector,
        _Int64Vector,
        _UInt8Vector,
        _UInt16Vector,
        _UInt32Vector,
        _UInt64Vector,
        _BitVector,
        _UnitVector,
        _DiagonalMatrix,
        _SymmetricMatrix,
        _AntiSymmetricMatrix,
    ]
)

_properties.apply_properties_mutation_patches(
    classes=[
        _IntegerMatrix,
        _Int8Matrix,
        _Int16Matrix,
        _Int64Matrix,
        _UInt8Matrix,
        _UInt16Matrix,
        _UInt32Matrix,
        _UInt64Matrix,
        _FloatMatrix,
        _Float16Matrix,
        _Float32Matrix,
        _ComplexFloat16Matrix,
        _ComplexFloat32Matrix,
        _ComplexFloat64Matrix,
        _TriangularFloatMatrix,
        _TriangularIntegerMatrix,
        _DenseBitMatrix,
        _TriangularBitMatrix,
        _FloatVector,
        _Float32Vector,
        _Float16Vector,
        _ComplexFloat16Vector,
        _ComplexFloat32Vector,
        _ComplexFloat64Vector,
        _Int8Vector,
        _IntegerVector,
        _Int16Vector,
        _Int64Vector,
        _UInt8Vector,
        _UInt16Vector,
        _UInt32Vector,
        _UInt64Vector,
        _BitVector,
        _UnitVector,
    ]
)

_properties.apply_properties_view_patches(
    classes=[
        _IntegerMatrix,
        _Int8Matrix,
        _Int16Matrix,
        _Int64Matrix,
        _UInt8Matrix,
        _UInt16Matrix,
        _UInt32Matrix,
        _UInt64Matrix,
        _FloatMatrix,
        _Float16Matrix,
        _Float32Matrix,
        _ComplexFloat16Matrix,
        _ComplexFloat32Matrix,
        _ComplexFloat64Matrix,
        _TriangularFloatMatrix,
        _TriangularIntegerMatrix,
        _DenseBitMatrix,
        _TriangularBitMatrix,
        _FloatVector,
        _Float32Vector,
        _Float16Vector,
        _ComplexFloat16Vector,
        _ComplexFloat32Vector,
        _ComplexFloat64Vector,
        _Int8Vector,
        _IntegerVector,
        _Int16Vector,
        _Int64Vector,
        _UInt8Vector,
        _UInt16Vector,
        _UInt32Vector,
        _UInt64Vector,
        _BitVector,
        _UnitVector,
    ],
    track_matrix=_track_matrix,
    mark_temporary_if_auto=_mark_temporary_if_auto,
)

_properties.apply_properties_operator_patches(
    classes=[
        _IntegerMatrix,
        _Int8Matrix,
        _Int16Matrix,
        _Int64Matrix,
        _UInt8Matrix,
        _UInt16Matrix,
        _UInt32Matrix,
        _UInt64Matrix,
        _FloatMatrix,
        _Float16Matrix,
        _Float32Matrix,
        _ComplexFloat16Matrix,
        _ComplexFloat32Matrix,
        _ComplexFloat64Matrix,
        _TriangularFloatMatrix,
        _TriangularIntegerMatrix,
        _DenseBitMatrix,
        _TriangularBitMatrix,
    ]
)

_properties.apply_properties_arithmetic_patches(
    classes=[
        _IntegerMatrix,
        _Int8Matrix,
        _Int16Matrix,
        _Int64Matrix,
        _UInt8Matrix,
        _UInt16Matrix,
        _UInt32Matrix,
        _UInt64Matrix,
        _FloatMatrix,
        _Float16Matrix,
        _Float32Matrix,
        _ComplexFloat16Matrix,
        _ComplexFloat32Matrix,
        _ComplexFloat64Matrix,
        _TriangularFloatMatrix,
        _TriangularIntegerMatrix,
        _DenseBitMatrix,
        _TriangularBitMatrix,
        _FloatVector,
        _Float32Vector,
        _Float16Vector,
        _ComplexFloat16Vector,
        _ComplexFloat32Vector,
        _ComplexFloat64Vector,
        _Int8Vector,
        _IntegerVector,
        _Int16Vector,
        _Int64Vector,
        _UInt8Vector,
        _UInt16Vector,
        _UInt32Vector,
        _UInt64Vector,
        _BitVector,
        _UnitVector,
    ]
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


def vector(
    source: Any,
    dtype: Any = None,
    *,
    max_in_ram_bytes: int | None = None,
    **kwargs: Any,
) -> Any:
    """Create a 1D vector from vector-like input.

    NumPy alignment: this is a data constructor. Use `zeros/ones/empty` for allocation.
    The optional max_in_ram_bytes routes NumPy inputs through native-backed asarray when set.
    """
    max_in_ram_bytes = kwargs.pop("max_in_ram_bytes", max_in_ram_bytes)
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
        native=_native,
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
        max_in_ram_bytes=max_in_ram_bytes,
    )


def matrix(source: Any, dtype: Any = None, **kwargs: Any) -> Any:
    """Create a vector or matrix from data.

    - 1D input -> vector
    - 2D input -> matrix

    NumPy alignment: this is a data constructor. Use `zeros/ones/empty` for allocation.

    ``storage`` controls backing for 2D NumPy-array input:

    - ``storage="ram"`` (default): RAM-first, spills to disk only when the memory
      threshold is exceeded.
    - ``storage="disk"``: force a disk-backed (mmap) matrix regardless of size.

    For explicit persistence use `save()`/`load()`; for automatic spill control use
    `set_memory_threshold()`.
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

    # Internal BlockMatrix support (Phase F integration): if the input is already
    # an internal block/view/thunk matrix, treat it as a matrix.
    try:
        from ._internal.blockmatrix import BlockMatrix  # type: ignore
        from ._internal.submatrix_view import SubmatrixView  # type: ignore
        from ._internal.thunks import ThunkBlock  # type: ignore
    except Exception:  # pragma: no cover
        BlockMatrix = None  # type: ignore[assignment]
        SubmatrixView = None  # type: ignore[assignment]
        ThunkBlock = None  # type: ignore[assignment]

    if BlockMatrix is not None and isinstance(source, BlockMatrix):
        if dtype is not None or kwargs:
            raise TypeError(
                "matrix(...) does not accept dtype/kwargs when source is already a matrix. "
                "Pass data instead."
            )
        return source
    if SubmatrixView is not None and isinstance(source, SubmatrixView):
        if dtype is not None or kwargs:
            raise TypeError(
                "matrix(...) does not accept dtype/kwargs when source is already a matrix. "
                "Pass data instead."
            )
        return source
    if ThunkBlock is not None and isinstance(source, ThunkBlock):
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

            # Phase F integration: disambiguate block-grid construction.
            # - All elements are matrices -> BlockMatrix
            # - No elements are matrices -> dense data constructor
            # - Mixed matrices and scalars -> error
            native_matrix_base = getattr(_native, "MatrixBase", None)

            def _is_matrix_obj(x: Any) -> bool:
                if native_matrix_base is not None and isinstance(x, native_matrix_base):
                    return True
                if BlockMatrix is not None and isinstance(x, BlockMatrix):
                    return True
                if SubmatrixView is not None and isinstance(x, SubmatrixView):
                    return True
                if ThunkBlock is not None and isinstance(x, ThunkBlock):
                    return True
                return False

            any_matrix = False
            all_matrix = True
            for row in source:
                if not _is_sequence_like(row):
                    all_matrix = False
                    continue
                for item in row:
                    is_m = _is_matrix_obj(item)
                    any_matrix = any_matrix or is_m
                    all_matrix = all_matrix and is_m

            if any_matrix and not all_matrix:
                raise TypeError(
                    "matrix(...) 2D input mixes matrices and scalars; this is ambiguous. "
                    "Provide either all matrices (block grid) or all numeric scalars (dense data)."
                )

            if all_matrix:
                if dtype is not None or kwargs:
                    raise TypeError("matrix(block_grid) does not accept dtype/kwargs")
                if BlockMatrix is None:
                    raise ImportError("BlockMatrix support is unavailable")
                return BlockMatrix(source)

            if dtype is None:
                return _matrix_api.Matrix(source, **kwargs)
            return _matrix_api.Matrix(source, dtype=dtype, **kwargs)
        return vector(source, dtype=dtype, **kwargs)

    raise TypeError("matrix(...) expects 1D or 2D input")


def _deduce_dtype_from_value(value: Any) -> str:
    """Map a written Python value to a concrete dtype token.

    Matches the promotion ethos: bool -> "bool", int -> "int32",
    float -> "float64", complex -> "complex_float64". NumPy scalar types are
    normalized through `normalize_dtype`.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int32"
    if isinstance(value, float):
        return "float64"
    if isinstance(value, complex):
        return "complex_float64"
    token = _normalize_dtype(type(value), np_module=_np)
    if token is None:
        raise TypeError(
            f"cannot infer a dtype from a value of type {type(value).__name__}; "
            "pass dtype= explicitly"
        )
    return token


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


def _allocate_like(shape: Any, kind: str, dtype: Any, kwargs: dict[str, Any]) -> Any:
    """Allocate a vector/matrix of the given dtype and apply the fill pattern.

    kind is one of "zeros", "ones", or "empty" (empty applies no fill).
    """
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
        if kind == "zeros":
            vec.fill(0)
        elif kind == "ones":
            vec.fill(1)
        return vec

    if not _is_sequence_like(shape):
        raise TypeError("shape must be an int or a tuple")
    if len(shape) == 1:
        return _allocate_like(int(shape[0]), kind, dtype, kwargs)
    if len(shape) != 2:
        raise ValueError("shape must be 1D or 2D")
    rows, cols = int(shape[0]), int(shape[1])
    if rows != cols:
        mat = _allocate_dense_matrix_by_shape(rows, cols, dtype=dtype, kwargs=kwargs)
    else:
        mat = _matrix_api.Matrix(rows, dtype=dtype, **kwargs)
    if kind == "zeros":
        if hasattr(mat, "fill"):
            mat.fill(0)
    elif kind == "ones":
        if hasattr(mat, "fill"):
            mat.fill(1)
    return mat


def _lazy_allocate(shape: Any, kind: str, kwargs: dict[str, Any]) -> Any:
    """Build a dtype-deferred allocation wrapper for the given fill pattern."""
    if isinstance(shape, int):
        shp = (int(shape),)
        ndim = 1
    elif _is_sequence_like(shape):
        if len(shape) == 1:
            shp = (int(shape[0]),)
            ndim = 1
        elif len(shape) == 2:
            shp = (int(shape[0]), int(shape[1]))
            ndim = 2
        else:
            raise ValueError("shape must be 1D or 2D")
    else:
        raise TypeError("shape must be an int or a tuple")

    def _materialize(dtype: str) -> Any:
        return _allocate_like(shape, kind, dtype, kwargs)

    return _lazy_allocation.LazyAllocated(
        kind=kind,
        shape=shp,
        ndim=ndim,
        materialize=_materialize,
        deduce_dtype=_deduce_dtype_from_value,
    )


def zeros(shape: Any, *, dtype: Any = None, **kwargs: Any) -> Any:
    """Allocate a vector/matrix filled with zeros.

    With no dtype, returns a dtype-deferred wrapper that resolves to int32 on
    first use (or to the dtype of the first written value).
    """
    if dtype is None:
        return _lazy_allocate(shape, "zeros", kwargs)
    return _allocate_like(shape, "zeros", dtype, kwargs)


def ones(shape: Any, *, dtype: Any = None, **kwargs: Any) -> Any:
    """Allocate a vector/matrix filled with ones.

    With no dtype, returns a dtype-deferred wrapper that resolves to int32 on
    first use (or to the dtype of the first written value).
    """
    if dtype is None:
        return _lazy_allocate(shape, "ones", kwargs)
    return _allocate_like(shape, "ones", dtype, kwargs)


def empty(shape: Any, *, dtype: Any = None, **kwargs: Any) -> Any:
    """Allocate a vector/matrix without guaranteeing initialization.

    With no dtype, returns a dtype-deferred wrapper; using it before a write
    raises (no silent wrong answer).
    """
    if dtype is None:
        return _lazy_allocate(shape, "empty", kwargs)
    return _allocate_like(shape, "empty", dtype, kwargs)


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
    io_observer=_IO_OBS,
    streaming_manager=_STREAMING_MANAGER,
)

def matmul(a: Any, b: Any) -> Any:
    """
    Perform matrix multiplication.
    
    If both inputs are TriangularBitMatrices, uses the optimized C++ implementation.
    Otherwise, performs generic multiplication (slow).
    """
    return _ops.matmul(a, b, deps=_OPS_DEPS)


def dot(a: Any, b: Any) -> Any:
    """Compute a dot product or matrix product, matching NumPy's `np.dot`.

    Semantics:
    - vector . vector: scalar inner product (no implicit conjugation).
    - matrix . matrix: matrix multiplication.
    - matrix . vector / vector . matrix: matrix-vector product.

    Matrix cases route through `matmul`.
    """
    native_vector_base = getattr(_native, "VectorBase", None)
    a_is_vector = native_vector_base is not None and isinstance(a, native_vector_base)
    b_is_vector = native_vector_base is not None and isinstance(b, native_vector_base)

    if a_is_vector and b_is_vector:
        fn = getattr(a, "dot", None)
        if fn is None:
            raise TypeError("pycauset.dot: expected a vector-like object with a .dot(other) method")
        return fn(b)

    result = _ops.matmul(a, b, deps=_OPS_DEPS)

    # np.dot(vector, matrix) returns a 1-D vector, but the native vec-mat path
    # yields a transposed (row) vector; un-transpose it to match np.dot.
    if a_is_vector and not b_is_vector:
        transpose = getattr(result, "transpose", None)
        if callable(transpose):
            try:
                result = transpose()
            except Exception:
                pass
    return result


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


def norm(x: Any, ord: Any = None) -> float:
    """Return the norm of a vector or matrix.

    - ord=None: Frobenius norm (matrix) / L2 norm (vector), cached.
    - ord='fro': Frobenius norm.
    - ord=2: spectral norm (matrix, largest singular value) / L2 norm (vector).
    - ord=1, ord=inf, ord='nuc': 1-norm, infinity-norm, nuclear norm.
    """

    if ord is not None:
        # Explicit ord value: use ops.norm, which is correct for the spectral
        # (2) norm and routes 1/inf/nuc through NumPy.
        return _ops.norm(x, ord=ord, deps=_OPS_DEPS)

    # Cached-derived fast path (R1_PROPERTIES): if the object exposes a properties
    # mapping and already has a cached norm, use it.
    try:
        props = getattr(x, "properties", None)
        if props is not None:
            cached = props.get("norm")  # type: ignore[union-attr]
            if cached is not None:
                return float(cached)
    except Exception:
        pass

    fn = getattr(_native, "norm", None)
    if fn is None:
        raise RuntimeError("Native function norm is not available")
    out = float(fn(x))

    # Best-effort populate cache.
    try:
        props = getattr(x, "properties", None)
        if props is not None:
            props["norm"] = out  # type: ignore[index]
    except Exception:
        pass

    return out


def sum(x: Any) -> complex:
    """Return the sum of all elements in a vector or matrix.

    Notes:
    - For real inputs, the result is returned as `complex` with zero imaginary part.
    - For complex inputs, the full complex sum is returned.
    """

    # Cached-derived fast path (R1_PROPERTIES): if the object exposes a properties
    # mapping and already has a cached sum, use it.
    try:
        props = getattr(x, "properties", None)
        if props is not None:
            cached = props.get("sum")  # type: ignore[union-attr]
            if cached is not None:
                return complex(cached)
    except Exception:
        pass

    fn = getattr(_native, "sum", None)
    if fn is None:
        raise RuntimeError("Native function sum is not available")
    out = complex(fn(x))

    # Best-effort populate cache.
    try:
        props = getattr(x, "properties", None)
        if props is not None:
            props["sum"] = out  # type: ignore[index]
    except Exception:
        pass

    return out

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


def bitwise_and(a: Any, b: Any) -> Any:
    """Elementwise bitwise AND of two bit matrices or vectors."""
    return _ops.bitwise_and(a, b, deps=_OPS_DEPS)


def bitwise_or(a: Any, b: Any) -> Any:
    """Elementwise bitwise OR of two bit matrices or vectors."""
    return _ops.bitwise_or(a, b, deps=_OPS_DEPS)


def bitwise_xor(a: Any, b: Any) -> Any:
    """Elementwise bitwise XOR of two bit matrices or vectors."""
    return _ops.bitwise_xor(a, b, deps=_OPS_DEPS)


def bitwise_nand(a: Any, b: Any) -> Any:
    """Elementwise bitwise NAND (NOT AND)."""
    return _ops.bitwise_nand(a, b, deps=_OPS_DEPS)


def bitwise_nor(a: Any, b: Any) -> Any:
    """Elementwise bitwise NOR (NOT OR)."""
    return _ops.bitwise_nor(a, b, deps=_OPS_DEPS)


def bitwise_xnor(a: Any, b: Any) -> Any:
    """Elementwise bitwise XNOR (NOT XOR)."""
    return _ops.bitwise_xnor(a, b, deps=_OPS_DEPS)


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


def lstsq(a: Any, b: Any) -> Any:
    """Least-squares solve (endpoint-first baseline).

    Returns only the solution `x` (unlike NumPy, which returns a tuple).
    """
    return _ops.lstsq(a, b, deps=_OPS_DEPS)


def slogdet(a: Any) -> tuple[float, float]:
    """Return `(sign, logabsdet)` using the current determinant implementation."""
    return _ops.slogdet(a, deps=_OPS_DEPS)


def cond(a: Any, p: Any = None) -> float:
    """Condition number estimate via `norm(a) * norm(invert(a))`."""
    return _ops.cond(a, deps=_OPS_DEPS, p=p)


def eigh(a: Any) -> tuple[Any, Any]:
    """Eigen-decomposition for symmetric/Hermitian matrices (NumPy fallback)."""
    return _ops.eigh(a, deps=_OPS_DEPS)


def eigvalsh(a: Any) -> Any:
    """Eigenvalues for symmetric/Hermitian matrices (NumPy fallback)."""
    return _ops.eigvalsh(a, deps=_OPS_DEPS)


def eigvals_arnoldi(a: Any, k: int, m: int, tol: float = 1e-6) -> Any:
    """Top-k eigenvalues via Arnoldi/Lanczos-style iteration (native when available)."""

    return _ops.eigvals_arnoldi(a, k, m, tol, deps=_OPS_DEPS)


def solve_triangular(*args: Any, **kwargs: Any) -> Any:
    if "deps" in kwargs:
        raise TypeError("solve_triangular: deps is internal")
    kwargs["deps"] = _OPS_DEPS
    return _ops.solve_triangular(*args, **kwargs)


def pinv(a: Any) -> Any:
    return _ops.pinv(a, deps=_OPS_DEPS)


def eig(a: Any) -> tuple[Any, Any]:
    """Eigen-decomposition for general matrices (NumPy fallback)."""
    return _ops.eig(a, deps=_OPS_DEPS)


def eigvals(a: Any) -> Any:
    """Eigenvalues for general matrices (NumPy fallback)."""
    return _ops.eigvals(a, deps=_OPS_DEPS)


def eigvals_skew(a: Any, k: int) -> Any:
    """Top-k (by magnitude) eigenvalues of a real skew-symmetric matrix.

    A real skew-symmetric matrix (``A == -A.T``) has purely imaginary eigenvalues
    that come in ``+/-i*lambda`` pairs (plus a zero eigenvalue for odd dimension).
    Returns a complex vector of length ``min(k, n)`` sorted by descending
    magnitude.
    """

    return _ops.eigvals_skew(a, k, deps=_OPS_DEPS)


def eig_skew(a: Any, k: int) -> tuple[Any, Any]:
    """Top-k eigenvalues and eigenvectors of a real skew-symmetric matrix.

    Returns ``(w, v)``: ``w`` is a complex vector of the top-|k| eigenvalues by
    magnitude, and ``v`` is the ``n x min(k, n)`` complex matrix of the matching
    eigenvectors, in the same order. This is the diagonalization used to build the
    Sorkin-Johnston vacuum ``W`` from ``i*Delta``.
    """

    return _ops.eig_skew(a, k, deps=_OPS_DEPS)


def trace(a: Any) -> Any:
    """Return the sum of the diagonal elements."""
    return _ops.trace(a, deps=_OPS_DEPS)


def cholesky(a: Any) -> Any:
    """Return the Cholesky decomposition."""
    return _ops.cholesky(a, deps=_OPS_DEPS)


def qr(a: Any, mode: str = 'reduced') -> Any:
    """Return QR decomposition."""
    return _ops.qr(a, mode=mode, deps=_OPS_DEPS)


def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Any:
    """Return SVD decomposition."""
    return _ops.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, deps=_OPS_DEPS)


def lu(a: Any) -> Any:
    """Return LU decomposition (P, L, U)."""
    return _ops.lu(a, deps=_OPS_DEPS)


def solve(a: Any, b: Any) -> Any:
    """Solve linear system AX = B."""
    return _ops.solve(a, b, deps=_OPS_DEPS)


def determinant(a: Any) -> Any:
    """Return the determinant of a square matrix."""
    return _ops.determinant(a, deps=_OPS_DEPS)


def svdvals(a: Any) -> Any:
    """Singular values of a matrix, descending."""
    return _ops.svdvals(a, deps=_OPS_DEPS)


def matrix_rank(a: Any, tol: Any = None) -> int:
    """Numerical rank of a matrix."""
    return _ops.matrix_rank(a, tol=tol, deps=_OPS_DEPS)


def matrix_power(a: Any, n: int) -> Any:
    """Integer power of a square matrix (A^n)."""
    return _ops.matrix_power(a, n, deps=_OPS_DEPS)


def outer(a: Any, b: Any) -> Any:
    """Outer product of two vectors."""
    return _ops.outer(a, b, deps=_OPS_DEPS)


def cross(a: Any, b: Any) -> Any:
    """Cross product of two 3-element vectors."""
    return _ops.cross(a, b, deps=_OPS_DEPS)


def vecdot(a: Any, b: Any) -> Any:
    """Conjugate dot product: sum(conj(a) * b)."""
    return _ops.vecdot(a, b, deps=_OPS_DEPS)


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


def _as_square_array(data: Any, *, name: str) -> Any:
    if _np is None:
        raise RuntimeError("NumPy is required for symmetric/antisymmetric construction")
    arr = _np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"{name}() expects a 2D array, got {arr.ndim}D")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name}() requires a square matrix, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name}() does not support empty input")
    return arr


def _is_inexact_dtype(arr: Any) -> bool:
    return bool(_np is not None and _np.issubdtype(arr.dtype, _np.inexact))


def symmetric(data: Any, *, rtol: float | None = None, atol: float | None = None) -> Any:
    """Create a symmetric matrix (A == A.T) with validation.

    Storage depends on the input dtype:

    - float32/float16/float64 -> native `SymmetricMatrix` (float64), which stores
      only the upper triangle including the diagonal, roughly halving memory.
    - integer/bool -> the corresponding dense matrix with `is_symmetric=True`
      asserted (exact integer storage, no packing in R1).

    Validation always runs: exact `array_equal` for integer/bool input and
    tolerance-based `allclose` (rtol/atol) for floating input. Complex input is
    rejected: the native SymmetricMatrix is real-valued.
    """
    arr = _as_square_array(data, name="symmetric")
    if _np.iscomplexobj(arr):
        raise TypeError("symmetric() operates on real matrices; the native SymmetricMatrix is float64")

    if _is_inexact_dtype(arr):
        if not _np.allclose(
            arr, arr.T, rtol=(1e-5 if rtol is None else rtol), atol=(1e-8 if atol is None else atol)
        ):
            raise ValueError("symmetric() input is not symmetric (A != A.T within tolerance)")
    else:
        if not _np.array_equal(arr, arr.T):
            raise ValueError("symmetric() input is not symmetric (A != A.T exactly)")

    if _is_inexact_dtype(arr):
        cls = getattr(_native, "SymmetricMatrix", None)
        if cls is None:
            raise ImportError("SymmetricMatrix is not available in the native extension")
        n = arr.shape[0]
        out = cls(n)
        for i in range(n):
            for j in range(i, n):
                out.set(i, j, float(arr[i, j]))
        out.properties["is_symmetric"] = True
        _track_matrix(out)
        return out

    # Integer/bool: preserve the exact dtype as a dense matrix and assert structure.
    out = matrix(arr)
    out.properties["is_symmetric"] = True
    return out


def antisymmetric(data: Any, *, rtol: float | None = None, atol: float | None = None) -> Any:
    """Create an anti-symmetric (skew-symmetric) matrix (A == -A.T) with validation.

    Storage depends on the input dtype:

    - float32/float16/float64 -> native `AntiSymmetricMatrix` (float64), which
      stores only the strict upper triangle (the diagonal is structurally zero).
    - integer/bool -> the corresponding dense matrix with `is_anti_symmetric=True`
      asserted (exact integer storage, no packing in R1).

    Validation always runs and also requires a zero diagonal. Complex input is
    rejected: the native AntiSymmetricMatrix is real-valued.
    """
    arr = _as_square_array(data, name="antisymmetric")
    if _np.iscomplexobj(arr):
        raise TypeError("antisymmetric() operates on real matrices; the native AntiSymmetricMatrix is float64")

    if _is_inexact_dtype(arr):
        tol_rtol = 1e-5 if rtol is None else rtol
        tol_atol = 1e-8 if atol is None else atol
        if not _np.allclose(arr, -arr.T, rtol=tol_rtol, atol=tol_atol):
            raise ValueError("antisymmetric() input is not anti-symmetric (A != -A.T within tolerance)")
        if not _np.allclose(_np.diag(arr), 0.0, rtol=tol_rtol, atol=tol_atol):
            raise ValueError("antisymmetric() input has a non-zero diagonal")
    else:
        if not _np.array_equal(arr, -arr.T):
            raise ValueError("antisymmetric() input is not anti-symmetric (A != -A.T exactly)")
        if not bool(_np.all(_np.diag(arr) == 0)):
            raise ValueError("antisymmetric() input has a non-zero diagonal")

    if _is_inexact_dtype(arr):
        cls = getattr(_native, "AntiSymmetricMatrix", None)
        if cls is None:
            raise ImportError("AntiSymmetricMatrix is not available in the native extension")
        n = arr.shape[0]
        out = cls(n)
        for i in range(n):
            for j in range(i + 1, n):
                out.set(i, j, float(arr[i, j]))
        out.properties["is_anti_symmetric"] = True
        out.properties["has_zero_diagonal"] = True
        _track_matrix(out)
        return out

    # Integer/bool: preserve the exact dtype as a dense matrix and assert structure.
    out = matrix(arr)
    out.properties["is_anti_symmetric"] = True
    out.properties["has_zero_diagonal"] = True
    return out


def diagonal(data: Any) -> Any:
    """Create a diagonal matrix from a 1D vector of entries or a 2D matrix.

    - A 1D vector supplies the diagonal entries.
    - A 2D square matrix uses its diagonal entries (off-diagonal values are ignored).

    Float input produces a native `DiagonalMatrix` (float64). Integer/bool input
    produces a dense matrix with `is_diagonal=True` asserted (exact storage).
    """
    if _np is None:
        raise RuntimeError("NumPy is required for diagonal construction")
    arr = _np.asarray(data)
    if arr.ndim == 1:
        entries = arr
        n = int(arr.shape[0])
    elif arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("diagonal() requires a square 2D matrix or a 1D vector")
        entries = _np.diag(arr)
        n = int(arr.shape[0])
    else:
        raise ValueError("diagonal() expects a 1D vector or a 2D square matrix")

    if _is_inexact_dtype(entries) and not _np.iscomplexobj(entries):
        cls = getattr(_native, "DiagonalMatrix", None)
        if cls is None:
            raise ImportError("DiagonalMatrix is not available in the native extension")
        out = cls(n)
        for i in range(n):
            out.set_diagonal(i, float(entries[i]))
        out.properties["is_diagonal"] = True
        _track_matrix(out)
        return out

    # Integer/bool (or complex): dense matrix with the diagonal structure asserted.
    if arr.ndim == 2:
        out = matrix(arr)
    else:
        out = matrix(_np.diag(entries))
    out.properties["is_diagonal"] = True
    return out


@contextmanager
def precision_mode(mode: str):
    """Temporarily override the thread-local promotion precision mode.

    Examples:
        with pycauset.precision_mode("highest"):
            c = a @ b

    Notes:
        This controls promotion decisions (storage dtype selection). It does not
        directly control accelerator internal compute dtype.
    """
    get_fn = getattr(_native, "get_precision_mode", None)
    set_fn = getattr(_native, "set_precision_mode", None)
    if get_fn is None or set_fn is None:
        raise RuntimeError("Native precision mode controls are not available")

    previous = str(get_fn())
    set_fn(mode)
    try:
        yield
    finally:
        set_fn(previous)


# Alias for IdentityMatrix
I = getattr(_native, "IdentityMatrix", None)  # noqa: E741 (public identity alias, NumPy convention)

# NumPy-compatible aliases: np.linalg.det -> det, np.linalg.matrix_rank -> rank shorthand.
det = determinant
rank = matrix_rank

# Lazy top-level visualization verbs (R2_VIZ): these exist for non-causet objects
# and as zero-import sugar; the primary citizens are the CausalSet.plot_* methods.
def plot_embedding(causet, **kwargs):
    from .vis import plot_embedding as _f
    return _f(causet, **kwargs)


def plot_hasse(causet, **kwargs):
    from .vis import plot_hasse as _f
    return _f(causet, **kwargs)


def plot_causal_matrix(causet, **kwargs):
    from .vis import plot_causal_matrix as _f
    return _f(causet, **kwargs)


def show(causet):
    """One-verb sugar: plot a causal set's embedding and open it in the browser."""
    return causet.plot_embedding().show()


# Python-level field API
from .field import ContinuumCorrelatedField as ContinuumCorrelatedField
from .field import CorrelatedField as CorrelatedField
from .field import Field as Field
from .field import State as State


def __getattr__(name):
    if name == "bool":
        return "bool"
    return getattr(_native, name)


# Native symbols that are internal machinery, not public API. They stay reachable
# as `pycauset.<name>` (via __getattr__) but are excluded from `__all__` so
# `from pycauset import *` does not surface implementation details.
_INTERNAL_NATIVE_EXPORTS = frozenset(
    {
        # Lazy expression-template machinery.
        "LazyMatrix",
        "lazy_add",
        "lazy_cos",
        "lazy_mul_scalar",
        "lazy_sin",
        "lazy_sub",
        # Memory / I/O / op-registry machinery.
        "MemoryGovernor",
        "IOAccelerator",
        "OpContract",
        "OpRegistry",
        # Native storage plumbing (public API is set_backing_dir / save / load).
        "get_storage_root",
        "set_storage_root",
        # Causet-generation internals (public API is pc.causet).
        "make_coordinates",
        "sprinkle",
    }
)

__all__ = [
    name for name in dir(_native)
    if not name.startswith("_") and name not in _INTERNAL_NATIVE_EXPORTS
]

# Add pure-Python facade symbols (and any optional native symbols) to __all__.
# IMPORTANT: Only include names that actually resolve; otherwise
# `from pycauset import *` can fail if a symbol is not exported by the native module.
_extra_exports = [
    "save",
    "load",
    "to_numpy",
    "set_export_max_bytes",
    "set_backing_dir",
    "get_memory_threshold",
    "set_memory_threshold",
    "get_io_streaming_threshold",
    "set_io_streaming_threshold",
    "last_io_trace",
    "clear_io_traces",
    "keep_temp_files",
    "seed",
    "bool",
    "matrix",
    "vector",
    "zeros",
    "ones",
    "empty",
    "load_npy",
    "load_npz",
    "save_npy",
    "save_npz",
    "convert_file",
    "causal_matrix",
    "TriangularBitMatrix",
    "TriangularMatrix",
    "matmul",
    "dot",
    "divide",
    "norm",
    "compute_k",
    "bitwise_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_nand",
    "bitwise_nor",
    "bitwise_xnor",
    "invert",
    "solve",
    "lstsq",
    "slogdet",
    "cond",
    "eigh",
    "eigvalsh",
    "solve_triangular",
    "lu",
    "cholesky",
    "svd",
    "pinv",
    "trace",
    "determinant",
    "det",
    "svdvals",
    "matrix_rank",
    "rank",
    "matrix_power",
    "outer",
    "cross",
    "vecdot",
    "eig",
    "eigvals",
    "eigvals_skew",
    "eig_skew",
    "eigvals_arnoldi",
    "identity",
    "diagonal",
    "symmetric",
    "antisymmetric",
    "precision_mode",
    "I",
    "causet",
    "CausalSet",
    "spacetime",
    "synthetic",
    "field",
    "MinkowskiDiamond",
    "MinkowskiCylinder",
    "MinkowskiBox",
    "MemoryHint",
    "AccessPattern",
    "Field",
    "CorrelatedField",
    "ContinuumCorrelatedField",
    "State",
    "plot_embedding",
    "plot_hasse",
    "plot_causal_matrix",
    "show",
]

for _name in _extra_exports:
    if _name in __all__:
        continue
    # pc.bool resolves via module __getattr__ (a module global named bool would
    # shadow the builtin), so it is recognized explicitly here.
    if _name in globals() or _name == "bool" or getattr(_native, _name, None) is not None:
        __all__.append(_name)

# Lazy ufunc patching is intentionally disabled: the C++ operator overloads handle
# NumPy interoperability directly (see R1_NUMPY_PLAN.md).

