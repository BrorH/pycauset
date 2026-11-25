"""Python package wrapper that augments the native pycauset extension."""
from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import os
import warnings

# Thread pools: OpenMP (our element-wise kernels), our OpenBLAS, and NumPy's own bundled
# OpenBLAS share the cores. By default each pool spins after a call (libgomp's active wait;
# OpenBLAS spins ~2^28 cycles), starving whichever pool runs next: a 1M-element dot took
# 1-4 ms instead of 0.1, NumPy's own matmul ran 1.7x slower next to us (FINDINGS F-012).
# Both libraries read these once when they load, so they are set before the native module is
# imported (NumPy's OpenBLAS may already be loaded and keep its setting). A user value wins.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "4")  # 2^4 cycles: the OpenBLAS minimum
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Any

from ._internal import coercion as _coercion
from ._internal import export_guard as _export_guard
from ._internal import factories as _factories
from ._internal import formatting as _formatting
from ._internal import io_observability as _io_observability
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


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------
# When True, pycauset prints diagnostic information to stderr: matrix sizes and
# storage placement (RAM vs disk), operation timings, sprinkling details, and
# file I/O. Toggle it at runtime with ``pycauset.debug_mode = True`` (default
# False). Every line is prefixed with ``[pycauset]``.
debug_mode: bool = False

import sys as _sys
import time as _time


def _debug(message: str) -> None:
    """Emit one diagnostic line when ``pycauset.debug_mode`` is True."""
    if debug_mode:
        print(f"[pycauset] {message}", file=_sys.stderr, flush=True)


def _debug_time(label: str):
    """Context manager that logs the wall-clock time of the wrapped block."""
    class _Timer:
        def __enter__(self):
            self._t0 = _time.perf_counter()
            return self

        def __exit__(self, *exc_info):
            _debug(f"{label}: {_time.perf_counter() - self._t0:.3f}s")
            return False

    return _Timer()


def _matrix_desc(matrix: Any) -> str:
    """A compact ``name shape dtype storage`` summary of a matrix/vector."""
    name = type(matrix).__name__
    try:
        rows = matrix.rows()
    except Exception:
        rows = None
    try:
        cols = matrix.cols()
    except Exception:
        cols = None

    shape = ""
    if rows is not None and cols is not None:
        shape = f"{rows}x{cols}"
    elif rows is not None:
        shape = f"{rows}"

    dtype = ""
    try:
        dtype = str(matrix.get_data_type())
    except Exception:
        pass

    backing = ""
    try:
        bf = matrix.get_backing_file()
    except Exception:
        bf = None
    if bf is not None:
        backing = "RAM" if bf == ":memory:" else f"disk ({bf})"

    return " ".join(part for part in (name, shape, dtype, backing) if part)


_native_mod.configure_windows_dll_search_paths(package_dir=os.path.dirname(__file__))
_native = _native_mod.import_native_extension(package=__name__)

# One-shot native configuration (OpenBLAS thread default).
from ._internal.native_setup import configure_openblas_threads

configure_openblas_threads()

# Import modules that depend on the native extension after it is loaded.
from . import field as field
from . import spacetime as spacetime
from . import synthetic as synthetic
from ._storage import cleanup_storage, set_temporary_file
from .causet import CausalSet


def _debug_resolve_promotion(
    op: str,
    a_dtype: str,
    b_dtype: str,
) -> dict[str, object]:
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

# One class per structure; the dtype is an attribute (A.dtype). See review/RESTRUCTURE.md 3.1 item 4.
DenseMatrix = _native.DenseMatrix
TriangularMatrix = _native.TriangularMatrix
DiagonalMatrix = _native.DiagonalMatrix
IdentityMatrix = _native.IdentityMatrix
SymmetricMatrix = _native.SymmetricMatrix
AntiSymmetricMatrix = _native.AntiSymmetricMatrix
DenseVector = _native.DenseVector
UnitVector = _native.UnitVector

_MATRIX_CLASSES = (
    DenseMatrix,
    TriangularMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    SymmetricMatrix,
    AntiSymmetricMatrix,
)
_VECTOR_CLASSES = (DenseVector, UnitVector)



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

    _debug(f"set_backing_dir -> {new_root}")
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
    _debug(f"set_memory_threshold -> {int(target)} bytes")
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
    _debug(f"allocated {_matrix_desc(instance)}")


def _register_cleanup() -> None:
    _runtime.register_cleanup(keep_temp_files_getter=lambda: keep_temp_files)


_register_cleanup()

# NumPy export safety hooks for native matrices/vectors
_MatrixBaseType = getattr(_native, "MatrixBase", None)
_VectorBaseType = getattr(_native, "VectorBase", None)


def _guarded_array_export(self: Any, dtype: Any = None, copy: Any = None) -> Any:
    # NumPy 2 protocol: copy=False means "never copy". PyCauset storage can't be shared with
    # NumPy, so that request raises; otherwise one native copy (the size guard applies).
    if copy is False:
        raise ValueError("a pycauset matrix can't be exported to NumPy without a copy")
    return _export_guard.export_to_numpy(self, allow_huge=False, dtype=dtype, copy=True)


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
    *_MATRIX_CLASSES,
    *_VECTOR_CLASSES,
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
    """

    return _export_guard.export_to_numpy(obj, allow_huge=allow_huge, dtype=dtype, copy=copy)

_PERSISTENCE_DEPS = _SimpleNamespace(CausalSet=CausalSet, native=_native)


def save(obj: Any, path: str | Path) -> None:
    _debug(f"save {_matrix_desc(obj)} -> {path}")
    with _debug_time(f"save -> {path}"):
        return _persistence.save(obj, path, deps=_PERSISTENCE_DEPS)


def load(path: str | Path) -> Any:
    _debug(f"load {path}")
    with _debug_time(f"load <- {path}"):
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
_linalg_cache.patch_inverse(native=_native, classes=list(_MATRIX_CLASSES))

_formatting.configure(np_module=_np, edge_items=4)


def _mark_temporary_if_auto(matrix: Any) -> None:
    _runtime.mark_temporary_if_auto(matrix)


def _coerce_general_matrix(candidate: Any) -> tuple[int, list[list[Any]]]:
    return _coercion.coerce_general_matrix(candidate, np_module=_np)

_matrix_api.configure(np_module=_np, native=_native, track_matrix=_track_matrix)
_matrix_api.register_native_matrices(*_MATRIX_CLASSES)

_patching.apply_native_storage_patches(
    matrix_classes=[
        *((c, "backing_file") for c in _MATRIX_CLASSES),
    ],
    vector_classes=[
        *((c, "backing_file") for c in _VECTOR_CLASSES),
    ],
    track_matrix=_track_matrix,
    mark_temporary_if_auto=_mark_temporary_if_auto,
)


# Debug-mode timing for the `@` operator, which calls the native
# `MatrixBase.__matmul__` directly (bypassing the `pc.matmul` wrapper above).
def _wrap_matmul_for_debug(cls: Any) -> None:
    original = getattr(cls, "__matmul__", None)
    if original is None:
        return

    def _timed(self: Any, other: Any) -> Any:
        _debug(f"matmul {_matrix_desc(self)} @ {_matrix_desc(other)}")
        with _debug_time("matmul @"):
            return original(self, other)

    try:
        cls.__matmul__ = _timed
    except Exception:
        pass


for _mcls in (
    *_MATRIX_CLASSES,
):
    _wrap_matmul_for_debug(_mcls)

_properties.apply_properties_patches(
    classes=[
        *_MATRIX_CLASSES,
        *_VECTOR_CLASSES,
    ]
)

_properties.apply_properties_mutation_patches(
    classes=[
        *_MATRIX_CLASSES,
        *_VECTOR_CLASSES,
    ]
)

_properties.apply_properties_view_patches(
    classes=[
        *_MATRIX_CLASSES,
        *_VECTOR_CLASSES,
    ],
    track_matrix=_track_matrix,
    mark_temporary_if_auto=_mark_temporary_if_auto,
)

_properties.apply_properties_operator_patches(
    classes=[
        *_MATRIX_CLASSES,
    ]
)

_properties.apply_properties_arithmetic_patches(
    classes=[
        *_MATRIX_CLASSES,
        *_VECTOR_CLASSES,
    ]
)



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


def _allocate_dense_matrix_by_shape(rows: int, cols: int, *, dtype: Any, kwargs: dict[str, Any]) -> Any:
    dtype_norm = _normalize_dtype(dtype, np_module=_np)
    if dtype_norm is None:
        raise TypeError("Unsupported dtype")
    return DenseMatrix(rows, cols, dtype=dtype_norm, **kwargs)


def _allocate_like(shape: Any, kind: str, dtype: Any, kwargs: dict[str, Any]) -> Any:
    """Allocate a vector/matrix of the given dtype and apply the fill pattern.

    kind is one of "zeros", "ones", or "empty" (empty applies no fill).
    """
    if isinstance(shape, int):
        vec = _factories.vector_factory(
            int(shape),
            dtype=dtype,
            np_module=_np,
            native=_native,
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


def zeros(shape: Any, *, dtype: Any = "float64", **kwargs: Any) -> Any:
    """A vector (int shape) or matrix ((rows, cols) shape) filled with zeros.

    dtype defaults to float64, as in NumPy; pass dtype= for any other element type.
    """
    return _allocate_like(shape, "zeros", dtype if dtype is not None else "float64", kwargs)


def ones(shape: Any, *, dtype: Any = "float64", **kwargs: Any) -> Any:
    """A vector (int shape) or matrix ((rows, cols) shape) filled with ones.

    dtype defaults to float64, as in NumPy; pass dtype= for any other element type.
    """
    return _allocate_like(shape, "ones", dtype if dtype is not None else "float64", kwargs)


def empty(shape: Any, *, dtype: Any = "float64", **kwargs: Any) -> Any:
    """A vector (int shape) or matrix ((rows, cols) shape) without initializing the elements.

    dtype defaults to float64, as in NumPy; pass dtype= for any other element type.
    """
    return _allocate_like(shape, "empty", dtype if dtype is not None else "float64", kwargs)


def causal_matrix(source: Any, populate: bool = True, **kwargs: Any) -> Any:
    """Causal matrix: a strictly upper-triangular TriangularMatrix of dtype 'bool'.

    `source` is a size n (random with p=0.5 when populate=True, else empty) or 2-D data whose
    strict upper triangle is used.
    """
    return _factories.causal_matrix_factory(
        source,
        populate=populate,
        np_module=_np,
        TriangularMatrix=TriangularMatrix,
        coerce_general_matrix=_coerce_general_matrix,
        kwargs=kwargs,
    )


causal_matrix.random = TriangularMatrix.random


def causet(
    *,
    n: int = None,
    density: float = None,
    spacetime=None,
    seed: int | str | None = None,
    matrix=None,
) -> CausalSet:
    """Lower-case convenience factory that returns a CausalSet."""
    st_name = type(spacetime).__name__ if spacetime is not None else "default"
    if matrix is not None:
        _debug(f"causet from matrix (n={n})")
    else:
        _debug(f"sprinkle n={n} density={density} spacetime={st_name} seed={seed}")
    with _debug_time("causet"):
        return CausalSet(n=n, density=density, spacetime=spacetime, seed=seed, matrix=matrix)

_OPS_DEPS = _ops.OpsDeps(
    native=_native,
    np_module=_np,
    Matrix=_matrix_api.Matrix,
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
    _debug(f"matmul {_matrix_desc(a)} @ {_matrix_desc(b)}")
    with _debug_time("matmul"):
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


def sum(x: Any) -> int | float | complex:
    """Return the sum of all elements in a vector or matrix, like ``numpy.sum``.

    Integer and bool inputs give an exact Python ``int`` (no wrap; ``OverflowError`` beyond
    128 bits), real floats a ``float``, complex inputs a ``complex``. Float sums are
    deterministic (fixed summation order, independent of the thread count).
    """
    props = getattr(x, "properties", None)
    if props is not None:
        cached = props.get("sum")
        if cached is not None:
            return cached
    out = _native.sum(x)
    if props is not None:
        props["sum"] = out
    return out

def compute_k(matrix: Any, a: float):
    """
    Compute K = C(aI + C)^-1.
    
    Args:
        matrix: The causal matrix C (bit TriangularMatrix).
        a: The scalar a.
        
    Returns:
        A float64 TriangularMatrix representing K.
    """
    return _ops.compute_k(matrix, a, deps=_OPS_DEPS)


def bitwise_not(matrix: Any) -> Any:
    """
    Compute the bitwise inversion (NOT) of a matrix.
    
    Args:
        matrix: The matrix to invert. Must have an integer or bit dtype.
        
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
    _debug(f"invert {_matrix_desc(matrix)}")
    with _debug_time("invert"):
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
    _debug(f"solve {_matrix_desc(a)} with RHS {_matrix_desc(b)}")
    with _debug_time("solve"):
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
    if isinstance(x, bool):
        raise TypeError("identity(x): bool is not a valid size")
    if isinstance(x, int):
        return IdentityMatrix(x)
    if isinstance(x, _native.MatrixBase):
        return IdentityMatrix(x.rows(), x.cols())
    if isinstance(x, _native.VectorBase):
        return IdentityMatrix(len(x))
    if _is_sequence_like(x) and len(x) == 2 and all(isinstance(d, int) and not isinstance(d, bool) for d in x):
        return IdentityMatrix(int(x[0]), int(x[1]))
    raise TypeError("identity(x) expects an int, [rows, cols], or a matrix/vector")


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


_HIDDEN_NATIVE_NAMES = frozenset(
    {
        # Spacetimes live under pycauset.spacetime (public API), never at the
        # top level. The native C++ classes stay reachable as pycauset._native.*.
        "MinkowskiDiamond",
        "MinkowskiCylinder",
        "MinkowskiBox",
        "CausalSpacetime",
    }
)


def __getattr__(name):
    if name == "bool":
        return "bool"
    if name in _HIDDEN_NATIVE_NAMES:
        raise AttributeError(
            f"module 'pycauset' has no attribute {name!r}; "
            f"spacetimes live in pycauset.spacetime"
        )
    return getattr(_native, name)


# Native symbols that are internal machinery, not public API. They stay reachable
# as `pycauset.<name>` (via __getattr__) but are excluded from `__all__` so
# `from pycauset import *` does not surface implementation details.
_INTERNAL_NATIVE_EXPORTS = frozenset(
    {
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
    if not name.startswith("_")
    and name not in _INTERNAL_NATIVE_EXPORTS
    and name not in _HIDDEN_NATIVE_NAMES
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
    "DenseMatrix",
    "TriangularMatrix",
    "DiagonalMatrix",
    "IdentityMatrix",
    "SymmetricMatrix",
    "AntiSymmetricMatrix",
    "DenseVector",
    "UnitVector",
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
    "I",
    "causet",
    "CausalSet",
    "spacetime",
    "synthetic",
    "field",
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
    "debug_mode",
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

