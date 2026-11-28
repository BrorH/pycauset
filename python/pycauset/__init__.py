"""Python package wrapper that augments the native pycauset extension."""
from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import contextlib as _contextlib
import operator as _operator
import os
import warnings

# Thread pools: OpenMP (our element-wise kernels), our OpenBLAS, and NumPy's own bundled
# OpenBLAS share the cores. By default each pool spins after a call (libgomp's active wait;
# OpenBLAS spins ~2^28 cycles), starving whichever pool runs next: a 1M-element dot took
# 1-4 ms instead of 0.1, NumPy's own matmul ran 1.7x slower next to us (FINDINGS F-012).
# Both libraries read these once when they load, so they are set before the native module is
# imported (NumPy's OpenBLAS may already be loaded and keep its setting). A user value wins.
import sys as _sys

# NumPy imported first, with no timeout set: its OpenBLAS threads already spin after every call
# and PyCauset's BLAS operations next to them ran up to 50x slower (norm of 1024 x 1024: 32 ms
# instead of 0.6 ms; F-096). Nothing can be changed in NumPy's library now. The Performance
# Guide says how to avoid it; there is no import-time warning (it broke `-W error` for the usual
# import order, review 3.8 C-13). `_numpy_imported_first` records the case for diagnostics.
_numpy_imported_first = "numpy" in _sys.modules and "OPENBLAS_THREAD_TIMEOUT" not in os.environ
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
# 2^18 cycles (~0.1 ms): long enough to keep threads awake between the many small BLAS calls
# inside one LAPACK factorization (at 2^4, qr / eigh / svd ran 2x slower: F-108), far shorter
# than the default 2^28 (~0.1 s) whose spinning starved the other pool (8 ms -> 28 ms).
os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "18")

from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Any

from ._internal import coercion as _coercion
from ._internal import export_guard as _export_guard
from ._internal import factories as _factories
from ._internal import formatting as _formatting
from ._internal import io_observability as _io_observability
from ._internal import dtypes as _dtypes
from ._internal import matrix_api as _matrix_api
from ._internal import native as _native_mod
from ._internal import ops as _ops
from ._internal.thunks import StaleThunkError as StaleThunkError
from ._internal import persistence as _persistence
from ._internal import runtime as _runtime_mod
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

import numpy as _np  # a dependency (pyproject); the native module needs it too

# Public dtype tokens (NumPy's names): the constructors take them, and a matrix's `.dtype`
# compares equal to them (and to NumPy's own dtypes). `int_` is NumPy's (int64; PyCauset's
# default for Python ints is int32, F-103); `uint` and `float_` meant something else than
# NumPy's (`float_` is gone from NumPy 2) and are gone (review 3.8 C-17).
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
int_ = "int64"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
uint64 = "uint64"
float16 = "float16"
float32 = "float32"
float64 = "float64"
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
# When True, pycauset prints one line to stderr per matmul, inverse, solve, save and load
# (the operands: class, shape, dtype, RAM or file; and the wall-clock time), and per change of
# the memory threshold or backing directory (debug_mode.md). Toggle it at runtime
# with ``pycauset.debug_mode = True`` (default False). Every line is prefixed with
# ``[pycauset]``.
debug_mode: bool = False

import time as _time


def _debug(message: str) -> None:
    """Emit one diagnostic line when ``pycauset.debug_mode`` is True."""
    if debug_mode:
        print(f"[pycauset] {message}", file=_sys.stderr, flush=True)


def _debug_time(label: str):
    """Context manager that logs the wall-clock time of the wrapped block (a no-op unless
    ``debug_mode``: the per-call cost of the wrappers is part of every operation, N.6)."""
    if not debug_mode:
        return _contextlib.nullcontext()

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
    # (It called a get_data_type that does not exist, so no line had a dtype, and vectors
    # had no shape; review 3.8 C-21.)
    shape = "x".join(str(d) for d in getattr(matrix, "shape", ()) or ())
    dtype = str(getattr(matrix, "dtype", "") or "")

    backing = ""
    bf = matrix.get_backing_file() if hasattr(matrix, "get_backing_file") else None
    if bf:
        backing = "RAM" if bf == ":memory:" else f"disk ({bf})"

    return " ".join(part for part in (name, shape, dtype, backing) if part)


_native_mod.configure_windows_dll_search_paths(package_dir=os.path.dirname(__file__))
_native = _native_mod.import_native_extension(package=__name__)

from ._storage import cleanup_storage

# Causal sets (R1: sprinkling into the Minkowski diamond, the order methods, synthetic
# orders). Fields, plots and the rest of the spacetime library are parked in
# experimental/physics.
from . import spacetime as spacetime
from . import synthetic as synthetic
from .causet import CausalSet


# Internal/test helper: the promotion resolver's decision for (op, a_dtype, b_dtype).
_debug_resolve_promotion = _native._debug_resolve_promotion


# Internal/test helpers: the last dispatched kernel tag (thread-local), and clearing it.
_debug_last_kernel_trace = _native._debug_last_kernel_trace
_debug_clear_kernel_trace = _native._debug_clear_kernel_trace



# One class per structure; the dtype is an attribute (A.dtype). See review/RESTRUCTURE.md 3.1 item 4.
DenseMatrix = _native.DenseMatrix
TriangularMatrix = _native.TriangularMatrix
DiagonalMatrix = _native.DiagonalMatrix
IdentityMatrix = _native.IdentityMatrix
SymmetricMatrix = _native.SymmetricMatrix
AntiSymmetricMatrix = _native.AntiSymmetricMatrix
HermitianMatrix = _native.HermitianMatrix
SkewHermitianMatrix = _native.SkewHermitianMatrix
DenseVector = _native.DenseVector
UnitVector = _native.UnitVector

_MATRIX_CLASSES = (
    DenseMatrix,
    TriangularMatrix,
    DiagonalMatrix,
    IdentityMatrix,
    SymmetricMatrix,
    AntiSymmetricMatrix,
    HermitianMatrix,
    SkewHermitianMatrix,
)
_VECTOR_CLASSES = (DenseVector, UnitVector)



_runtime = _runtime_mod.Runtime(
    cleanup_storage=cleanup_storage,
)

_DEFAULT_MEMORY_THRESHOLD = _native.get_memory_threshold()
_IO_OBS = _io_observability.IOObservability(_native)


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
    except OSError:  # the old root cannot be created or resolved: nothing lives there
        old_root = None

    # Native code allocates backing files; configure it directly.
    _native._set_storage_root(str(new_root))

    if old_root is not None and new_root != old_root and _runtime.has_own_temp_files(old_root):
        warnings.warn(
            "Changing the backing directory after matrices have been created can leave "
            "some objects backed by the old directory and may complicate cleanup. "
            "Prefer calling pycauset.set_backing_dir(...) immediately after import.",
            PyCausetStorageWarning,
            stacklevel=2,
        )

    _debug(f"set_backing_dir -> {new_root}")
    return _runtime.set_storage_root(new_root)


def get_memory_threshold() -> int:
    """Objects up to this many bytes live in RAM; larger ones are file-backed."""
    return int(_native.get_memory_threshold())


def _byte_count(limit: Any, what: str) -> int:
    """A non-negative integer number of bytes; anything else raises before a setting changes
    (0.5 became 0 and "10" a pybind11 RuntimeError after the Python side had changed, review
    3.8 C-16)."""
    import operator

    try:
        n = operator.index(limit)
    except TypeError:
        raise TypeError(f"{what}: expected an integer number of bytes or None, got {limit!r}") from None
    if n < 0:
        raise ValueError(f"{what}: must be >= 0, got {n}")
    if n >= 2**64:
        raise ValueError(f"{what}: must be below 2**64, got {n}")
    return n


def set_memory_threshold(limit: int | None) -> int:
    """Objects up to `limit` bytes live in RAM, larger ones are file-backed; `None` restores
    the threshold in effect at import (the engine's default). Returns the threshold now in effect."""
    target = _DEFAULT_MEMORY_THRESHOLD if limit is None else _byte_count(limit, "set_memory_threshold")
    _native.set_memory_threshold(target)
    _debug(f"set_memory_threshold -> {target} bytes")
    return int(_native.get_memory_threshold())


def last_io_trace(op: str | None = None) -> dict[str, Any] | None:
    """Return the most recent IO trace (optionally filtered by op name)."""

    return _IO_OBS.last(op)


def clear_io_traces() -> None:
    """Clear recorded IO traces (observability/debug only)."""

    _IO_OBS.clear()


# One default backing directory for Python and the engine, fixed at import (the engine took the
# working directory at its first file-backed object, so after os.chdir the two disagreed).
# Not created until a file-backed object needs it. Leftovers of crashed runs are swept if it
# exists.
_native._set_storage_root(str(_runtime.storage_root()))
_runtime.initial_cleanup()
def _register_cleanup() -> None:
    _runtime.register_cleanup()


_register_cleanup()

# NumPy export safety hooks for native matrices/vectors
_MatrixBaseType = _native.MatrixBase
_VectorBaseType = _native.VectorBase


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
    native_type.__array__ = _guarded_array_export
    # Disallow ufuncs so standard ops take precedence (forcing __radd__ over NumPy's buffer path)
    native_type.__array_ufunc__ = None
    native_type.__array_priority__ = 1e6
    setattr(native_type, "__pycauset_array_wrapped__", True)


def _unpickle(arr: Any, structure: str, has_diagonal: bool, transposed: bool) -> Any:
    """Rebuild a pickled matrix or vector: its values, then its structure class."""
    if structure == "unit_vector":
        k = int(_np.flatnonzero(arr)[0]) if _np.any(arr) else 0
        out = UnitVector(arr.shape[0], k, dtype=_matrix_api.dtype_or_raise(arr.dtype))
        if arr.size and arr[k] != 1:
            out = out * arr[k].item()
    elif structure == "identity":  # no stored values: the identity times its factor
        out = IdentityMatrix(arr.shape[0], arr.shape[1], dtype=_matrix_api.dtype_or_raise(arr.dtype))
        if arr.size and arr[0, 0] != 1:
            out = out * arr[0, 0].item()
    else:
        out = asarray(arr)
        if structure not in ("dense", "vector"):
            out = _native._pack_as(out, structure, has_diagonal)
    return out.T if transposed else out


def _reduce(self: Any) -> Any:
    """Pickling (multiprocessing, joblib, copy.deepcopy): the values (an explicit copy, so the
    export ceiling does not apply) and the structure class (review 3.8 C-19)."""
    structure = _native._structure_of(self)
    hd = getattr(self, "has_diagonal", False)
    has_diagonal = bool(hd() if callable(hd) else hd)
    transposed = isinstance(self, _native.VectorBase) and self.is_transposed()
    arr = to_numpy(self.T if transposed else self, allow_huge=True)
    return (_unpickle, (arr, structure, has_diagonal, transposed))


# The native __setitem__ takes every key and value itself (NumPy scalars included; cast errors
# are TypeErrors through the translator). A Python wrapper re-dispatched each write and tripled
# its cost (review 3.8 C-14).
for _cls in (_MatrixBaseType, _VectorBaseType, *_MATRIX_CLASSES, *_VECTOR_CLASSES):
    _install_array_export_hook(_cls)
    _cls.__reduce__ = _reduce


# `.dtype` is a DType: the PyCauset name, equal to NumPy's dtype too (dtypes.DType). The bases
# define the native property; the structure classes inherit the wrapped one.
def _wrap_dtype(cls: Any) -> None:
    native = cls.__dict__["dtype"]
    cls.dtype = property(lambda self: _dtypes.DType(native.__get__(self)), doc=native.__doc__)


for _cls in (_MatrixBaseType, _VectorBaseType):
    _wrap_dtype(_cls)


def set_export_max_bytes(limit: int | None) -> None:
    """Set the materialization ceiling (bytes) for NumPy exports.

    Exports larger than it raise unless `allow_huge=True`. None removes the ceiling: then an
    export raises only when it would not fit in free RAM.
    """

    if limit is not None:
        limit = _byte_count(limit, "set_export_max_bytes")
    _native._set_numpy_export_max_bytes(limit)  # the engine first: both or neither change
    _export_guard.set_max_bytes(limit)


def to_numpy(obj: Any, *, allow_huge: bool = False, dtype: Any = None, copy: bool = True) -> Any:
    """Convert a PyCauset object to a new NumPy array under the export size guard.

    An export larger than free RAM or the `set_export_max_bytes` ceiling raises unless
    `allow_huge=True` (RAM- and file-backed objects alike). `dtype` converts the result;
    `copy=False` raises (every export is a copy).
    """

    return _export_guard.export_to_numpy(obj, allow_huge=allow_huge, dtype=dtype, copy=copy)

_PERSISTENCE_DEPS = _SimpleNamespace(CausalSet=CausalSet, native=_native)


def save(obj: Any, path: str | Path) -> None:
    if debug_mode:
        _debug(f"save {_matrix_desc(obj)} -> {path}")
    with _debug_time(f"save -> {path}"):
        return _persistence.save(obj, path, deps=_PERSISTENCE_DEPS)


def load(path: str | Path) -> Any:
    _debug(f"load {path}")
    with _debug_time(f"load <- {path}"):
        return _persistence.load(path, deps=_PERSISTENCE_DEPS)


def _array_to_pycauset(arr: Any) -> Any:
    if not isinstance(arr, _np.ndarray):
        arr = _np.array(arr)

    if arr.ndim == 1:
        return vector(arr)
    if arr.ndim == 2:
        return matrix(arr)
    raise ValueError("Only 1D or 2D arrays can be imported")


def load_npy(path: str | Path) -> Any:
    # Memory-mapped: the import reads from the page cache and goes to a file when the array
    # does not fit (it read the whole file into RAM first, then copied it, review 3.9 C-12).
    arr = _np.load(path, mmap_mode="r")
    return _array_to_pycauset(arr)


def load_npz(path: str | Path, *, key: str | None = None) -> Any:
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
    """Write `obj` as a .npy file at exactly `path` (through a file handle: np.save appended
    ".npy" to a path without it, and the returned path did not exist, review 3.8 C-15)."""
    arr = to_numpy(obj, allow_huge=allow_huge, dtype=dtype)
    with open(path, "wb") as f:
        _np.save(f, arr)
    return Path(path)


def save_npz(
    obj: Any,
    path: str | Path,
    *,
    allow_huge: bool = False,
    dtype: Any = None,
    key: str = "array",
) -> Path:
    arr = to_numpy(obj, allow_huge=allow_huge, dtype=dtype)
    with open(path, "wb") as f:  # at exactly `path` (np.savez appended ".npz")
        _np.savez(f, **{key: arr})
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
    - npz imports default to the alphabetically first key unless npz_key is provided.
    - dtype converts .npy / .npz exports; with a .pycauset destination it raises.
    - Exports respect the NumPy materialization guard via allow_huge.
    """

    src = Path(src_path)
    dst = Path(dst_path)

    def _infer_format(p: Path, explicit: str | None, which: str) -> str:
        if explicit:
            return explicit.lower()
        suf = p.suffix.lower().lstrip(".")
        if suf in {"pycauset", "npy", "npz"}:
            return suf
        hint = "; provide dst_format" if which == "destination" else " (.pycauset, .npy or .npz)"
        raise ValueError(f"Could not infer the {which} format from {p}{hint}")

    src_fmt = _infer_format(src, None, "source")
    dst_fmt = _infer_format(dst, dst_format, "destination")
    if dst_fmt == "pycauset" and dtype is not None:
        # It was ignored silently (review 3.8 C-15): a .pycauset file keeps the source's dtype.
        raise TypeError("convert_file: dtype applies to .npy / .npz destinations only; cast first, "
                        "e.g. pc.save(pc.asarray(pc.load(src), dtype=...), dst)")

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
            close()  # drops the storage; cannot fail

    return dst


_native.MatrixBase.save = save
_native.MatrixBase.conjugate = _native.MatrixBase.conj  # NumPy's two names, one code path
_native.VectorBase.save = save

_formatting.configure(np_module=_np, edge_items=4)


_matrix_api.configure(np_module=_np, native=_native)
_matrix_api.register_native_matrices(*_MATRIX_CLASSES)



# Debug-mode timing for the `@` operator, which calls the native
# `MatrixBase.__matmul__` directly (bypassing the `pc.matmul` wrapper above).
def _wrap_matmul_for_debug(cls: Any) -> None:
    original = getattr(cls, "__matmul__", None)
    if original is None:
        return

    def _timed(self: Any, other: Any) -> Any:
        if not debug_mode:
            return original(self, other)
        _debug(f"matmul {_matrix_desc(self)} @ {_matrix_desc(other)}")
        with _debug_time("matmul @"):
            return original(self, other)

    cls.__matmul__ = _timed


for _mcls in (
    *_MATRIX_CLASSES,
):
    _wrap_matmul_for_debug(_mcls)





def _is_scalar_0d(value: Any) -> bool:
    # Scalar/0D inputs are explicitly unsupported for matrix/vector factories.
    # (Use zeros/ones/empty for allocation.)
    if isinstance(value, (bool, int, float, complex)):
        return True
    return isinstance(value, _np.generic) or (isinstance(value, _np.ndarray) and value.ndim == 0)


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
    `max_in_ram_bytes`: a vector larger than this goes to a file (vector.md).
    """
    max_in_ram_bytes = kwargs.pop("max_in_ram_bytes", max_in_ram_bytes)
    if _is_scalar_0d(source):
        raise TypeError(
            "vector(...) constructs from data; shape allocation uses zeros/ones/empty. "
            "Scalars/0D are not supported."
        )

    _VectorBase = _native.VectorBase
    if _VectorBase is not None and isinstance(source, _VectorBase):
        if dtype is not None or kwargs:
            raise TypeError(
                "vector(...) does not accept dtype/kwargs when source is already a vector. "
                "Pass data instead."
            )
        return source

    _MatrixBase = _native.MatrixBase
    if _MatrixBase is not None and isinstance(source, _MatrixBase):
        raise TypeError("vector(...) expects 1D input; use matrix(...) for 2D")

    if isinstance(source, _np.ndarray) and source.ndim != 1:
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

    # NumPy arrays first: the common case (2.9 us per call when checked last, F-096).
    if isinstance(source, _np.ndarray):
        if source.ndim == 2:
            if dtype is None:
                return _matrix_api.Matrix(source, **kwargs)
            return _matrix_api.Matrix(source, dtype=dtype, **kwargs)
        if source.ndim == 1:
            return vector(source, dtype=dtype, **kwargs)
        raise TypeError("matrix(...) expects 1D or 2D input")

    # Already a matrix or vector (engine objects and the internal block / view / thunk
    # matrices): returned as they are.
    from ._internal.blockmatrix import BlockMatrix
    from ._internal.submatrix_view import SubmatrixView
    from ._internal.thunks import ThunkBlock

    if isinstance(source, (_native.MatrixBase, _native.VectorBase, BlockMatrix, SubmatrixView, ThunkBlock)):
        if dtype is not None or kwargs:
            what = "vector" if isinstance(source, _native.VectorBase) else "matrix"
            raise TypeError(
                f"matrix(...) does not accept dtype/kwargs when source is already a {what}. "
                "Pass data instead."
            )
        return source

    if _is_sequence_like(source):
        if len(source) == 0:
            return vector(source, dtype=dtype, **kwargs)  # np.asarray([]): an empty float64 1-D array
        first = source[0]
        if _is_sequence_like(first):
            # Reject higher-rank nested sequences deterministically.
            if len(first) > 0 and _is_sequence_like(first[0]):
                raise TypeError("matrix(...) expects 1D or 2D input")

            # Phase F integration: disambiguate block-grid construction.
            # - All elements are matrices -> BlockMatrix
            # - No elements are matrices -> dense data constructor
            # - Mixed matrices and scalars -> error
            native_matrix_base = _native.MatrixBase

            def _is_matrix_obj(x: Any) -> bool:
                if native_matrix_base is not None and isinstance(x, native_matrix_base):
                    return True
                if isinstance(x, BlockMatrix):
                    return True
                if isinstance(x, SubmatrixView):
                    return True
                if isinstance(x, ThunkBlock):
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

            if all_matrix and any_matrix:  # (vacuously all for [[]], which is a 1 x 0 matrix)
                if dtype is not None or kwargs:
                    raise TypeError("matrix(block_grid) does not accept dtype/kwargs")
                return BlockMatrix(source)

            if dtype is None:
                return _matrix_api.Matrix(source, **kwargs)
            return _matrix_api.Matrix(source, dtype=dtype, **kwargs)
        return vector(source, dtype=dtype, **kwargs)

    raise TypeError("matrix(...) expects 1D or 2D input")


def _allocate_like(shape: Any, kind: str, dtype: Any, kwargs: dict[str, Any]) -> Any:
    """A vector (int or 1-tuple shape) or matrix ((rows, cols)) of `dtype`; kind is "zeros",
    "ones" or "empty". New storage is zero (calloc'd RAM, or a freshly sized file), so only
    "ones" fills. `storage` / `max_in_ram_bytes` place it, as for `matrix`; other keywords raise."""
    kw = dict(kwargs)
    storage = kw.pop("storage", None)
    limit = kw.pop("max_in_ram_bytes", None)
    if kw:
        raise TypeError(f"{kind}() got unexpected keyword arguments: {sorted(kw)}")
    if not isinstance(shape, bool) and hasattr(shape, "__index__"):  # an int, np.int64, ...
        shape = (shape,)
    if not _is_sequence_like(shape):
        raise TypeError("shape must be an int or a tuple")
    shape = tuple(_operator.index(d) for d in shape)  # NumPy's sizes (a float is a TypeError)
    if len(shape) not in (1, 2):
        raise ValueError("shape must be 1D or 2D")
    out = _matrix_api.allocate(shape, dtype, storage=storage, max_in_ram_bytes=limit)
    if kind == "ones":
        out.fill(1)
    return out


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
        native=_native,
        square_source=_square_source,
        kwargs=kwargs,
    )


causal_matrix.random = TriangularMatrix.random


def causet(
    *,
    n: int | None = None,
    density: float | None = None,
    spacetime=None,
    seed: int | str | None = None,
    matrix=None,
) -> CausalSet:
    """A causal set: sprinkled (`n` or `density`, `spacetime`, `seed`) or from a causal matrix."""
    return CausalSet(n=n, density=density, spacetime=spacetime, seed=seed, matrix=matrix)


_OPS_DEPS = _ops.OpsDeps(
    native=_native,
    np_module=_np,
    Matrix=_matrix_api.Matrix,
    warnings_module=warnings,
    io_observer=_IO_OBS,
)

def matmul(a: Any, b: Any) -> Any:
    """Matrix product ``a @ b`` (NumPy's ``matmul``); the engine picks the kernel by structure and dtype."""
    # (No debug line here: the engine call goes through `a @ b`, whose wrapper logs it; both
    # logged, review 3.8 C-21.)
    return _ops.matmul(a, b, deps=_OPS_DEPS)


def dot(a: Any, b: Any) -> Any:
    """Compute a dot product or matrix product, matching NumPy's `np.dot`.

    Semantics:
    - vector . vector: scalar inner product (no implicit conjugation).
    - matrix . matrix: matrix multiplication.
    - matrix . vector / vector . matrix: matrix-vector product.

    Matrix cases route through `matmul`.
    """
    native_vector_base = _native.VectorBase
    a_is_vector = native_vector_base is not None and isinstance(a, native_vector_base)
    b_is_vector = native_vector_base is not None and isinstance(b, native_vector_base)

    if a_is_vector and b_is_vector:
        fn = getattr(a, "dot", None)
        if fn is None:
            raise TypeError("pycauset.dot: expected a vector-like object with a .dot(other) method")
        return fn(b)

    return _ops.matmul(a, b, deps=_OPS_DEPS)  # vector @ matrix is 1-D, as np.dot


def asarray(a: Any, dtype: Any = None) -> Any:
    """A pycauset vector (1-D) or matrix (2-D) of an array-like, as NumPy's `asarray`: a pycauset
    object is returned as it is (no copy) when no other dtype is asked for; NumPy arrays, lists and
    tuples are imported (one import path with `matrix` / `vector`)."""
    if isinstance(a, (_native.MatrixBase, _native.VectorBase)) and (dtype is None or _matrix_api.dtype_or_raise(dtype) == a.dtype):
        return a
    arr = _np.asarray(a)
    if arr.ndim not in (1, 2):
        raise ValueError(f"asarray: expected 1-D or 2-D input, got {arr.ndim}-D")
    return _matrix_api.from_array(arr, _matrix_api.dtype_or_raise(dtype))


def add(a: Any, b: Any) -> Any:
    """Element-wise `a + b` with NumPy's broadcasting; array-like operands are imported."""
    return _ops.elementwise(lambda x, y: x + y, a, b, "add", deps=_OPS_DEPS)


def subtract(a: Any, b: Any) -> Any:
    """Element-wise `a - b` with NumPy's broadcasting; array-like operands are imported."""
    return _ops.elementwise(lambda x, y: x - y, a, b, "subtract", deps=_OPS_DEPS)


def multiply(a: Any, b: Any) -> Any:
    """Element-wise `a * b` with NumPy's broadcasting; array-like operands are imported."""
    return _ops.elementwise(lambda x, y: x * y, a, b, "multiply", deps=_OPS_DEPS)


def divide(a: Any, b: Any) -> Any:
    """Element-wise `a / b` with NumPy's broadcasting; array-like operands are imported."""
    return _ops.elementwise(lambda x, y: x / y, a, b, "divide", deps=_OPS_DEPS)


def norm(x: Any, ord: Any = None) -> float:
    """Return the norm of a vector or matrix.

    - ord=None: Frobenius norm (matrix) / L2 norm (vector).
    - ord='fro': Frobenius norm.
    - ord=2: spectral norm (matrix, largest singular value) / L2 norm (vector).
    - ord=1, ord=inf, ord='nuc': 1-norm, infinity-norm, nuclear norm.
    """

    if ord is not None:
        return _ops.norm(x, ord=ord, deps=_OPS_DEPS)
    return float(_native.norm(_ops._import_operand(x, deps=_OPS_DEPS)))


def sum(x: Any) -> int | float | complex:
    """Return the sum of all elements in a vector or matrix, like ``numpy.sum``.

    Integer and bool inputs give an exact Python ``int`` (no wrap; ``OverflowError`` beyond
    128 bits), real floats a ``float``, complex inputs a ``complex``. Float sums are
    deterministic (fixed summation order, independent of the thread count).
    """
    return _native.sum(_ops._import_operand(x, deps=_OPS_DEPS))


def materialize(a: Any) -> Any:
    """`a` with its elements stored (materialize.md); an array-like is imported (review 3.8
    C-10: an ndarray or list raised pybind's TypeError dump)."""
    return _native.materialize(_ops._import_operand(a, deps=_OPS_DEPS))

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


def bitwise_not(a: Any) -> Any:
    """Element-wise NOT (`~a`) of a bool or integer matrix or vector, as NumPy's `invert`."""
    return _ops.bitwise_not(a, deps=_OPS_DEPS)


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


def inv(a: Any) -> Any:
    """The inverse of a square matrix, as NumPy's `linalg.inv` (integers give float64).

    Raises numpy.linalg.LinAlgError (a ValueError) if the matrix is singular."""
    if debug_mode:
        _debug(f"inv {_matrix_desc(a)}")
    with _debug_time("inv"):
        return _ops.inv(a, deps=_OPS_DEPS)


def invert(a: Any) -> Any:
    """Element-wise NOT (`~a`), NumPy's `invert` (the same as `bitwise_not`). The matrix inverse
    is `inv` (review 3.8 C-7: `invert` was the inverse, a NumPy name with another meaning)."""
    return _ops.bitwise_not(a, deps=_OPS_DEPS)


def lstsq(a: Any, b: Any) -> Any:
    """x minimizing ||a @ x - b|| (minimum norm if rank-deficient): NumPy's lstsq(a, b)[0],
    computed with LAPACK ?gelsd. Returns only x (NumPy returns a tuple)."""
    return _ops.lstsq(a, b, deps=_OPS_DEPS)


def slogdet(a: Any) -> tuple[float, float]:
    """Return `(sign, logabsdet)` using the current determinant implementation."""
    return _ops.slogdet(a, deps=_OPS_DEPS)


def cond(a: Any, p: Any = None) -> float:
    """Condition number with NumPy's definition: smax/smin for p None/2, else ||A||_p ||A^-1||_p."""
    return _ops.cond(a, deps=_OPS_DEPS, p=p)


def eigh(a: Any) -> tuple[Any, Any]:
    """Eigenvalues (real, ascending) and eigenvectors of a symmetric/Hermitian matrix (lower triangle)."""
    return _ops.eigh(a, deps=_OPS_DEPS)


def eigvalsh(a: Any) -> Any:
    """Eigenvalues (real, ascending) of a symmetric/Hermitian matrix (lower triangle)."""
    return _ops.eigvalsh(a, deps=_OPS_DEPS)


def eigvals_arnoldi(a: Any, k: int, m: int, tol: float = 1e-6) -> Any:
    """Top-k eigenvalues via Arnoldi/Lanczos-style iteration (native when available)."""

    return _ops.eigvals_arnoldi(a, k, m, tol, deps=_OPS_DEPS)


def solve_triangular(a: Any, b: Any) -> Any:
    """Solve A X = B for a triangular, diagonal or identity A (by class)."""
    return _ops.solve_triangular(a, b, deps=_OPS_DEPS)


def pinv(a: Any) -> Any:
    """Moore-Penrose pseudoinverse (NumPy's default cutoff)."""
    return _ops.pinv(a, deps=_OPS_DEPS)


def eig(a: Any) -> tuple[Any, Any]:
    """Complex eigenvalues and right eigenvectors of a general square matrix."""
    return _ops.eig(a, deps=_OPS_DEPS)


def eigvals(a: Any) -> Any:
    """Complex eigenvalues of a general square matrix."""
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
    """Lower Cholesky factor L with A = L Lᴴ; ValueError when not positive definite."""
    return _ops.cholesky(a, deps=_OPS_DEPS)


def qr(a: Any, mode: str = 'reduced') -> Any:
    """QR decomposition; mode 'reduced', 'complete' or 'r' (NumPy's shapes)."""
    return _ops.qr(a, mode=mode, deps=_OPS_DEPS)


def svd(a: Any, full_matrices: bool = True, compute_uv: bool = True) -> Any:
    """Return SVD decomposition."""
    return _ops.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, deps=_OPS_DEPS)


def lu(a: Any) -> Any:
    """Return LU decomposition (P, L, U)."""
    return _ops.lu(a, deps=_OPS_DEPS)


def solve(a: Any, b: Any) -> Any:
    """Solve linear system AX = B."""
    if debug_mode:
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


def identity(x: Any, dtype: Any = "float64") -> Any:
    """Create an identity-like matrix of `dtype` (float64, as NumPy's `identity`).

    Accepted inputs:
    - int N (Python or NumPy): returns an N×N identity matrix.
    - sequence [rows, cols]: returns a rows×cols rectangular identity (ones on the diagonal, zeros elsewhere).
    - Matrix/Vector: returns an identity-like matrix sized to the input (matrix: rows×cols, vector: N×N).
    """
    def size(d: Any) -> Any:
        return None if isinstance(d, bool) or not hasattr(d, "__index__") else _operator.index(d)

    if size(x) is not None:
        return IdentityMatrix(size(x), dtype=dtype)
    if isinstance(x, _native.MatrixBase):
        return IdentityMatrix(x.rows(), x.cols(), dtype=dtype)
    if isinstance(x, _native.VectorBase):
        return IdentityMatrix(x.size, dtype=dtype)
    if _is_sequence_like(x) and len(x) == 2 and all(size(d) is not None for d in x):
        return IdentityMatrix(size(x[0]), size(x[1]), dtype=dtype)
    raise TypeError("identity(x) expects an int, [rows, cols], or a matrix/vector")


def _input_array(data: Any) -> Any:
    """`data` as a NumPy array for the structured constructors. Python values follow the
    matrix rules (Python ints are int32, F-103; np.asarray made them int64, review 3.8 C-11)."""
    if isinstance(data, (list, tuple)):
        flat: list[Any] = []

        def walk(x: Any) -> None:
            if isinstance(x, (list, tuple)):
                for y in x:
                    walk(y)
            else:
                flat.append(x)

        walk(data)
        inferred = _dtypes.infer_sequence_dtype(flat, np_module=_np)
        return _np.asarray(data, dtype=_matrix_api._NUMPY_DTYPE.get(inferred, "complex64"))
    return _np.asarray(data)


def _square_source(data: Any, *, name: str) -> Any:
    """The input of a structured constructor as an engine matrix: a PyCauset matrix as it is
    (no NumPy export, review 3.8 C-4), array-likes imported once (Python ints are int32)."""
    if isinstance(data, _native.MatrixBase):
        m = data
    elif isinstance(data, _native.VectorBase):
        raise ValueError(f"{name}() expects a 2D array, got 1D")
    else:
        if _ops._is_block(data):
            raise TypeError(f"{name}() takes a matrix, not a BlockMatrix (it is not densified implicitly)")
        arr = _input_array(data)
        if arr.ndim != 2:
            raise ValueError(f"{name}() expects a 2D array, got {arr.ndim}D")
        m = matrix(arr)
    if m.rows() != m.cols():
        raise ValueError(f"{name}() requires a square matrix, got shape {m.shape}")
    return m


def _validated(m: Any, structure: str, rtol: float | None, atol: float | None, what: str) -> Any:
    """`m` packed as `structure` after the engine's check (exact for integer and bool, NumPy's
    `isclose(rtol, atol)` for float and complex)."""
    r, a = (1e-5 if rtol is None else rtol), (1e-8 if atol is None else atol)
    if not _native._has_structure(m, structure, r, a):
        inexact = str(m.dtype).startswith(("float", "complex"))
        raise ValueError(f"{what} ({'within tolerance' if inexact else 'exactly'})")
    return _native._pack_as(m, structure)


def symmetric(data: Any, *, rtol: float | None = None, atol: float | None = None) -> Any:
    """A `SymmetricMatrix` (A == A.T, packed upper triangle) of the input's dtype.

    The input is validated in the engine: exactly for integer/bool, with NumPy's
    `isclose(rtol, atol)` element by element for float and complex input (only the upper
    triangle is stored). A complex symmetric matrix is not Hermitian; use `hermitian` for
    A == A.conj().T.
    """
    return _validated(_square_source(data, name="symmetric"), "symmetric", rtol, atol,
                      "symmetric() input is not symmetric: A != A.T")


def antisymmetric(data: Any, *, rtol: float | None = None, atol: float | None = None) -> Any:
    """An `AntiSymmetricMatrix` (A == -A.T, strict upper triangle stored) of the input's dtype.

    Signed integer, float and complex dtypes (unsigned and bool cannot negate: ValueError).
    Validated like `symmetric`; the diagonal must be zero.
    """
    return _validated(_square_source(data, name="antisymmetric"), "antisymmetric", rtol, atol,
                      "antisymmetric() input is not anti-symmetric: A != -A.T or a non-zero diagonal")


def hermitian(data: Any, *, rtol: float | None = None, atol: float | None = None) -> Any:
    """A `HermitianMatrix` (A == A.conj().T, packed upper triangle) of the input's complex dtype.

    Real input is promoted to the complex dtype of its precision (float16 / float32 to
    complex_float32, others to complex_float64, as NumPy's `astype(complex)` would not narrow);
    a real symmetric matrix is Hermitian, and `symmetric` gives a real packed type. Validated
    like `symmetric` (always within tolerance); the diagonal must be real.
    """
    m = _square_source(data, name="hermitian")
    dt = str(m.dtype)
    if not dt.startswith("complex"):
        m = _native._cast(m, "complex_float32" if dt in ("float16", "float32") else "complex_float64")
    r, a = (1e-5 if rtol is None else rtol), (1e-8 if atol is None else atol)
    if not _native._has_structure(m, "hermitian", r, a):
        raise ValueError("hermitian() input is not Hermitian: A != A.conj().T (within tolerance)")
    return _native._pack_as(m, "hermitian")


def diagonal(data: Any) -> Any:
    """A `DiagonalMatrix` of the input's dtype from a 1-D vector of entries or the diagonal of
    a square 2-D matrix (off-diagonal values are ignored)."""
    if isinstance(data, _native.VectorBase):
        return _native._diagonal_from_vector(data)
    if not isinstance(data, _native.MatrixBase) and not _ops._is_block(data):
        arr = _input_array(data)
        if arr.ndim == 1:
            return _native._diagonal_from_vector(_native._asarray(arr))
        if arr.ndim != 2:
            raise ValueError("diagonal() expects a 1D vector or a 2D square matrix")
        data = arr
    try:
        m = _square_source(data, name="diagonal")
    except ValueError:
        raise ValueError("diagonal() expects a 1D vector or a 2D square matrix") from None
    return _native._pack_as(m, "diagonal")


# Alias for IdentityMatrix
I = _native.IdentityMatrix  # noqa: E741 (public identity alias, NumPy convention)

# NumPy-compatible aliases: np.linalg.det -> det, np.linalg.matrix_rank -> rank shorthand.
det = determinant
rank = matrix_rank

# Plots (pycauset.vis): lightweight, lazy Plotly import. Causal-set visualization is a
# future release of its own.
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
    """Plot a causal set's embedding and open it in the browser."""
    return causet.plot_embedding().show()


def __getattr__(name):
    if name == "bool":
        return "bool"
    try:
        return getattr(_native, name)
    except AttributeError:
        # (The message named the private module, pycauset._pycauset; review 3.8 C-21.)
        raise AttributeError(f"module 'pycauset' has no attribute {name!r}") from None


# Native symbols that are internal machinery, not public API. They stay reachable
# as `pycauset.<name>` (via __getattr__) but are excluded from `__all__` so
# `from pycauset import *` does not surface implementation details.
_INTERNAL_NATIVE_EXPORTS = frozenset(
    {
        # Memory / I/O machinery.
        "MemoryGovernor",
        # Native storage plumbing (public API is set_backing_dir / save / load).
        "set_storage_root",
    }
)

__all__ = [
    name for name in dir(_native)
    if not name.startswith("_")
    and name not in _INTERNAL_NATIVE_EXPORTS
]

# Add pure-Python facade symbols (and any optional native symbols) to __all__.
# IMPORTANT: Only include names that actually resolve; otherwise
# `from pycauset import *` can fail if a symbol is not exported by the native module.
_extra_exports = [
    "plot_embedding",
    "plot_hasse",
    "plot_causal_matrix",
    "show",
    "causet",
    "CausalSet",
    "spacetime",
    "synthetic",
    "save",
    "load",
    "to_numpy",
    "set_export_max_bytes",
    "set_backing_dir",
    "get_memory_threshold",
    "set_memory_threshold",
    "last_io_trace",
    "clear_io_traces",
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
    "HermitianMatrix",
    "SkewHermitianMatrix",
    "DenseVector",
    "UnitVector",
    "matmul",
    "dot",
    "asarray",
    "add",
    "subtract",
    "multiply",
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
    "inv",
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
    "hermitian",
    "I",
    "MemoryHint",
    "AccessPattern",
    "debug_mode",
    "StaleThunkError",
]

for _name in _extra_exports:
    if _name in __all__:
        continue
    # pc.bool resolves via module __getattr__ (a module global named bool would
    # shadow the builtin), so it is recognized explicitly here.
    if _name in globals() or _name == "bool" or getattr(_native, _name, None) is not None:
        __all__.append(_name)


