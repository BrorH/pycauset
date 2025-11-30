"""Python package wrapper that augments the native pycauset extension."""
from __future__ import annotations

import atexit
import inspect
import os
import random
import re
import warnings
import weakref
import uuid
import shutil
from collections.abc import Sequence as _SequenceABC
from pathlib import Path
from typing import Any, Sequence, Tuple
from importlib import import_module as _import_module

from ._storage import StorageRegistry, cleanup_storage, set_temporary_file

try:  # NumPy is optional at runtime
    import numpy as _np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    _np = None

_native = _import_module(".pycauset", package=__name__)
_TriangularBitMatrix = _native.TriangularBitMatrix
_original_triangular_bit_matrix_init = _TriangularBitMatrix.__init__
_original_triangular_bit_matrix_random = _TriangularBitMatrix.random

_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*=.+(?:CausalMatrix|TriangularBitMatrix)", re.IGNORECASE)
_STORAGE_ROOT: Path | None = None
_EDGE_ITEMS = 3


def _storage_root() -> Path:
    global _STORAGE_ROOT
    if _STORAGE_ROOT is not None:
        return _STORAGE_ROOT
    env = os.environ.get("PYCAUSET_STORAGE_DIR")
    if env:
        base = Path(env).expanduser()
    else:
        base = Path.cwd().resolve() / ".pycauset"
    base.mkdir(parents=True, exist_ok=True)
    _STORAGE_ROOT = base
    return base

# Perform initial cleanup of temporary files from previous runs
cleanup_storage(_storage_root())

_STORAGE_REGISTRY = StorageRegistry(_storage_root())
_LIVE_MATRICES: weakref.WeakSet = weakref.WeakSet()
keep_temp_files: bool = False
seed: int | None = None
def _track_matrix(instance: Any) -> None:
    try:
        _LIVE_MATRICES.add(instance)
    except TypeError:
        pass


def _release_tracked_matrices() -> None:
    for matrix in list(_LIVE_MATRICES):
        close = getattr(matrix, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _register_cleanup() -> None:
    def _finalize() -> None:
        _release_tracked_matrices()
        if not keep_temp_files:
            cleanup_storage(_storage_root())

    atexit.register(_finalize)


_register_cleanup()


def _is_simple_name(candidate: str) -> bool:
    if not candidate:
        return False
    seps = [os.sep]
    if os.altsep:
        seps.append(os.altsep)
    return not any(sep in candidate for sep in seps)


def _sanitize_name(name: str | None, fallback: str) -> str:
    target = (name or fallback or "matrix").strip()
    if not target:
        target = fallback or "matrix"
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", target)
    if not cleaned:
        cleaned = fallback or "matrix"
    if cleaned in {".", ".."}:
        cleaned = fallback or "matrix"
    if not cleaned.endswith(".pycauset"):
        cleaned = f"{cleaned}.pycauset"
    return cleaned


def _infer_assignment_target() -> str | None:
    try:
        stack = inspect.stack()
    except OSError:
        return None
    try:
        for frame_info in stack[2:8]:
            context = frame_info.code_context
            if not context:
                continue
            for line in context:
                match = _ASSIGNMENT_RE.match(line.strip())
                if match:
                    return match.group(1)
    finally:
        del stack
    return None


def _resolve_backing_path(backing: Any, fallback: str | None = None) -> str:
    if backing in (None, ""):
        inferred = _infer_assignment_target()
        base = inferred if inferred else (fallback or "matrix")
        # Append UUID to ensure uniqueness for temporary files
        unique_name = f"{base}_{uuid.uuid4().hex[:8]}.pycauset"
        return str(_storage_root() / unique_name)

    if isinstance(backing, os.PathLike):
        candidate = Path(backing).expanduser()
    else:
        candidate = Path(str(backing)).expanduser()

    if _is_simple_name(str(backing)) and candidate.parent == Path('.') and not candidate.is_absolute():
        filename = _sanitize_name(str(backing), fallback or "matrix")
        return str(_storage_root() / filename)

    if candidate.suffix != ".pycauset":
        candidate = candidate.with_suffix(candidate.suffix + ".pycauset" if candidate.suffix else ".pycauset")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _should_register_auto(backing: Any) -> bool:
    return backing in (None, "")


def _prepare_backing_args(args: Sequence[Any], kwargs: dict[str, Any], target_arg: str = "saveas") -> Tuple[Tuple[Any, ...], dict[str, Any]]:
    mutable_args = list(args)
    provided = None
    kw_alias = None
    if len(mutable_args) >= 2:
        provided = mutable_args[1]
    else:
        for candidate in ("saveas", "backing_file"):
            if candidate in kwargs:
                provided = kwargs[candidate]
                kw_alias = candidate
                break

    if provided is not None:
        warnings.warn(
            "The 'backing_file' (or 'saveas') argument is deprecated and will be ignored. "
            "All matrices are now created as temporary files. "
            "Use 'pycauset.save(matrix, path)' to save the matrix to a specific location.",
            DeprecationWarning,
            stacklevel=3
        )

    # Always resolve to a temporary path (None -> temp file)
    resolved = _resolve_backing_path(None)
    
    if len(mutable_args) >= 2:
        mutable_args[1] = resolved
    else:
        if kw_alias and kw_alias != target_arg:
            kwargs.pop(kw_alias, None)
        kwargs[target_arg] = resolved

    return tuple(mutable_args), kwargs


def _extract_size_hint(args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("N")


def _is_sequence_like(value: Any) -> bool:
    return isinstance(value, _SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def _coerce_sequence_rows(candidate: Any) -> tuple[int, list[list[Any]]]:
    if not _is_sequence_like(candidate):
        raise TypeError("Matrix data must be provided as a square nested sequence or a NumPy array.")
    rows = [list(row) for row in candidate]
    if not rows:
        raise ValueError("Matrix data must not be empty.")
    size = len(rows)
    for row in rows:
        if not _is_sequence_like(row):
            raise TypeError("Each matrix row must be a sequence of entries.")
        if len(row) != size:
            raise ValueError("Matrix data must describe a square matrix (same number of rows and columns).")
    return size, rows


def _coerce_general_matrix(candidate: Any) -> tuple[int, list[list[Any]]]:
    size_attr = getattr(candidate, "size", None)
    get_attr = getattr(candidate, "get", None)
    if callable(size_attr) and callable(get_attr):
        size = int(size_attr())
        if size < 0:
            raise ValueError("Matrix size must be non-negative.")
        rows: list[list[Any]] = []
        for i in range(size):
            row: list[Any] = []
            for j in range(size):
                row.append(get_attr(i, j))
            rows.append(row)
        return size, rows

    if _np is not None:
        try:
            array = _np.asarray(candidate)
        except Exception:
            array = None
        else:
            if array.ndim != 2:
                raise ValueError("Matrix input must be a 2D square structure.")
            if array.shape[0] != array.shape[1]:
                raise ValueError("Matrix input must be square (rows == columns).")
            return int(array.shape[0]), array.tolist()

    return _coerce_sequence_rows(candidate)


def _mark_temporary_if_auto(matrix: Any) -> None:
    """Mark the matrix file as temporary if it resides in the storage root."""
    if not hasattr(matrix, "get_backing_file"):
        return
    try:
        path = Path(matrix.get_backing_file()).resolve()
        root = _storage_root().resolve()
        # Check if path is inside root (implies auto-generated)
        if path.is_relative_to(root):
            # It's an auto-generated file, mark it as temporary
            if hasattr(matrix, "set_temporary"):
                matrix.set_temporary(True)
            else:
                # Fallback if method not available (should be on all MatrixBase)
                set_temporary_file(path, True)
    except (ValueError, OSError, AttributeError):
        pass

def _patched_triangular_bit_matrix_init(self, *args: Any, **kwargs: Any) -> None:
    mutable_args = list(args)
    if len(mutable_args) > 0 and isinstance(mutable_args[0], float):
        mutable_args[0] = int(mutable_args[0])
    elif "N" in kwargs and isinstance(kwargs["N"], float):
        kwargs["N"] = int(kwargs["N"])
    
    # Enforce temporary storage: ignore user provided backing_file/saveas
    
    # Check if user tried to provide a path
    if len(mutable_args) >= 2 or "saveas" in kwargs or "backing_file" in kwargs:
         warnings.warn(
            "Providing a backing file during creation is deprecated. "
            "Matrices are now created as temporary files. Use pycauset.save(matrix, path) to persist them.",
            DeprecationWarning,
            stacklevel=2
        )
    
    # Strip path arguments
    if len(mutable_args) >= 2:
        mutable_args = mutable_args[:1]
    kwargs.pop("saveas", None)
    kwargs.pop("backing_file", None)
    
    # Generate temp path
    resolved = _resolve_backing_path(None)
    
    # Call original init with (N, resolved)
    # Note: C++ signature is now (n, saveas)
    _original_triangular_bit_matrix_init(self, mutable_args[0], resolved)
    _track_matrix(self)
    _mark_temporary_if_auto(self)


def _maybe_register_result(result: Any) -> None:
    if result is None:
        return
    if not hasattr(result, "get_backing_file"):
        return
    _track_matrix(result)
    _mark_temporary_if_auto(result)


_METHODS_RETURNING_MATRIX = ["multiply", "invert", "__invert__", "__add__", "__sub__", "__mul__", "__rmul__"]


def _patch_matrix_methods(cls: Any) -> None:
    for name in _METHODS_RETURNING_MATRIX:
        if not hasattr(cls, name):
            continue
        original = getattr(cls, name)
        
        def make_wrapper(orig: Any) -> Any:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = orig(self, *args, **kwargs)
                _maybe_register_result(result)
                return result
            return wrapper
            
        setattr(cls, name, make_wrapper(original))


def _patch_matrix_class(cls: Any, target_arg: str = "backing_file") -> None:
    original_init = cls.__init__
    
    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        mutable_args = list(args)
        if len(mutable_args) > 0 and isinstance(mutable_args[0], float):
            mutable_args[0] = int(mutable_args[0])
        elif "N" in kwargs and isinstance(kwargs["N"], float):
            kwargs["N"] = int(kwargs["N"])
            
        new_args, new_kwargs = _prepare_backing_args(tuple(mutable_args), kwargs, target_arg=target_arg)
        
        original_init(self, *new_args, **new_kwargs)
        _track_matrix(self)
        _mark_temporary_if_auto(self)
        
    cls.__init__ = _patched_init
    _patch_matrix_methods(cls)


def _patched_triangular_bit_matrix_random(
    N: int, density: float = 0.5, backing_file: Any = None, seed: int | None = None, seed_override: int | None = None
):
    N = int(N)
    resolved = _resolve_backing_path(backing_file, fallback="random")
    if _should_register_auto(backing_file):
        _STORAGE_REGISTRY.register_auto_file(resolved)
    
    actual_seed = seed
    if actual_seed is None:
        actual_seed = seed_override
    if actual_seed is None:
        actual_seed = globals().get("seed")

    matrix = _original_triangular_bit_matrix_random(N, density, resolved, actual_seed)
    _track_matrix(matrix)
    return matrix



def _edge_indices(length: int) -> tuple[list[int], list[int], bool]:
    if length <= _EDGE_ITEMS * 2:
        return list(range(length)), [], False
    head = list(range(_EDGE_ITEMS))
    tail = list(range(length - _EDGE_ITEMS, length))
    return head, tail, True


def _format_value(value: Any) -> str:
    if _np is not None and isinstance(value, _np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_matrix_row(matrix: Any, row_index: int, col_head: list[int], col_tail: list[int], truncated: bool) -> str:
    entries: list[str] = []
    
    # Check for scalar scaling
    scalar = getattr(matrix, "scalar", 1.0)
    use_scaling = scalar != 1.0

    for col in col_head:
        val = matrix.get(row_index, col)
        if use_scaling:
            val = val * scalar
        entries.append(_format_value(val))
    if truncated:
        entries.append("...")
    for col in col_tail:
        val = matrix.get(row_index, col)
        if use_scaling:
            val = val * scalar
        entries.append(_format_value(val))
    return " ".join(entries)


def _matrix_str(self) -> str:
    size = self.size()
    
    # Build header info
    info = [f"shape=({size}, {size})"]
    
    # Check for scalar
    if hasattr(self, "scalar") and self.scalar != 1.0:
        info.append(f"scalar={self.scalar}")
        
    # Check for seed
    if hasattr(self, "seed") and self.seed != 0:
        info.append(f"seed={self.seed}")
        
    header = f"{self.__class__.__name__}({', '.join(info)})"

    if size == 0:
        return header + "\n[]"
    row_head, row_tail, rows_truncated = _edge_indices(size)
    col_head, col_tail, cols_truncated = _edge_indices(size)

    lines = [header, "["]
    for row_index in row_head:
        row_repr = _format_matrix_row(self, row_index, col_head, col_tail, cols_truncated)
        lines.append(f" [{row_repr}]")
    if rows_truncated:
        lines.append(" ...")
    for row_index in row_tail:
        row_repr = _format_matrix_row(self, row_index, col_head, col_tail, cols_truncated)
        lines.append(f" [{row_repr}]")
    lines.append("]")
    return "\n".join(lines)


import abc

_TriangularBitMatrix.__str__ = _matrix_str


class MatrixMixin:
    def __str__(self) -> str:
        return _matrix_str(self)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} shape={self.shape}>"


class Matrix(MatrixMixin, metaclass=abc.ABCMeta):
    """Base class for all matrix types in pycauset.
    
    Also serves as an in-memory dense matrix implementation that accepts integers, lists, or NumPy arrays.
    """

    def __init__(self, source: Any, saveas: Any = None):
        if saveas not in (None, ""):
            warnings.warn(
                "pycauset.Matrix does not persist to disk; use pycauset.TriangularBitMatrix for storage.",
                RuntimeWarning,
                stacklevel=2,
            )

        if isinstance(source, float):
            source = int(source)

        if isinstance(source, int):
            if source < 0:
                raise ValueError("Matrix dimension must be non-negative.")
            self._size = source
            self._data: list[list[Any]] = [
                [0 for _ in range(source)] for _ in range(source)
            ]
        else:
            size, rows = _coerce_general_matrix(source)
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


# Inject methods and register subclasses
for cls in (_TriangularBitMatrix, _native.IntegerMatrix, _native.FloatMatrix, _native.TriangularFloatMatrix):
    cls.__str__ = MatrixMixin.__str__
    cls.__repr__ = MatrixMixin.__repr__
    Matrix.register(cls)

# Register triangular subclasses
TriangularMatrix.register(_TriangularBitMatrix)
TriangularMatrix.register(_native.TriangularFloatMatrix)

# Patch TriangularBitMatrix init/random
_TriangularBitMatrix.__init__ = _patched_triangular_bit_matrix_init
_TriangularBitMatrix.random = staticmethod(_patched_triangular_bit_matrix_random)
_patch_matrix_methods(_TriangularBitMatrix)

# Patch other matrix classes to ensure cleanup
_patch_matrix_class(_native.IntegerMatrix, target_arg="backing_file")
_patch_matrix_class(_native.FloatMatrix, target_arg="backing_file")
_patch_matrix_class(_native.TriangularFloatMatrix, target_arg="backing_file")
if hasattr(_native, "TriangularIntegerMatrix"):
    _patch_matrix_class(_native.TriangularIntegerMatrix, target_arg="backing_file")

# Aliases
TriangularBitMatrix = _TriangularBitMatrix
IntegerMatrix = _native.IntegerMatrix
FloatMatrix = _native.FloatMatrix
TriangularFloatMatrix = _native.TriangularFloatMatrix
if hasattr(_native, "TriangularIntegerMatrix"):
    TriangularIntegerMatrix = _native.TriangularIntegerMatrix

def CausalMatrix(*args, **kwargs):
    """Factory function for creating TriangularBitMatrix instances (formerly CausalMatrix)."""
    return TriangularBitMatrix(*args, **kwargs)

CausalMatrix.random = TriangularBitMatrix.random

_native_matmul = getattr(_native, "matmul")

def matmul(a: Any, b: Any, saveas: str | None = None) -> Any:
    """
    Perform matrix multiplication.
    
    If both inputs are TriangularBitMatrices, uses the optimized C++ implementation.
    Otherwise, performs generic multiplication (slow).
    """
    if isinstance(a, _TriangularBitMatrix) and isinstance(b, _TriangularBitMatrix):
        if saveas is not None:
            warnings.warn(
                "The 'saveas' argument is deprecated and will be removed in a future version. "
                "Please use 'pycauset.save(matrix, path)' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        resolved = _resolve_backing_path(saveas, fallback="matmul")
        # if _should_register_auto(saveas):
        #     _STORAGE_REGISTRY.register_auto_file(resolved)
        result = _native_matmul(a, b, resolved)
        _track_matrix(result)
        _mark_temporary_if_auto(result)
        return result
    
    # Generic fallback
    if not (hasattr(a, "shape") and hasattr(b, "shape")):
        raise TypeError("Inputs must be matrix-like objects with a shape property.")
        
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
    N = a.shape[0]
    M = b.shape[1]
    
    # If numpy is available, use it
    if _np is not None:
        # Try to convert to numpy arrays
        try:
            a_np = _np.array([[a.get(i, j) for j in range(a.shape[1])] for i in range(a.shape[0])])
            b_np = _np.array([[b.get(i, j) for j in range(b.shape[1])] for i in range(b.shape[0])])
            res_np = _np.matmul(a_np, b_np)
            return Matrix(res_np)
        except Exception:
            pass # Fallback to slow loop
            
    # Slow generic loop
    res = Matrix(N) 
    if N != M:
         raise NotImplementedError("Generic multiplication currently only supports square result matrices.")
         
    for i in range(N):
        for j in range(M):
            val = 0
            for k in range(a.shape[1]):
                val += a.get(i, k) * b.get(k, j)
            res.set(i, j, val)
            
    return res

def _generate_temp_path(prefix: str) -> str:
    root = _storage_root()
    name = f"{prefix}_{uuid.uuid4().hex}.bin"
    path = root / name
    return str(path)

def compute_k(matrix: TriangularBitMatrix, a: float):
    """
    Compute K = C(aI + C)^-1.
    
    Args:
        matrix: The TriangularBitMatrix C.
        a: The scalar a.
        
    Returns:
        A TriangularFloatMatrix representing K.
    """
    resolved = _resolve_backing_path(None, fallback="k_matrix")
    # _STORAGE_REGISTRY.register_auto_file(resolved) # Deprecated
        
    _native.compute_k_matrix(matrix, a, resolved, 0)
    result = _native.load(resolved)
    _track_matrix(result)
    _mark_temporary_if_auto(result)
    return result


def bitwise_not(matrix: Any) -> Any:
    """
    Compute the bitwise inversion (NOT) of a matrix.
    
    Args:
        matrix: The matrix to invert. Must be a TriangularBitMatrix or IntegerMatrix.
        
    Returns:
        A new matrix with inverted bits.
    """
    if hasattr(matrix, "__invert__"):
        return ~matrix
        
    if _np is not None:
        try:
            return _np.invert(matrix)
        except Exception:
            pass
            
    raise TypeError("Object does not support bitwise inversion.")


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
    if hasattr(matrix, "invert"):
        return matrix.invert()
        
    if _np is not None:
        try:
            return _np.linalg.inv(matrix)
        except Exception:
            pass
            
    raise TypeError("Object does not support matrix inversion.")


def save(matrix: Any, path: str | os.PathLike) -> None:
    """
    Save a matrix to a permanent location.
    
    This function attempts to create a hard link to the matrix's backing file.
    If that fails (e.g. cross-device), it falls back to copying the file.
    
    Args:
        matrix: The matrix object to save. Must have a backing file.
        path: The destination path.
    """
    if not hasattr(matrix, "get_backing_file"):
        raise TypeError("The provided object does not support file-backed storage.")
        
    source = Path(matrix.get_backing_file())
    if not source.exists():
        raise FileNotFoundError(f"Backing file not found: {source}")
        
    dest = Path(path).resolve()
    if dest.is_dir():
        dest = dest / source.name
        
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        if os.path.samefile(source, dest):
            return
        os.unlink(dest)
        
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)
        
    # Ensure the saved file is marked as permanent
    set_temporary_file(dest, False)


def __getattr__(name):
    return getattr(_native, name)


__all__ = [name for name in dir(_native) if not name.startswith("__")]
__all__.extend(["save", "keep_temp_files", "seed", "Matrix", "TriangularMatrix", "CausalMatrix", "TriangularBitMatrix", "compute_k", "bitwise_not", "invert"])
