"""Python package wrapper that augments the native pycauset extension."""
from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import atexit
import inspect
import os
import random
import re
import warnings
import weakref
import uuid
import shutil
import abc
from collections.abc import Sequence as _SequenceABC
from pathlib import Path
from typing import Any, Sequence, Tuple
from importlib import import_module as _import_module

from ._storage import StorageRegistry, cleanup_storage, set_temporary_file
from .causet import CausalSet

# Alias for CausalSet
Causet = CausalSet

try:  # NumPy is optional at runtime
    import numpy as _np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    _np = None

_native = _import_module("._pycauset", package=__name__)
_TriangularBitMatrix = _native.TriangularBitMatrix
_original_triangular_bit_matrix_init = _TriangularBitMatrix.__init__
_original_triangular_bit_matrix_random = _TriangularBitMatrix.random

# Private aliases for native classes
_IntegerMatrix = _native.IntegerMatrix
_FloatMatrix = _native.FloatMatrix
_TriangularFloatMatrix = _native.TriangularFloatMatrix
_TriangularIntegerMatrix = getattr(_native, "TriangularIntegerMatrix", None)
_DenseBitMatrix = getattr(_native, "DenseBitMatrix", None)

# Vector classes
_FloatVector = getattr(_native, "FloatVector", None)
_IntegerVector = getattr(_native, "IntegerVector", None)
_BitVector = getattr(_native, "BitVector", None)

_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*=.+(?:CausalMatrix|TriangularBitMatrix)", re.IGNORECASE)
_STORAGE_ROOT: Path | None = None
_EDGE_ITEMS = 4


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
    N: int, p: float = 0.5, backing_file: Any = None, seed: int | None = None, seed_override: int | None = None
):
    N = int(N)
    resolved = _resolve_backing_path(backing_file, fallback="random")
    # if _should_register_auto(backing_file):
    #     _STORAGE_REGISTRY.register_auto_file(resolved)
    
    actual_seed = seed
    if actual_seed is None:
        actual_seed = seed_override
    if actual_seed is None:
        actual_seed = globals().get("seed")

    matrix = _original_triangular_bit_matrix_random(N, p, resolved, actual_seed)
    _track_matrix(matrix)
    _mark_temporary_if_auto(matrix)
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


_TriangularBitMatrix.__str__ = _matrix_str


class MatrixMixin:
    def __str__(self) -> str:
        return _matrix_str(self)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} shape={self.shape}>"


class Matrix(MatrixMixin, metaclass=abc.ABCMeta):
    """
    The Matrix class. 
    
    This class acts as a smart factory. 
    - If you provide data that fits into an optimized C++ backend (integers, triangular), 
      it returns an instance of that optimized class.
    - Otherwise, it initializes a standard Python-list based Matrix.
    """
    def __new__(cls, size_or_data: Any, dtype: Any = None, **kwargs: Any):
        # If subclassing, behave like a normal class
        if cls is not Matrix:
            return super().__new__(cls)
            
        # Resolve dtype
        target_dtype = None
        if dtype is not None:
            if dtype in (int, "int", "int32", "int64"):
                target_dtype = "int"
            elif dtype in (float, "float", "float64", "float32"):
                target_dtype = "float"
            elif dtype in (bool, "bool", "bool_"):
                target_dtype = "bool"
            elif _np is not None:
                if dtype in (_np.int32, _np.int64, _np.integer):
                    target_dtype = "int"
                elif dtype in (_np.float64, _np.float32, _np.floating):
                    target_dtype = "float"
                elif dtype in (_np.bool_, _np.bool):
                    target_dtype = "bool"
        
        # 1. Handle creation by Size
        if isinstance(size_or_data, (int, float)) and (isinstance(size_or_data, int) or size_or_data.is_integer()):
            n = int(size_or_data)
            if target_dtype == "int":
                return _IntegerMatrix(n, **kwargs)
            elif target_dtype == "bool":
                if _DenseBitMatrix:
                    return _DenseBitMatrix(n, **kwargs)
                return _IntegerMatrix(n, **kwargs) # Fallback
            else:
                # Default to FloatMatrix if no dtype or float
                return _FloatMatrix(n, **kwargs)
        
        # 2. Handle creation by Data
        data = size_or_data
        
        # Check for NumPy
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                if dtype is None:
                    # Infer dtype from numpy array
                    if data.dtype.kind in ('i', 'u'):
                        target_dtype = "int"
                    elif data.dtype.kind == 'f':
                        target_dtype = "float"
                    elif data.dtype.kind == 'b':
                        target_dtype = "bool"
                data = data.tolist()
        except ImportError:
            pass

        # Analyze data to pick the best backend
        try:
            is_integer = True
            is_triangular = True
            rows = len(data)
            
            if rows == 0:
                return _FloatMatrix(0, **kwargs)

            for i, row in enumerate(data):
                if len(row) != rows:
                    # Not square? Fallback to standard Python Matrix (this class)
                    return super(Matrix, cls).__new__(cls)
                
                for j, val in enumerate(row):
                    # Check Type (only if not forced)
                    if target_dtype is None:
                        if is_integer and not isinstance(val, (int, bool)):
                            is_integer = False
                    
                    # Check Triangularity (strictly upper)
                    if is_triangular and j <= i and val != 0:
                        is_triangular = False
            
            # Helper to create and fill
            def create_and_fill(cls_type, data_source):
                size, rows = _coerce_general_matrix(data_source)
                mat = cls_type(size, **kwargs)
                for i in range(size):
                    for j in range(size):
                        val = rows[i][j]
                        if val != 0:
                            mat.set(i, j, val)
                return mat

            # Dispatch Logic
            if target_dtype == "int":
                if is_triangular and _TriangularIntegerMatrix:
                    return create_and_fill(_TriangularIntegerMatrix, data)
                return create_and_fill(_IntegerMatrix, data)
            
            elif target_dtype == "bool":
                if is_triangular:
                    return create_and_fill(_TriangularBitMatrix, data)
                if _DenseBitMatrix:
                    return create_and_fill(_DenseBitMatrix, data)
                return create_and_fill(_IntegerMatrix, data)

            elif target_dtype == "float":
                if is_triangular:
                    return create_and_fill(_TriangularFloatMatrix, data)
                return create_and_fill(_FloatMatrix, data)

            # Auto-detection (target_dtype is None)
            if is_integer:
                if is_triangular and _TriangularIntegerMatrix:
                    return create_and_fill(_TriangularIntegerMatrix, data)
                else:
                    return create_and_fill(_IntegerMatrix, data)
            else:
                if is_triangular:
                    return create_and_fill(_TriangularFloatMatrix, data)
                else:
                    return create_and_fill(_FloatMatrix, data)

        except Exception:
            # If anything goes wrong during analysis, fallback to standard Python Matrix
            return super(Matrix, cls).__new__(cls)

    def __init__(self, size_or_data: Any, saveas: Any = None):
        """
        Initializes the Python-based Matrix. 
        NOTE: This is ONLY called if __new__ returns a standard Matrix object.
        If __new__ returns a C++ object, this __init__ is skipped.
        """
        if saveas not in (None, ""):
            warnings.warn(
                "pycauset.Matrix does not persist to disk; use pycauset.save() for storage.",
                RuntimeWarning,
                stacklevel=2,
            )

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


# Inject methods and register subclasses
for cls in (_TriangularBitMatrix, _IntegerMatrix, _FloatMatrix, _TriangularFloatMatrix):
    cls.__str__ = MatrixMixin.__str__
    cls.__repr__ = MatrixMixin.__repr__
    Matrix.register(cls)

if _TriangularIntegerMatrix:
    _TriangularIntegerMatrix.__str__ = MatrixMixin.__str__
    _TriangularIntegerMatrix.__repr__ = MatrixMixin.__repr__
    Matrix.register(_TriangularIntegerMatrix)

if _DenseBitMatrix:
    _DenseBitMatrix.__str__ = MatrixMixin.__str__
    _DenseBitMatrix.__repr__ = MatrixMixin.__repr__
    Matrix.register(_DenseBitMatrix)

# Register triangular subclasses
TriangularMatrix.register(_TriangularBitMatrix)
TriangularMatrix.register(_TriangularFloatMatrix)
if _TriangularIntegerMatrix:
    TriangularMatrix.register(_TriangularIntegerMatrix)

# Patch TriangularBitMatrix init/random
_TriangularBitMatrix.__init__ = _patched_triangular_bit_matrix_init
_TriangularBitMatrix.random = staticmethod(_patched_triangular_bit_matrix_random)
_patch_matrix_methods(_TriangularBitMatrix)

# Patch other matrix classes to ensure cleanup
_patch_matrix_class(_IntegerMatrix, target_arg="backing_file")
_patch_matrix_class(_FloatMatrix, target_arg="backing_file")
_patch_matrix_class(_TriangularFloatMatrix, target_arg="backing_file")
if _TriangularIntegerMatrix:
    _patch_matrix_class(_TriangularIntegerMatrix, target_arg="backing_file")

if _DenseBitMatrix:
    _patch_matrix_class(_DenseBitMatrix, target_arg="backing_file")

# Patch vector classes
if _FloatVector:
    _patch_matrix_class(_FloatVector, target_arg="backing_file")
if _IntegerVector:
    _patch_matrix_class(_IntegerVector, target_arg="backing_file")
if _BitVector:
    _patch_matrix_class(_BitVector, target_arg="backing_file")

# Aliases (Only CausalMatrix and Matrix are public now)
TriangularBitMatrix = _TriangularBitMatrix

def Vector(size_or_data: Any, dtype: str | None = None, **kwargs) -> Any:
    """
    Factory function for creating Vector instances.
    
    Args:
        size_or_data: Size of the vector (int) or data (list/array).
        dtype: 'float', 'int', or 'bool'. If None, inferred from data.
        **kwargs: Additional arguments (ignored for now).
    """
    # 1. Handle creation by Size
    if isinstance(size_or_data, (int, float)) and (isinstance(size_or_data, int) or size_or_data.is_integer()):
        n = int(size_or_data)
        if dtype == "int" and _IntegerVector:
            return _IntegerVector(n, **kwargs)
        elif dtype == "bool" and _BitVector:
            return _BitVector(n, **kwargs)
        elif _FloatVector:
            return _FloatVector(n, **kwargs)
        else:
            raise ImportError("Vector classes not available in native extension.")

    # 2. Handle creation by Data
    data = size_or_data
    
    # Check for NumPy
    if _np is not None and isinstance(data, _np.ndarray):
        if dtype is None:
            if data.dtype.kind in ('i', 'u'):
                dtype = "int"
            elif data.dtype.kind == 'f':
                dtype = "float"
            elif data.dtype.kind == 'b':
                dtype = "bool"
        data = data.tolist()

    # Infer dtype if not provided
    if dtype is None:
        dtype = "float" # Default
        if all(isinstance(x, (int, bool)) for x in data):
            dtype = "int"
        if all(isinstance(x, bool) for x in data):
             dtype = "bool"

    n = len(data)
    
    if dtype == "int" and _IntegerVector:
        vec = _IntegerVector(n, **kwargs)
    elif dtype == "bool" and _BitVector:
        vec = _BitVector(n, **kwargs)
    elif _FloatVector:
        vec = _FloatVector(n, **kwargs)
    else:
        raise ImportError("Vector classes not available in native extension.")

    for i, val in enumerate(data):
        vec[i] = val
        
    return vec

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
    # Case 1: source is an integer size
    if isinstance(source, (int, float)) and (isinstance(source, int) or source.is_integer()):
        n = int(source)
        if populate:
            return TriangularBitMatrix.random(n, p=0.5, **kwargs)
        return TriangularBitMatrix(n, **kwargs)

    # Case 2: source is data (list of lists, numpy array, etc.)
    # If it's a numpy array, try to let the native constructor handle it (fast path)
    if _np is not None and isinstance(source, _np.ndarray):
        # Check for non-binary values
        if source.dtype != bool:
            if not _np.all(_np.isin(source, [0, 1])):
                warnings.warn(
                    "Input data contains non-binary values. They will be converted to boolean (True/False).",
                    UserWarning,
                    stacklevel=2
                )
        
        # Check for non-zero values in lower triangle or diagonal
        if _np.any(_np.tril(source) != 0):
             warnings.warn(
                "Input data contains non-zero values in the lower triangle or diagonal. "
                "CausalMatrix is strictly upper triangular; these values will be ignored.",
                UserWarning,
                stacklevel=2
            )

        try:
            # Ensure bool type for the native constructor
            if source.dtype != bool:
                source = source.astype(bool)
            return TriangularBitMatrix(source, **kwargs)
        except Exception:
            # Fallback to generic coercion if native init fails
            pass

    # Generic coercion (slow path, but works for lists/iterables)
    size, rows = _coerce_general_matrix(source)
    matrix = TriangularBitMatrix(size, **kwargs)
    
    has_non_binary = False
    has_lower_triangular = False

    # Populate strictly upper triangular part
    for i in range(size):
        for j in range(size):
            val = rows[i][j]
            
            # Check for non-binary (only once to avoid spam)
            if not has_non_binary:
                if val not in (0, 1, False, True, 0.0, 1.0):
                    has_non_binary = True
            
            # Check for lower triangular/diagonal
            if j <= i:
                if val: # Truthy check
                    has_lower_triangular = True
                continue

            # Truthy check handles 1, 1.0, True
            if val:
                matrix.set(i, j, True)
    
    if has_non_binary:
        warnings.warn(
            "Input data contains non-binary values. They will be converted to boolean (True/False).",
            UserWarning,
            stacklevel=2
        )
        
    if has_lower_triangular:
        warnings.warn(
            "Input data contains non-zero values in the lower triangle or diagonal. "
            "CausalMatrix is strictly upper triangular; these values will be ignored.",
            UserWarning,
            stacklevel=2
        )
                
    return matrix

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
        
    # Always copy to ensure the saved file is independent of the temporary one.
    # Hardlinking would share the file header, so marking the destination as
    # permanent would also mark the source as permanent, preventing cleanup.
    shutil.copy2(source, dest)
        
    # Ensure the saved file is marked as permanent
    set_temporary_file(dest, False)


# Monkey-patch MatrixBase to add the save method to all matrix classes
if hasattr(_native, "MatrixBase"):
    _native.MatrixBase.save = save


# Alias for IdentityMatrix
I = _native.IdentityMatrix


def __getattr__(name):
    return getattr(_native, name)


__all__ = [name for name in dir(_native) if not name.startswith("__")]
__all__.extend(["save", "keep_temp_files", "seed", "Matrix", "Vector", "TriangularMatrix", "CausalMatrix", "TriangularBitMatrix", "compute_k", "bitwise_not", "invert", "I", "CausalSet", "Causet"])
# Remove deprecated classes from __all__ if they were added by dir(_native)
for _deprecated in ["IntegerMatrix", "FloatMatrix", "TriangularFloatMatrix", "TriangularIntegerMatrix", "FloatVector", "IntegerVector", "BitVector"]:
    if _deprecated in __all__:
        __all__.remove(_deprecated)
