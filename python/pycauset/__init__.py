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
from collections.abc import Sequence as _SequenceABC
from pathlib import Path
from typing import Any, Sequence, Tuple
from importlib import import_module as _import_module

from ._storage import StorageRegistry

try:  # NumPy is optional at runtime
    import numpy as _np
except ImportError:  # pragma: no cover - exercised when numpy is absent
    _np = None

_native = _import_module(".pycauset", package=__name__)
_CausalMatrix = _native.CausalMatrix
_original_causal_matrix_init = _CausalMatrix.__init__
_original_causal_matrix_random = _CausalMatrix.random

_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*=.+CausalMatrix", re.IGNORECASE)
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


_STORAGE_REGISTRY = StorageRegistry(_storage_root())
_LIVE_MATRICES: weakref.WeakSet = weakref.WeakSet()
save: bool = False
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
        _STORAGE_REGISTRY.finalize(bool(save))

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
        filename = _sanitize_name(inferred, fallback or "matrix")
        return str(_storage_root() / filename)

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


def _prepare_backing_args(args: Sequence[Any], kwargs: dict[str, Any]) -> Tuple[Tuple[Any, ...], dict[str, Any]]:
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

    resolved = _resolve_backing_path(provided)
    if _should_register_auto(provided):
        _STORAGE_REGISTRY.register_auto_file(resolved)

    if len(mutable_args) >= 2:
        mutable_args[1] = resolved
    else:
        if kw_alias and kw_alias != "saveas":
            kwargs.pop(kw_alias, None)
        kwargs["saveas"] = resolved

    return tuple(mutable_args), kwargs


def _extract_size_hint(args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("N")


def _should_seed_constructor(args: Sequence[Any], kwargs: dict[str, Any]) -> bool:
    populate = kwargs.get("populate")
    if populate is None:
        populate = True
    if not populate:
        return False
    candidate = _extract_size_hint(args, kwargs)
    return isinstance(candidate, int)


def _maybe_apply_seed(args: Sequence[Any], kwargs: dict[str, Any]) -> Tuple[Sequence[Any], dict[str, Any]]:
    if kwargs.get("seed") is not None or seed is None:
        return args, kwargs
    if not _should_seed_constructor(args, kwargs):
        return args, kwargs
    kwargs["seed"] = seed
    return args, kwargs


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


def _patched_causal_matrix_init(self, *args: Any, **kwargs: Any) -> None:
    mutable_args = list(args)
    if len(mutable_args) > 0 and isinstance(mutable_args[0], float):
        mutable_args[0] = int(mutable_args[0])
    elif "N" in kwargs and isinstance(kwargs["N"], float):
        kwargs["N"] = int(kwargs["N"])
        
    new_args, new_kwargs = _prepare_backing_args(tuple(mutable_args), kwargs)
    new_args, new_kwargs = _maybe_apply_seed(new_args, new_kwargs)
    _original_causal_matrix_init(self, *new_args, **new_kwargs)
    _track_matrix(self)


def _patched_causal_matrix_random(
    N: int, density: float = 0.5, backing_file: Any = None, seed_override: int | None = None
):
    N = int(N)
    resolved = _resolve_backing_path(backing_file, fallback="random")
    if _should_register_auto(backing_file):
        _STORAGE_REGISTRY.register_auto_file(resolved)
    actual_seed = seed_override if seed_override is not None else seed
    matrix = _original_causal_matrix_random(N, density, resolved, actual_seed)
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
    for col in col_head:
        entries.append(_format_value(matrix.get(row_index, col)))
    if truncated:
        entries.append("...")
    for col in col_tail:
        entries.append(_format_value(matrix.get(row_index, col)))
    return " ".join(entries)


def _matrix_str(self) -> str:
    size = self.size()
    header = f"{self.__class__.__name__}(shape=({size}, {size}))"
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

_CausalMatrix.__str__ = _matrix_str


class MatrixMixin:
    def __str__(self) -> str:
        return _matrix_str(self)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} shape={self.shape}>"


class Matrix(MatrixMixin, metaclass=abc.ABCMeta):
    """Base class for all matrix types in pycauset.
    
    Also serves as an in-memory dense matrix implementation that accepts integers, lists, or NumPy arrays.
    """

    def __init__(self, source: Any, saveas: Any = None, *, populate: bool = True):
        if saveas not in (None, ""):
            warnings.warn(
                "pycauset.Matrix does not persist to disk; use pycauset.CausalMatrix for storage.",
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
            if populate and source:
                self._populate_random()
        else:
            size, rows = _coerce_general_matrix(source)
            self._size = size
            self._data = rows

        _track_matrix(self)

    def _populate_random(self) -> None:
        rng = random.Random(seed) if seed is not None else random.Random()
        for i in range(self._size):
            for j in range(self._size):
                self._data[i][j] = 1 if rng.random() >= 0.5 else 0

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
for cls in (_CausalMatrix, _native.IntegerMatrix, _native.FloatMatrix, _native.TriangularFloatMatrix):
    cls.__str__ = MatrixMixin.__str__
    cls.__repr__ = MatrixMixin.__repr__
    Matrix.register(cls)

# Register triangular subclasses
TriangularMatrix.register(_CausalMatrix)
TriangularMatrix.register(_native.TriangularFloatMatrix)

# Patch CausalMatrix init/random
_CausalMatrix.__init__ = _patched_causal_matrix_init
_CausalMatrix.random = staticmethod(_patched_causal_matrix_random)

# Aliases
CausalMatrix = _CausalMatrix
IntegerMatrix = _native.IntegerMatrix
FloatMatrix = _native.FloatMatrix
TriangularFloatMatrix = _native.TriangularFloatMatrix

_native_matmul = getattr(_native, "matmul")

def matmul(a: Any, b: Any, saveas: str | None = None) -> Any:
    """
    Perform matrix multiplication.
    
    If both inputs are CausalMatrices, uses the optimized C++ implementation.
    Otherwise, performs generic multiplication (slow).
    """
    if isinstance(a, _CausalMatrix) and isinstance(b, _CausalMatrix):
        resolved = _resolve_backing_path(saveas, fallback="matmul")
        if _should_register_auto(saveas):
            _STORAGE_REGISTRY.register_auto_file(resolved)
        return _native_matmul(a, b, resolved)
    
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
    res = Matrix(N, populate=False) 
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

def compute_k(matrix: CausalMatrix, a: float, saveas: str | None = None):
    """
    Compute K = C(aI + C)^-1.
    
    Args:
        matrix: The CausalMatrix C.
        a: The scalar a.
        saveas: Optional path to save the result. If None, a temporary file is used.
        
    Returns:
        A FloatMatrix representing K.
    """
    resolved = _resolve_backing_path(saveas, fallback="k_matrix")
    if _should_register_auto(saveas):
        _STORAGE_REGISTRY.register_auto_file(resolved)
        
    _native.compute_k_matrix(matrix, a, resolved, 0)
    return _native.FloatMatrix(matrix.size(), resolved)


def __getattr__(name):
    return getattr(_native, name)


__all__ = [name for name in dir(_native) if not name.startswith("__")]
__all__.extend(["save", "seed", "Matrix", "TriangularMatrix", "CausalMatrix", "compute_k"])
