from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dense_python import DensePythonMatrix


def _safe_rows_cols(obj: Any) -> tuple[int, int] | None:
    try:
        return int(obj.rows()), int(obj.cols())
    except Exception:
        try:
            shape = getattr(obj, "shape", None)
            if isinstance(shape, tuple) and len(shape) == 2:
                return int(shape[0]), int(shape[1])
        except Exception:
            pass
    return None


def _safe_get(obj: Any, i: int, j: int) -> Any:
    get_fn = getattr(obj, "get", None)
    if callable(get_fn):
        return get_fn(i, j)
    try:
        return obj[i, j]
    except Exception as e:
        raise TypeError("source does not support element access via get(i,j) or [i,j]") from e


def _dense_from_get(obj: Any) -> DensePythonMatrix:
    shape = _safe_rows_cols(obj)
    if shape is None:
        raise TypeError("expected a matrix-like object")
    r, c = shape
    data: list[list[Any]] = []
    for i in range(r):
        data.append([_safe_get(obj, i, j) for j in range(c)])
    return DensePythonMatrix(data)


@dataclass(frozen=True)
class _Rect:
    row0: int
    col0: int
    rows: int
    cols: int


class SubmatrixView:
    """A no-copy rectangular view into a matrix-like source.

    Phase C scope:
    - Construction validation (bounds, shape)
    - Element access by delegating into the source
    - View composition: a view-of-a-view collapses into a single view
    - Structure-only printing (repr/str must not access elements)

    This is internal and intentionally minimal.
    """

    def __init__(self, source: Any, row0: int, col0: int, rows: int, cols: int):
        if not all(isinstance(x, int) for x in (row0, col0, rows, cols)):
            raise TypeError("SubmatrixView requires integer row0/col0/rows/cols")
        if row0 < 0 or col0 < 0:
            raise ValueError("SubmatrixView offsets must be non-negative")
        if rows < 0 or cols < 0:
            raise ValueError("SubmatrixView shape must be non-negative")

        # Compose views deterministically.
        if isinstance(source, SubmatrixView):
            base = source.source
            row0 = source.row_offset + row0
            col0 = source.col_offset + col0
            source = base

        shape = _safe_rows_cols(source)
        if shape is None:
            raise TypeError("SubmatrixView source must be matrix-like (rows/cols or shape)")
        src_rows, src_cols = shape

        if row0 + rows > src_rows or col0 + cols > src_cols:
            raise ValueError("SubmatrixView rectangle is out of bounds")

        self._source = source
        self._rect = _Rect(row0=row0, col0=col0, rows=rows, cols=cols)

    # --- minimal matrix protocol ---

    def rows(self) -> int:
        return self._rect.rows

    def cols(self) -> int:
        return self._rect.cols

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows(), self.cols())

    @property
    def source(self) -> Any:
        return self._source

    @property
    def row_offset(self) -> int:
        return self._rect.row0

    @property
    def col_offset(self) -> int:
        return self._rect.col0

    @property
    def dtype(self) -> Any:
        # Best-effort: preserve dtype label if present.
        return getattr(self._source, "dtype", None)

    def get(self, i: int, j: int) -> Any:
        if not (isinstance(i, int) and isinstance(j, int)):
            raise TypeError("indices must be integers")
        if i < 0 or j < 0 or i >= self.rows() or j >= self.cols():
            raise IndexError("index out of range")
        return _safe_get(self._source, self.row_offset + i, self.col_offset + j)

    def __getitem__(self, key: Any) -> Any:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("matrix indices must be provided as [row, col]")
        i, j = key
        return self.get(int(i), int(j))

    # --- ops (Phase D fallback) ---

    def __add__(self, other: Any) -> Any:
        # Block-local materialization only; no global densification.
        return _dense_from_get(self) + other

    def __matmul__(self, other: Any) -> Any:
        # Block-local materialization only; no global densification.
        return _dense_from_get(self) @ other

    # --- printing (structure-only; must not access elements) ---

    def __repr__(self) -> str:
        return self._format()

    def __str__(self) -> str:
        return self._format()

    def _format(self) -> str:
        src_shape = _safe_rows_cols(self._source)
        src_shape_s = "?x?" if src_shape is None else f"{src_shape[0]}x{src_shape[1]}"
        return (
            f"SubmatrixView(shape={self.shape}, offset=({self.row_offset},{self.col_offset}), "
            f"source={type(self._source).__name__}({src_shape_s}))"
        )
