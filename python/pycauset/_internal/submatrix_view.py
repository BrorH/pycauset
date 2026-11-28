from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

from .coercion import safe_rows_cols as _safe_rows_cols


def _safe_get(obj: Any, i: int, j: int) -> Any:
    get_fn = getattr(obj, "get", None)
    if callable(get_fn):
        return get_fn(i, j)
    return obj[i, j]


@dataclass(frozen=True)
class _Rect:
    row0: int
    col0: int
    rows: int
    cols: int


class SubmatrixView:
    """A no-copy rectangular view into a matrix-like source.

    - Construction validates the bounds.
    - Elements are read from the source; operations read the tile as an engine slice of its
      leaf (`blockmatrix._leaf_operand`).
    - A view of a view collapses into a single view.
    - Printing shows the structure only (repr/str must not access elements).
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

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        """The window's values, under the export guard (`export_to_numpy`)."""
        from . import export_guard

        return export_guard.export_to_numpy(self, allow_huge=False, dtype=dtype, copy=copy is not False)

    def _export_numpy(self) -> Any:
        """A native source exports its own window natively, other sources element by element."""
        import numpy as np

        from . import export_guard

        r, c = self._rect.row0, self._rect.col0
        h, w = self._rect.rows, self._rect.cols
        if hasattr(self._source, "_to_numpy_fast") and hasattr(self._source, "__getitem__"):
            return export_guard.export_part(self._source[r : r + h, c : c + w])
        return np.array([[self._source.get(r + i, c + j) for j in range(w)] for i in range(h)])

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
        i, j = operator.index(i), operator.index(j)
        if i < 0 or j < 0 or i >= self.rows() or j >= self.cols():
            raise IndexError("index out of range")
        return _safe_get(self._source, self.row_offset + i, self.col_offset + j)

    def __getitem__(self, key: Any) -> Any:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("matrix indices must be provided as [row, col]")
        i, j = key
        return self.get(int(i), int(j))

    # (Operators: a view tile is an operand of the block layer, which reads it as an engine
    # window, `blockmatrix._leaf_operand`. The element-by-element Python fallbacks are gone,
    # review 3.10 C-12.)

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
