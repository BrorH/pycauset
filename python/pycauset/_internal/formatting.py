from __future__ import annotations

from typing import Any


_np: Any | None = None
_EDGE_ITEMS: int = 4


def configure(*, np_module: Any | None, edge_items: int = 4) -> None:
    global _np, _EDGE_ITEMS
    _np = np_module
    _EDGE_ITEMS = int(edge_items)


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


def _format_matrix_row(
    matrix: Any,
    row_index: int,
    col_head: list[int],
    col_tail: list[int],
    truncated: bool,
) -> str:
    entries: list[str] = []

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


def matrix_str(self: Any) -> str:
    if hasattr(self, "rows") and hasattr(self, "cols"):
        rows = self.rows()
        cols = self.cols()
    else:
        # Backwards-compat: older builds exposed only size() as the square dimension.
        rows = self.size()
        cols = self.size()

    info = [f"shape=({rows}, {cols})"]

    if hasattr(self, "scalar") and self.scalar != 1.0:
        info.append(f"scalar={self.scalar}")

    if hasattr(self, "seed") and self.seed != 0:
        info.append(f"seed={self.seed}")

    header = f"{self.__class__.__name__}({', '.join(info)})"

    if rows == 0 or cols == 0:
        return header + "\n[]"

    row_head, row_tail, rows_truncated = _edge_indices(rows)
    col_head, col_tail, cols_truncated = _edge_indices(cols)

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


class MatrixMixin:
    def __str__(self) -> str:
        return matrix_str(self)

    def __repr__(self) -> str:
        shape = getattr(self, "shape", None)
        return f"<{self.__class__.__name__} shape={shape}>"
