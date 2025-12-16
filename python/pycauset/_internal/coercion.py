from __future__ import annotations

from collections.abc import Sequence as _SequenceABC
from typing import Any


def is_sequence_like(value: Any) -> bool:
    return isinstance(value, _SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def coerce_sequence_rows(candidate: Any) -> tuple[int, list[list[Any]]]:
    if not is_sequence_like(candidate):
        raise TypeError(
            "Matrix data must be provided as a square nested sequence or a NumPy array."
        )
    rows = [list(row) for row in candidate]
    if not rows:
        raise ValueError("Matrix data must not be empty.")
    size = len(rows)
    for row in rows:
        if not is_sequence_like(row):
            raise TypeError("Each matrix row must be a sequence of entries.")
        if len(row) != size:
            raise ValueError(
                "Matrix data must describe a square matrix (same number of rows and columns)."
            )
    return size, rows


def coerce_general_matrix(candidate: Any, *, np_module: Any | None) -> tuple[int, list[list[Any]]]:
    size_attr: Any = getattr(candidate, "size", None)
    get_attr: Any = getattr(candidate, "get", None)
    if callable(size_attr) and callable(get_attr):
        size = int(size_attr())  # type: ignore[arg-type]
        if size < 0:
            raise ValueError("Matrix size must be non-negative.")
        rows: list[list[Any]] = []
        for i in range(size):
            row: list[Any] = []
            for j in range(size):
                row.append(get_attr(i, j))
            rows.append(row)
        return size, rows

    if np_module is not None:
        try:
            array = np_module.asarray(candidate)
        except Exception:
            array = None
        else:
            if array.ndim != 2:
                raise ValueError("Matrix input must be a 2D square structure.")
            if array.shape[0] != array.shape[1]:
                raise ValueError("Matrix input must be square (rows == columns).")
            return int(array.shape[0]), array.tolist()

    return coerce_sequence_rows(candidate)
