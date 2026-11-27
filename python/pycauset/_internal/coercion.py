from __future__ import annotations

from collections.abc import Sequence as _SequenceABC
from typing import Any


def is_sequence_like(value: Any) -> bool:
    return isinstance(value, _SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def safe_rows_cols(obj: Any) -> tuple[int, int] | None:
    """(rows, cols) of a matrix-like (`rows()`/`cols()` or a 2-tuple `shape`), else None."""
    if callable(getattr(obj, "rows", None)) and callable(getattr(obj, "cols", None)):
        return int(obj.rows()), int(obj.cols())
    shape = getattr(obj, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None
