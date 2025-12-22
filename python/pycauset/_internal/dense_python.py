from __future__ import annotations

from typing import Any


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
    return obj[i, j]


class DensePythonMatrix:
    """Small dense matrix used as an internal fallback.

    This exists to keep Phase D orchestration functional when blocks are Python
    views (e.g. `SubmatrixView`) that cannot be consumed by native ops.

    It is correctness-first and intentionally not optimized.
    """

    def __init__(self, data: list[list[Any]]):
        if not data or not data[0]:
            raise ValueError("DensePythonMatrix data must be non-empty")
        cols = len(data[0])
        if any(len(r) != cols for r in data):
            raise ValueError("DensePythonMatrix requires rectangular data")
        self._data = data

    def rows(self) -> int:
        return len(self._data)

    def cols(self) -> int:
        return len(self._data[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows(), self.cols())

    def get(self, i: int, j: int) -> Any:
        return self._data[i][j]

    def __getitem__(self, key: Any) -> Any:
        i, j = key
        return self.get(int(i), int(j))

    def __add__(self, other: Any) -> "DensePythonMatrix":
        shape = _safe_rows_cols(other)
        if shape is None or shape != self.shape:
            raise TypeError("add expects a matrix-like operand with matching shape")
        out: list[list[Any]] = []
        for i in range(self.rows()):
            out.append([self.get(i, j) + _safe_get(other, i, j) for j in range(self.cols())])
        return DensePythonMatrix(out)

    def __matmul__(self, other: Any) -> "DensePythonMatrix":
        shape = _safe_rows_cols(other)
        if shape is None:
            raise TypeError("matmul expects a matrix-like operand")
        r, k = self.shape
        k2, c = shape
        if k != k2:
            raise ValueError("matmul dimension mismatch")
        out: list[list[Any]] = []
        for i in range(r):
            row: list[Any] = []
            for j in range(c):
                acc = None
                for kk in range(k):
                    term = self.get(i, kk) * _safe_get(other, kk, j)
                    acc = term if acc is None else (acc + term)
                row.append(acc)
            out.append(row)
        return DensePythonMatrix(out)

    def __repr__(self) -> str:
        return f"DensePythonMatrix(shape={self.shape})"
