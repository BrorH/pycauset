from __future__ import annotations

import warnings
from typing import Any

from .dtypes import infer_sequence_dtype as _infer_sequence_dtype


def vector_factory(
    size_or_data: Any,
    *,
    dtype: Any,
    np_module: Any | None,
    native: Any,
    kwargs: dict[str, Any],
    max_in_ram_bytes: int | None = None,
) -> Any:
    """A DenseVector from a size or 1-D data (dtype inferred when not given). The same rules as
    `matrix`: a bad dtype or an unknown keyword raises, `storage` and `max_in_ram_bytes` place
    it (they were ignored for NumPy input, and a bad dtype gave float64, review 3.8 C-3)."""
    from . import matrix_api

    target = matrix_api.dtype_or_raise(dtype)
    kw = dict(kwargs)
    limit = kw.pop("max_in_ram_bytes", max_in_ram_bytes)
    storage = kw.pop("storage", None)
    matrix_api.check_storage(storage)
    if kw:
        raise TypeError(f"vector() got unexpected keyword arguments: {sorted(kw)}")

    if isinstance(size_or_data, (int, float)) and not isinstance(size_or_data, bool) and (
        isinstance(size_or_data, int) or size_or_data.is_integer()
    ):
        return matrix_api.allocate((int(size_or_data),), target, storage=storage, max_in_ram_bytes=limit)

    data: Any = size_or_data
    if not isinstance(data, np_module.ndarray):
        values = list(data)
        target = target or _infer_sequence_dtype(values, np_module=np_module)
        data = np_module.array(values, dtype=matrix_api._NUMPY_DTYPE.get(target, "complex64"))
        if data.ndim != 1:
            raise TypeError("vector(...) expects 1D input; use matrix(...) for 2D")
    return matrix_api.from_array(data, target, storage=storage, max_in_ram_bytes=limit)


def causal_matrix_factory(
    source: Any,
    *,
    populate: bool,
    np_module: Any | None,
    TriangularMatrix: Any,
    kwargs: dict[str, Any],
) -> Any:
    source_obj: Any = source

    # Case 1: integer size
    if isinstance(source_obj, (int, float)) and (
        isinstance(source_obj, int) or source_obj.is_integer()
    ):
        n = int(source_obj)
        if populate:
            return TriangularMatrix.random(n, p=0.5, **kwargs)
        return TriangularMatrix(n, dtype="bool", **kwargs)

    # 2-D data (NumPy arrays, nested lists, PyCauset matrices): its strict upper triangle.
    source_obj = np_module.asarray(source_obj)
    if source_obj.ndim != 2 or source_obj.shape[0] != source_obj.shape[1]:
        raise ValueError("causal_matrix: data must be a square 2-D array")
    if source_obj.dtype != bool:  # type: ignore[attr-defined]
        if not np_module.all(np_module.isin(source_obj, [0, 1])):
            warnings.warn(
                "Input data contains non-binary values. They will be converted to boolean (True/False).",
                UserWarning,
                stacklevel=2,
            )

    if np_module.any(np_module.tril(source_obj) != 0):
        warnings.warn(
            "Input data contains non-zero values in the lower triangle or diagonal. "
            "A causal matrix is strictly upper triangular; these values will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    if source_obj.dtype != bool:  # type: ignore[attr-defined]
        source_obj = source_obj.astype(bool)  # type: ignore[attr-defined]
    size = int(source_obj.shape[0])
    matrix = TriangularMatrix(size, dtype="bool", **kwargs)
    rows_idx, cols_idx = np_module.nonzero(np_module.triu(source_obj, 1))
    for i, j in zip(rows_idx.tolist(), cols_idx.tolist()):
        matrix.set(i, j, True)
    return matrix

