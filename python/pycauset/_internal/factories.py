from __future__ import annotations

import warnings
from typing import Any, Callable

from . import export_guard as _export_guard
from .dtypes import normalize_dtype as _normalize_dtype
from .native_threshold import temporary_native_memory_threshold


def vector_factory(
    size_or_data: Any,
    *,
    dtype: Any,
    np_module: Any | None,
    native: Any,
    kwargs: dict[str, Any],
    max_in_ram_bytes: int | None = None,
) -> Any:
    """Build a DenseVector from a size or 1-D data (dtype inferred when not given)."""
    dtype_norm = _normalize_dtype(dtype, np_module=np_module)
    ctor_kwargs: dict[str, Any] = dict(kwargs)
    limit = ctor_kwargs.pop("max_in_ram_bytes", max_in_ram_bytes)

    # 1) Creation by size
    if isinstance(size_or_data, (int, float)) and (
        isinstance(size_or_data, int) or size_or_data.is_integer()
    ):
        return native.DenseVector(int(size_or_data), dtype=dtype_norm or "float64", **ctor_kwargs)

    data: Any = size_or_data

    # 2) NumPy: real dtypes go through the native bulk import; complex is copied element-wise
    if isinstance(data, np_module.ndarray):
        if dtype_norm is None and data.dtype.kind != "c":
            if limit is not None:
                est = _export_guard.estimate_materialized_bytes(data)
                if est is not None and est > limit:
                    with temporary_native_memory_threshold(native, int(limit)):
                        return native.asarray(data)
            return native.asarray(data)
        if dtype_norm is None:
            dtype_norm = _normalize_dtype(data.dtype, np_module=np_module)
        data = data.tolist()

    # 3) Infer dtype from Python values
    if dtype_norm is None:
        dtype_norm = "float64"
        if any(isinstance(x, complex) for x in data):
            dtype_norm = "complex_float64"
        elif all(isinstance(x, (int, bool)) for x in data):
            dtype_norm = "int32"
        if all(isinstance(x, bool) for x in data):
            dtype_norm = "bool"

    vec = native.DenseVector(len(data), dtype=dtype_norm, **ctor_kwargs)
    for i, val in enumerate(data):
        vec[i] = val
    return vec


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

