from __future__ import annotations

import warnings
import operator
from typing import Any

from .dtypes import infer_sequence_dtype as _infer_sequence_dtype
from .warnings import PyCausetDTypeWarning, PyCausetWarning


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
    native: Any,
    square_source: Any,
    kwargs: dict[str, Any],
) -> Any:
    source_obj: Any = source

    # Case 1: an integer size (Python or NumPy; a float is a TypeError, as NumPy's sizes:
    # causal_matrix(4.0) was accepted, review 3.8 C-20)
    if isinstance(source_obj, float):
        raise TypeError("causal_matrix(n): the size must be an integer, not a float")
    if not isinstance(source_obj, bool) and isinstance(source_obj, (int, np_module.integer)):
        n = operator.index(source_obj)
        if populate:
            return TriangularMatrix.random(n, p=0.5, **kwargs)
        return TriangularMatrix(n, dtype="bool", **kwargs)

    # 2-D data (NumPy arrays, nested lists, PyCauset matrices): its strict upper triangle,
    # checked and packed in the engine (no NumPy export of PyCauset input, review 3.8 C-4).
    if kwargs:
        raise TypeError(f"causal_matrix(data): unexpected keyword arguments {sorted(kwargs)} (they apply to a size)")
    try:
        m = square_source(source_obj, name="causal_matrix")
    except ValueError:
        raise ValueError("causal_matrix: data must be a square 2-D array") from None
    nonbinary, lower = native._causal_input_check(m)
    if nonbinary and m.dtype != "bool":
        warnings.warn(
            "Input data contains non-binary values. They will be converted to boolean (True/False).",
            PyCausetDTypeWarning,
            stacklevel=3,  # the caller of pc.causal_matrix (review 3.8 C-20)
        )
    if lower:
        warnings.warn(
            "Input data contains non-zero values in the lower triangle or diagonal. "
            "A causal matrix is strictly upper triangular; these values will be ignored.",
            PyCausetWarning,
            stacklevel=3,
        )
    bits = m if m.dtype == "bool" else (m != 0)
    return native._pack_as(bits, "triangular", False)

