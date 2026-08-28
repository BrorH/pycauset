"""Dtype-deferred allocation wrappers for zeros/ones/empty.

When `dtype` is omitted, `pycauset.zeros`, `pycauset.ones`, and `pycauset.empty`
return one of these lightweight wrappers instead of a concrete native object.
The wrapper carries only shape metadata plus a fill pattern (zero / one / none)
and no concrete dtype. It materializes into a concrete native object on the
first operation that requires a dtype.

Dtype resolution order:

- explicit write (`set` / `__setitem__` / `fill`): deduced from the written
  value's Python type (bool -> "bool", int -> "int32", float -> "float64",
  complex -> "complex_float64").
- standalone read / export / str: `zeros`/`ones` resolve to int32 (deduced from
  their fill value 0 / 1); `empty` raises (no silent wrong answer).
- binary op: `zeros`/`ones` materialize to int32, except that a matmul against a
  native bit matrix materializes them as a bit matrix so the native bit-matmul
  path applies; `empty` raises.

This module is intentionally dependency-free (no NumPy import) so the deduction
and allocation decisions stay in the package facade.
"""

from __future__ import annotations

from typing import Any, Callable


def _is_bit_matrix(obj: Any) -> bool:
    name = type(obj).__name__
    return name == "DenseBitMatrix" or name == "TriangularBitMatrix"


class LazyAllocated:
    """A shape-only allocation whose concrete dtype is resolved on first use."""

    def __init__(
        self,
        *,
        kind: str,
        shape: tuple[int, ...],
        ndim: int,
        materialize: Callable[[str], Any],
        deduce_dtype: Callable[[Any], str],
    ) -> None:
        # kind is one of "zeros", "ones", "empty".
        self._kind = kind
        self._shape = shape
        self._ndim = ndim
        self._materialize_fn = materialize
        self._deduce_dtype_fn = deduce_dtype
        self._impl: Any | None = None
        self._dtype: str | None = None
        self.properties: dict[str, Any] = self._make_properties()

    def _make_properties(self) -> dict[str, Any]:
        if self._kind == "zeros":
            return {"is_zero": True}
        if self._kind == "ones":
            return {"is_constant": True, "constant_value": 1}
        return {}

    # --- metadata (no materialization) ---
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dtype(self) -> str | None:
        return self._dtype

    @property
    def kind(self) -> str:
        return self._kind

    def rows(self) -> int:
        return int(self._shape[0])

    def cols(self) -> int:
        return int(self._shape[1]) if self._ndim == 2 else 1

    def __len__(self) -> int:
        return int(self._shape[0])

    # --- materialization ---
    def _materialize_impl(self, dtype: str) -> Any:
        if self._impl is None:
            self._impl = self._materialize_fn(dtype)
            self._dtype = dtype
        return self._impl

    def _materialize(self, dtype: str | None = None) -> Any:
        """Read/op materialization.

        `zeros`/`ones` resolve to int32 (or to the hinted dtype); a still-typeless
        `empty` raises.
        """
        if self._impl is not None:
            return self._impl
        if self._kind == "empty":
            raise TypeError(
                "pycauset.empty() has no dtype yet; write a value first or pass dtype= explicitly"
            )
        return self._materialize_impl("int32" if dtype is None else dtype)

    def _materialize_from_write(self, value: Any) -> Any:
        if self._impl is not None:
            return self._impl
        return self._materialize_impl(self._deduce_dtype_fn(value))

    # --- writes (deduce dtype from the written value) ---
    def set(self, *args: Any) -> Any:
        value = args[-1]
        impl = self._materialize_from_write(value)
        return impl.set(*args)

    def __setitem__(self, key: Any, value: Any) -> None:
        impl = self._materialize_from_write(value)
        impl[key] = value

    def fill(self, value: Any) -> "LazyAllocated":
        self._materialize_from_write(value)
        self._impl.fill(value)
        return self

    # --- reads (resolve default dtype for zeros/ones, error for empty) ---
    def get(self, *args: Any) -> Any:
        return self._materialize().get(*args)

    def __getitem__(self, key: Any) -> Any:
        return self._materialize()[key]

    # --- export ---
    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        import numpy as np  # type: ignore

        impl = self._materialize()
        arr = np.asarray(impl)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        if copy:
            arr = arr.copy()
        return arr

    # --- binary operators ---
    def __matmul__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        self_dtype = "bool" if _is_bit_matrix(other_impl) else None
        return self._materialize(self_dtype) @ other_impl

    def __rmatmul__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        self_dtype = "bool" if _is_bit_matrix(other_impl) else None
        return other_impl @ self._materialize(self_dtype)

    def __add__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        return self._materialize() + other_impl

    def __radd__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        return other_impl + self._materialize()

    def __sub__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        return self._materialize() - other_impl

    def __rsub__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        return other_impl - self._materialize()

    def __mul__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        return self._materialize() * other_impl

    def __rmul__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        return other_impl * self._materialize()

    def __truediv__(self, other: Any) -> Any:
        other_impl = other._materialize() if isinstance(other, LazyAllocated) else other
        return self._materialize() / other_impl

    def __neg__(self) -> Any:
        return -self._materialize()

    def __pos__(self) -> Any:
        return +self._materialize()

    # --- repr/str ---
    def __repr__(self) -> str:
        dtype = self._dtype if self._dtype is not None else "<unresolved>"
        return f"LazyAllocated(kind={self._kind!r}, shape={self._shape!r}, dtype={dtype!r})"

    def __str__(self) -> str:
        return str(self._materialize())
