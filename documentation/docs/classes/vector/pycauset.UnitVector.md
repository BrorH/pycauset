# pycauset.UnitVector

A storage-less vector representing a standard basis vector $e_k$ (all zeros except a 1 at index $k$). Inherits from [VectorBase](pycauset.VectorBase.md).

This vector type uses 0 bytes of disk storage. It stores the active index $k$ in metadata.

## Constructor

```python
pycauset.UnitVector(n: int, active_index: int)
```

*   `n`: The size of the vector.
*   `active_index`: The index where the value is 1. Must be $0 \le k < n$.

## Properties

### `shape`
Returns a tuple representing the dimensions of the vector.

## Methods

### `__getitem__(i: int) -> float`
Get the value at index `i`. Returns 1.0 if `i == active_index`, else 0.0.

### `__len__() -> int`
Get the size of the vector.

### `__repr__() -> str`
String representation of the vector.

## Arithmetic

*   `UnitVector + UnitVector`:
    *   If active indices match: Returns a `UnitVector` (scaled).
    *   If active indices differ: Returns a `DenseVector` (e.g., $e_1 + e_2$).
*   `UnitVector + DenseVector`: Returns a `DenseVector`.
