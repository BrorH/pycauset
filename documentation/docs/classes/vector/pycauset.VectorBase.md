# pycauset.VectorBase

Base class for all vector types.

Vectors are persistent objects (RAM-backed for small sizes; disk-backed for large sizes) and integrate with matrix operations.

## Shape and transposes

- A non-transposed vector behaves like a 1D array with shape `(n,)`.
- `v.T` returns a transposed view (row-vector semantics).
- For complex vector types, `v.H` returns the conjugate-transpose view.

## Properties

### `properties`

Vectors expose `v.properties`, a typed mapping used for:

- user metadata and assertions (e.g. `is_sorted`), saved with the vector. They are not
  checked against the values and never change results; contradictory assertions raise.

See [[guides/Storage and Memory.md|Storage and Memory]].

## Fill

### `fill(value)`

Fill the vector with a scalar value.

This is an explicit full write. On very large disk-backed vectors, this can be a long I/O operation.

## Common operations

### Dot product

```python
v.dot(other) -> int | float | complex
```

Computes the dot product: an exact `int` for integer/bool vectors (`OverflowError` beyond
128 bits), a `float` for real float vectors, a `complex` if either is complex.

Notes:

- No implicit conjugation is applied. For a Hermitian-style inner product, use `v.conj().dot(other)` or `v.H @ other`.

### Matrix multiplication operator (`@`)

Vector `@` uses NumPy-like semantics:

- `v @ w` (vector @ vector) returns a scalar dot product.
- `v @ w.T` (column @ row) returns an outer-product matrix.

### Arithmetic

`+`, `-`, `*`, `/` with another vector (same length), a 1-D NumPy array or a scalar, on
either side, and unary `-`. Element-wise, with the same result dtypes as the equivalent
1 x n matrix operation (NumPy's rules; see the [[guides/Vector Guide|Vector Guide]]).

## NumPy interoperability

### `__array__()`

Converts the vector to a NumPy array.

Note: this materializes the full vector in memory.

## See also

- [[docs/functions/pycauset.dot.md|pycauset.dot]]
- [[docs/functions/pycauset.sum.md|pycauset.sum]]
- [[docs/functions/pycauset.vector.md|pycauset.vector]]
- [[guides/Vector Guide|Vector Guide]]

