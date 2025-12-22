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

- gospel assertions (e.g. ordering-related hints such as `is_sorted`), and
- cached-derived values (e.g. `norm`, `sum`) with strict validity.

Gospel assertions are authoritative (not truth-validated), and incompatible asserted states raise immediately (no payload scan).

See [[guides/release1/properties.md|R1 Properties]] and [[guides/Storage and Memory.md|Storage and Memory]].

## Fill

### `fill(value)`

Fill the vector with a scalar value.

This is an explicit full write. On very large disk-backed vectors, this can be a long I/O operation.

## Common operations

### Dot product

```python
v.dot(other) -> float | complex
```

Computes the dot product. For complex vectors this returns a complex number.

Notes:

- No implicit conjugation is applied. For a Hermitian-style inner product, use `v.conj().dot(other)` or `v.H @ other`.

### Matrix multiplication operator (`@`)

Vector `@` uses NumPy-like semantics:

- `v @ w` (vector @ vector) returns a scalar dot product.
- `v @ w.T` (column @ row) returns an outer-product matrix.

### Scalar arithmetic

Vectors support scalar arithmetic via Python operators:

- `v + s` / `s + v`
- `v * s` / `s * v`

Supported scalar types depend on the vector dtype (real vs complex).

## NumPy interoperability

### `__array__()`

Converts the vector to a NumPy array.

Note: this materializes the full vector in memory.

## See also

- [[docs/functions/pycauset.dot.md|pycauset.dot]]
- [[docs/functions/pycauset.sum.md|pycauset.sum]]
- [[docs/functions/pycauset.vector.md|pycauset.vector]]
- [[guides/Vector Guide|Vector Guide]]

