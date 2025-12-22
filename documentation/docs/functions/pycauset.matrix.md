# pycauset.matrix

```python
pycauset.matrix(source, dtype=None, **kwargs)
```

Create a 2D matrix from matrix-like input.

This is a data constructor (aligned with `np.array(...)` semantics). 

If `source` is 2D, returns a matrix. If `source` is 1D, returns a vector.

!!! warning "Pre-alpha note (block matrices)"
		`pycauset.matrix(...)` also supports constructing a block matrix when given a 2D grid of matrix objects.
		This feature is **pre-alpha** and currently returns a Python `BlockMatrix`.

Rectangular shapes are supported for dense matrices, including numeric dtypes (int/uint/float/complex) and boolean/bit matrices. Boolean 2D inputs use bit-packed storage (`DenseBitMatrix`).

## Block-grid construction (experimental)

For a 2D nested sequence (e.g. list-of-lists), `pycauset.matrix(...)` disambiguates between:

- **Dense data constructor**: a 2D grid of numeric scalars (no matrix objects).
- **Block matrix constructor**: a 2D grid where **every element is matrix-like**.
- **Error**: a 2D grid that **mixes** matrices and scalars.

The “matrix-like” check accepts native matrices plus block-matrix helper objects (views/thunks).

Rules:

- If any element is matrix-like and not all elements are matrix-like: raise `TypeError` (ambiguous input).
- If all elements are matrix-like: return a `BlockMatrix`.
	- `dtype` and `**kwargs` are rejected for block-grid input.

## Parameters

*   **source** (*sequence or numpy.ndarray*): 1D nested data (e.g. list) / 1D NumPy array, or 2D nested data (e.g. list-of-lists) / 2D NumPy array.
*   **dtype** (*str or type, optional*): Coerce storage dtype (e.g. `"float64"`, `"int32"`, `float`, `int`).
*   **kwargs**: Passed through to the backend constructor.

## Returns

*   **MatrixBase or VectorBase or BlockMatrix**: A concrete native matrix/vector for numeric data input, or a block matrix when `source` is a 2D grid of matrices.

	For block-grid input, the return is a `BlockMatrix`.

## Examples

```python
import pycauset

m = pycauset.matrix(((1, 2), (3, 4)))

# 1D input returns a vector
v = pycauset.matrix((1, 2, 3))

# Coerce dtype
m_f32 = pycauset.matrix(((1, 2), (3, 4)), dtype="float32")

# Block matrix (2D grid of matrices)
A = pycauset.matrix(((1.0, 0.0), (0.0, 1.0)))
B = pycauset.matrix(((2.0, 3.0), (4.0, 5.0)))
BM = pycauset.matrix(((A, B), (B, A)))

# Mixed matrices + scalars is rejected
try:
	pycauset.matrix(((A, 0.0),))
except TypeError:
	pass
```

## See also

*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
*   [[guides/Matrix Guide|Matrix Guide]]
*   [[internals/Block Matrices.md|Block Matrices]]
