# pycauset.matrix

```python
pycauset.matrix(source, dtype=None, **kwargs)
```

Create a 2D matrix from matrix-like input.

This is a data constructor (aligned with `np.array(...)` semantics). 

If `source` is 2D, returns a matrix. If `source` is 1D, returns a vector.

`pycauset.matrix(...)` also supports constructing a block matrix when given a 2D grid of matrix objects.

Rectangular shapes are supported for dense matrices, including numeric dtypes (int/uint/float/complex) and boolean/bit matrices. Boolean 2D inputs use bit-packed storage (`DenseMatrix` (dtype `bool`)).

## Block-grid construction

For a 2D nested sequence (e.g. list-of-lists), `pycauset.matrix(...)` disambiguates between:

- **Dense data constructor**: a 2D grid of numeric scalars (no matrix objects).
- **Block matrix constructor**: a 2D grid where **every element is matrix-like**.
- **Error**: a 2D grid that **mixes** matrices and scalars.

The “matrix-like” check accepts native matrices plus block-matrix helper objects (views/thunks).

Rules:

- If any element is matrix-like and not all elements are matrix-like: raise `TypeError` (ambiguous input).
- If all elements are matrix-like: return a `BlockMatrix`.
	- `dtype` and `**kwargs` are rejected for block-grid input.
	- Block-grid validation requires each block-row to share height and each block-col to share width; shape mismatches raise deterministically.
	- Block matrices support element access and `bm[r0:r1, c0:c1]` sub-grids (step 1, no copy).

## Parameters

*   **source** (*sequence or numpy.ndarray*): 1D nested data (e.g. list) / 1D NumPy array, or 2D nested data (e.g. list-of-lists) / 2D NumPy array.
*   **dtype** (*str or type, optional*): Coerce storage dtype (e.g. `"float64"`, `"int32"`, `float`, `int`). Without it, a NumPy array keeps its dtype, and Python integers become **int32** (NumPy: int64), as do `int` and `"int"`; see [[internals/DType System|DType System]] §3.2.
*   **storage** (*str, optional*): `"ram"` (default) keeps the matrix in RAM unless it is above the memory threshold or does not fit (then file-backed); `"disk"` places it in a file regardless of size, for any input (a size, a NumPy array, nested sequences). Any other value raises `ValueError`. For explicit persistence use [[pycauset.save]], and for the RAM / file threshold use [[pycauset.set_memory_threshold]].
*   **max_in_ram_bytes** (*int, optional*): a matrix larger than this goes to a file.
*   Other keywords raise `TypeError`.

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
