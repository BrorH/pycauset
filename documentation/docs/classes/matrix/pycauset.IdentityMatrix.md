# pycauset.IdentityMatrix

```python
class pycauset.IdentityMatrix(x)
class pycauset.IdentityMatrix(n)
class pycauset.IdentityMatrix(rows, cols)
```

A memory-efficient representation of an Identity Matrix. It stores no data on disk (only a header) and generates values on the fly ($1.0$ on the diagonal, $0.0$ elsewhere).

Inherits from [[pycauset.MatrixBase]].

## Parameters

*   **x** (*int | [rows, cols] | Matrix | Vector*): Convenience form. Accepts:
	- `N` (int) -> $N \times N$
	- `[rows, cols]` -> `rows × cols`
	- matrix -> `(x.rows(), x.cols())`
	- vector -> $N \times N$ where $N = x.size()$
*   **n** (*int*): The dimension of the matrix ($N \times N$).
*   **rows** (*int*): Number of rows.
*   **cols** (*int*): Number of columns.

`pycauset.identity(x)` is an equivalent lower-case convenience factory.

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(rows, cols)`.
*   **size()** (*int*): Total element count ($rows \times cols$).
*   **scalar** (*float*): The scaling factor of the matrix. Default is 1.0.

## Methods

### Indexing

Element reads use NumPy-style indexing: `x = I[i, j]`.

*   Returns `scalar` if $i == j$ and $i < \min(rows, cols)$.
*   Returns `0.0` if $i \neq j$.

### `multiply(other)`
Multiplies this matrix by another matrix.
*   If **other** is an `IdentityMatrix`, returns a new `IdentityMatrix` with multiplied scalars.
*   If **other** is another matrix type, performs standard matrix multiplication (result type depends on operands).

## Notes

There is no separate integer `IdentityMatrix` variant in the public Python API.

```python
# (no IdentityMatrixInt public binding)
```

## Examples

```python
import pycauset as pc

# Explicit shape
I35 = pc.IdentityMatrix(3, 5)

# Shape derived from an existing object
A = pc.FloatMatrix(2, 4)
IA = pc.identity(A)          # 2x4 identity-like

v = pc.IntegerVector(7)
Iv = pc.identity(v)          # 7x7
```

## See also

*   [[docs/functions/pycauset.identity.md|pycauset.identity]]
*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
*   [[guides/Matrix Guide|Matrix Guide]]

