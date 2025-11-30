# pycauset.IdentityMatrix

```python
class pycauset.IdentityMatrix(n)
```

A memory-efficient representation of an Identity Matrix. It stores no data on disk (only a header) and generates values on the fly ($1.0$ on the diagonal, $0.0$ elsewhere).

Inherits from [[pycauset.MatrixBase]].

## Parameters

*   **n** (*int*): The dimension of the matrix ($N \times N$).

## Properties

*   **shape** (*tuple*): The dimensions of the matrix `(N, N)`.
*   **size** (*int*): The dimension $N$ of the matrix.
*   **scalar** (*float*): The scaling factor of the matrix. Default is 1.0.

## Methods

### `get(i, j)`
Returns the value at row `i` and column `j`.
*   Returns `scalar` if $i == j$.
*   Returns `0.0` if $i \neq j$.

### `multiply(other)`
Multiplies this matrix by another matrix.
*   If **other** is an `IdentityMatrix`, returns a new `IdentityMatrix` with multiplied scalars.
*   If **other** is another matrix type, performs standard matrix multiplication (result type depends on operands).

### `add(other)`
Adds another matrix to this one.
*   If **other** is an `IdentityMatrix`, returns a new `IdentityMatrix` with added scalars.

### `subtract(other)`
Subtracts another matrix from this one.
*   If **other** is an `IdentityMatrix`, returns a new `IdentityMatrix` with subtracted scalars.

### `elementwise_multiply(other)`
Performs elementwise multiplication.
*   If **other** is an `IdentityMatrix`, returns a new `IdentityMatrix` with multiplied scalars (since off-diagonals are 0).

### `multiply_scalar(factor)`
Multiplies the matrix by a scalar factor.
*   **factor** (*float*): The value to multiply by.
*   **Returns**: A new `IdentityMatrix` with updated scalar.
