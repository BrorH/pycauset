# pycauset.matmul

```python
pycauset.matmul(a, b)
```

Perform matrix multiplication.

For native objects, this routes through the standard compute boundary (AutoSolver / device routing) and dispatches to optimized C++ implementations.

For block matrices (constructed via `pycauset.matrix(block_grid)` where every element is matrix-like), `pycauset.matmul(a, b)` preserves “once block, always block” by returning a thunked block-matrix result.

This function is NumPy-like:

- matrix-matrix: `(m, k) @ (k, n) -> (m, n)`
- matrix-vector: `(m, k) @ (k,) -> (m,)`
- vector-matrix: `(k,) @ (k, n) -> (1, n)` (row-vector semantics)
- vector-vector: `(k,) @ (k,) -> scalar` (dot)

## Parameters

*   **a** (*MatrixBase or VectorBase or BlockMatrix*): Left operand.
*   **b** (*MatrixBase or VectorBase or BlockMatrix*): Right operand.

Shape rule: `a.cols() == b.rows()`.

## Returns

*   **MatrixBase or VectorBase or scalar**: The result. The specific type and shape depend on input ranks.

When either operand is a `BlockMatrix`, the result is a `BlockMatrix` (typically holding lazy `ThunkBlock` output blocks).

## Evaluation (block matrices)

Block-matrix results are **semi-lazy**:

- Triggers (evaluate minimal required blocks): element access (e.g. `C[i, j]`), conversion to NumPy (`np.asarray(C)`), and persistence (`pycauset.save(C, ...)`).
- Non-triggers: `repr(C)`, `str(C)`, and partition metadata access.

See [[internals/Block Matrices.md|Block Matrices]] for details.

## Examples

```python
import pycauset as pc

# Dense matmul
A = pc.matrix(((1.0, 2.0), (3.0, 4.0)))
B = pc.matrix(((5.0,), (6.0,)))
C = pc.matmul(A, B)
assert C.shape == (2, 1)

# Block matmul (returns a BlockMatrix)
blk = pc.matrix(((1.0, 0.0), (0.0, 1.0)))
BM = pc.matrix(((blk, blk), (blk, blk)))
out = pc.matmul(BM, BM)

# Accessing an element triggers evaluation of the needed output block
_ = out[0, 0]
```

## See also

*   [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]]
*   [[docs/functions/pycauset.matrix.md|pycauset.matrix]]
*   [[guides/Matrix Guide|Matrix Guide]]
*   [[internals/Block Matrices.md|Block Matrices]]
