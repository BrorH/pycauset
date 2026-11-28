# pycauset.matmul

```python
pycauset.matmul(a, b)
```

Perform matrix multiplication.

For native objects, the engine entry `pycauset::matmul` picks the kernel by structure and dtype (`dispatch_matmul`: bit-packed, triangular, diagonal, identity and dense BLAS kernels).

For block matrices (constructed via `pycauset.matrix(block_grid)` where every element is matrix-like), `pycauset.matmul(a, b)` preserves “once block, always block” by returning a thunked block-matrix result with partition refinement (union of shared boundaries via `SubmatrixView` tiling).

This function is NumPy-like:

- matrix-matrix: `(m, k) @ (k, n) -> (m, n)`
- matrix-vector: `(m, k) @ (k,) -> (m,)`
- vector-matrix: `(k,) @ (k, n) -> (1, n)` (row-vector semantics)
- vector-vector: `(k,) @ (k,) -> scalar` (dot)

Bit operands (`bool`/`bit`, i.e. `DenseMatrix` (dtype `bool`) or `TriangularMatrix` (dtype `bool`)) promote
to `int32` on matmul because a bit-bit reduction can exceed `{0,1}`. This includes
the mixed `DenseMatrix (bool) @ TriangularMatrix (bool)` case (and its reverse).

## Result structure

The values are always NumPy's (`a @ b` on the same arrays); the result dtype follows the
promotion rules. The result keeps a structure when the product has it:

| Operands | Result |
|---|---|
| identity `@` X, X `@` identity | X scaled (X's structure) |
| upper triangular `@` upper triangular (real dtypes except float16) | `TriangularMatrix`; it has a diagonal only if both operands do |
| causal (strictly upper `bool`) `@` causal | `TriangularMatrix` (int32 path counts) |
| diagonal `@` diagonal (dtypes with a `DiagonalMatrix`) | `DiagonalMatrix` |
| anything else (including a lower triangular `T.T` operand) | dense |

Structured products skip structural zeros, so they differ from NumPy only where a skipped
`0 * inf` or `0 * nan` would have produced `nan`. Integer products raise `OverflowError` when
a result does not fit the result dtype.

## Parameters

*   **a** (*MatrixBase or VectorBase or BlockMatrix*): Left operand.
*   **b** (*MatrixBase or VectorBase or BlockMatrix*): Right operand.

Shape rule: `a.cols() == b.rows()`.

## Returns

*   **MatrixBase or VectorBase or scalar**: The result. The specific type and shape depend on input ranks.

When either operand is a `BlockMatrix`, the result is a `BlockMatrix` (typically holding lazy `ThunkBlock` output blocks).

## Evaluation (block matrices)

Block-matrix results are **semi-lazy**:

- Triggers (evaluate minimal required blocks): element access (e.g. `C[i, j]`), conversion to NumPy (`np.asarray(C)`), persistence (`pycauset.save(C, ...)`), and crossing the compute boundary.
- Non-triggers: `repr(C)`, `str(C)`, and partition metadata access.

Determinism:

- Fixed `k` order for each output block’s accumulation.
- Accumulator dtype is chosen from operand dtype metadata only (folding add-result dtype across terms) before evaluation.
- Each leaf product is an ordinary `pycauset.matmul`.

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
