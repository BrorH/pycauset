# Linear Algebra Operations

This guide shows how to use PyCauset’s linear algebra surface end-to-end: matrix math, solves/factorizations, spectral/SVD tools, stability utilities, and routing knobs.

## Shapes, dtypes, and storage

- Matrices are 2D only; vectors are treated as 1×N or N×1 matrices under the hood.
- Choose dtype at creation (`float64/32/16`, `int*`, `uint*`, `complex_float*`, `bit/bool`). No implicit promotion beyond the documented promotion rules.
- Large data may be memory-mapped to disk automatically; see [[guides/Storage and Memory|Storage and Memory]].

## Matrix construction (recap)

```python
import pycauset as pc
import numpy as np

A = pc.matrix([ [1., 2.], [3., 4.] ], dtype="float64")
B = pc.zeros((2, 2), dtype="float32")
C = pc.matrix(np.random.rand(2, 3))      # dtype inferred (float64)

```

See [[docs/functions/pycauset.matrix.md|pycauset.matrix]], [[docs/functions/pycauset.zeros.md|pycauset.zeros]], [[docs/functions/pycauset.ones.md|pycauset.ones]].

## Multiplication and elementwise ops

```python
M = pc.matrix([ [1, 2], [3, 4] ], dtype="float64")
N = pc.matrix([ [5, 6], [7, 8] ], dtype="float64")

P = M @ N                  # matmul
Q = pc.matmul(M, N)         # same as @
E = M * N                   # elementwise multiply
S = M + N                   # elementwise add
```
- Matmul shape rule: `M.cols() == N.rows()`; result `(M.rows(), N.cols())`.
- Elementwise ops follow NumPy 2D broadcasting.
- See [[docs/functions/pycauset.matmul.md|pycauset.matmul]], [[docs/functions/pycauset.sum.md|pycauset.sum]], [[docs/functions/pycauset.divide.md|pycauset.divide]].

## Solves and least squares

```python
A = pc.matrix([ [3., 1.], [1., 2.] ], dtype="float64")
b = pc.matrix([ [9.], [8.] ], dtype="float64")   # column vector (2×1)
x = pc.solve(A, b)

# Least squares
A_ls = pc.matrix([ [1., 1.], [1., 2.], [1., 3.] ], dtype="float64")
b_ls = pc.matrix([ [1.], [2.], [2.5] ], dtype="float64")
x_ls = pc.lstsq(A_ls, b_ls)
```
- `solve` requires square `A`; raises if singular/shape mismatch.
- `lstsq` works for over/underdetermined systems; returns best-fit.
- Triangular systems: use [[docs/functions/pycauset.solve_triangular.md|pycauset.solve_triangular]] for faster/safer solves on claimed triangular matrices.
- See [[docs/functions/pycauset.solve.md|pycauset.solve]], [[docs/functions/pycauset.lstsq.md|pycauset.lstsq]].

## Factorizations

```python
A = pc.matrix([ [4., 12., -16.], [12., 37., -43.], [-16., -43., 98.] ], dtype="float64")
L = pc.cholesky(A)     # lower-triangular

M = pc.matrix([ [1., 2.], [3., 4.] ], dtype="float64")
Llu, Ulu = pc.lu(M)
```
- `cholesky` expects Hermitian positive-definite; raises otherwise.
- `lu` factors square matrices; returns `(L, U)`.
- See [[docs/functions/pycauset.cholesky.md|pycauset.cholesky]], [[docs/functions/pycauset.lu.md|pycauset.lu]].

## Spectral (eigen) and SVD

```python
A = pc.matrix([ [0., -1.], [1., 0.] ], dtype="float64")
vals, vecs = pc.eig(A)

sym = pc.matrix([ [2., 1.], [1., 2.] ], dtype="float64")
vals_sym, vecs_sym = pc.eigh(sym)   # symmetric/Hermitian

U, s, Vh = pc.svd(sym)
pinv = pc.pinv(sym)
```
- Use `eig/eigvals` for general matrices; `eigh/eigvalsh` for symmetric/Hermitian (faster/stable).
- `svd` returns `(U, s, Vh)`; `pinv` uses SVD internally.
- See [[docs/functions/pycauset.eig.md|pycauset.eig]], [[docs/functions/pycauset.eigh.md|pycauset.eigh]], [[docs/functions/pycauset.eigvals.md|pycauset.eigvals]], [[docs/functions/pycauset.eigvalsh.md|pycauset.eigvalsh]], [[docs/functions/pycauset.svd.md|pycauset.svd]], [[docs/functions/pycauset.pinv.md|pycauset.pinv]].

## Inversion and stability utilities

```python
A = pc.matrix([ [4., 7.], [2., 6.] ], dtype="float64")
A_inv = pc.invert(A)

val, sign = pc.slogdet(A)
κ = pc.cond(A)
```
- `invert`/`inverse` require square, supported dtypes (float64/float32); errors on singular.
- `slogdet` returns `(sign, logabsdet)`; `cond` returns condition number.
- See [[docs/functions/pycauset.invert.md|pycauset.invert]], [[docs/functions/pycauset.slogdet.md|pycauset.slogdet]], [[docs/functions/pycauset.cond.md|pycauset.cond]].

## Dtype, precision, and device routing

- Respect user dtype; no silent promotion except documented mixes (e.g., float32+float64 → float64, int32 matmul accumulator warnings, bit→int32 promotions where necessary).
- Precision policy: set via [[docs/functions/pycauset.precision_mode.md|pycauset.precision_mode]] / [[docs/functions/pycauset.set_precision_mode.md|pycauset.set_precision_mode]].
- GPU routing is op- and dtype-dependent; when unsupported, the call falls back to CPU or raises a deterministic error. See [[guides/Performance Guide.md|Performance Guide]].

## Indexing/slicing recap

Dense matrices support NumPy-style indexing: basic slices are views; advanced (int/bool arrays) are copies; assignments broadcast and warn on dtype/overflow casts. Structured/triangular types reject slicing. See [[guides/Matrix Guide.md|Matrix Guide]] and [[docs/classes/matrix/pycauset.MatrixBase.md|pycauset.MatrixBase]].

## Persistence

- Matrices may spill to disk automatically; views share backing when using basic slices.
- Save/load with [[docs/functions/pycauset.save.md|pycauset.save]] and [[docs/functions/pycauset.load.md|pycauset.load]].
- Large-slice persistence policy for in-RAM sources is pending; current builds may error on unsupported cases.

## Troubleshooting & warnings

- Dtype cast warnings: `PyCausetDTypeWarning` when casting RHS arrays to a narrower/different dtype.
- Overflow risk warnings: `PyCausetOverflowRiskWarning` for float→int or narrowing casts in assignment; matmul may warn on int32 accumulation risk.
- Kernel guardrails: views with storage offsets are rejected by some kernels (`matmul`, `qr`, `lu`, `inverse`); materialize with `copy()` first.

## See also

- [[docs/index|API Reference]]
- [[guides/Numpy Integration.md|NumPy Integration]]
- [[guides/Matrix Guide.md|Matrix Guide]]
- [[guides/Storage and Memory.md|Storage and Memory]]
- [[project/protocols/NumPy Alignment Protocol.md|NumPy Alignment Protocol]]
- [[internals/Compute Architecture.md|Compute Architecture]]
