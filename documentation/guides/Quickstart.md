# Quickstart

PyCauset is NumPy for causal sets: matrices and vectors with NumPy's values and dtypes, plus
storage that knows its structure (bit-packed causal matrices, triangular, symmetric, diagonal)
and can live on disk. Ten minutes from `pip install` to linear algebra on a causal matrix.

## Install

```bash
pip install pycauset
```

Import PyCauset before NumPy: the two use separate BLAS thread pools, and importing NumPy
first slows PyCauset's BLAS/LAPACK calls (PyCauset warns when it happens).

## 1. Matrices and vectors

```python
import pycauset as pc
import numpy as np

A = pc.matrix(np.array([[4.0, 1.0], [2.0, 3.0]]))
v = pc.vector([1.0, 2.0])

A @ v              # matrix-vector product (a vector)
A.T @ A            # transposition is a view: nothing is copied
2 * A + 1          # element-wise, NumPy's broadcasting and dtypes
np.asarray(A)      # back to NumPy (one copy)
```

Results follow NumPy: `pc.matrix(np.ones((2, 2), np.float32)) + 1` is float32. The documented
differences (Python integers default to int32; mixed float precisions under-promote) are in
[[internals/DType System|DType System]].

## 2. A causal matrix

A causal matrix `C` has `C[i, j] = True` when element `i` precedes `j`. It is strictly upper
triangular and stored one bit per element.

```python
C = pc.causal_matrix(np.triu(np.ones((5, 5), dtype=bool), 1))  # a 5-element chain
type(C).__name__, C.dtype      # ('TriangularMatrix', 'bool')

paths = C @ C                  # number of 2-step paths between each pair (int32)
R = C + C.T                    # the comparability matrix: a lazy view over C's bits
```

`C @ C` counts on the packed bits; `C + C.T` costs no memory until it is read.

## 3. Structured matrices

```python
S = pc.symmetric(np.array([[2.0, 1.0], [1.0, 2.0]]))   # stores one triangle
w, V = pc.eigh(S)                                     # LAPACK, like np.linalg.eigh
D = pc.diagonal([1.0, 2.0, 3.0])                      # stores n values
pc.determinant(D)                                     # O(n) for a diagonal
```

Every operation keeps the structure when the result has it (diagonal @ diagonal is diagonal;
triangular + triangular is triangular) and returns NumPy's values.

## 4. Linear algebra

```python
pc.solve(A, v)          # like np.linalg.solve
pc.invert(A)            # also A.inverse()
pc.svd(A)               # U, S, Vt with NumPy's shapes
pc.norm(A, "fro"), pc.cond(A), pc.matrix_rank(A)
```

The full list is [[Linear Algebra Operations]].

## 5. Big matrices and saving

```python
pc.set_memory_threshold(256 * 1024**2)   # matrices above 256 MB are backed by files
pc.save(A, "a.pycauset")
B = pc.load("a.pycauset")                # memory-mapped: O(1) to open
```

Operations that do not fit in RAM run tile by tile; see [[Storage and Memory]].

## Where next

- [[Matrix Guide]] and [[Vector Guide]]: construction, indexing, views, structures.
- [[Linear Algebra Operations]]: every operation, with NumPy's semantics.
- [[Numpy Integration]]: converting, mixing, and what never goes through NumPy.
- [[Storage and Memory]] and [[Saving and Loading Data]]: disk-backed matrices and files.
- [[Lazy Evaluation]]: which results are views and when they materialize.
