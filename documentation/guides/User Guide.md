# User Guide

A guided tour of the engine: how PyCauset thinks about a matrix, and what that buys you.
Read it once and the rest of the guides fall into place. Every block runs as written
(`tests/engine/test_doc_snippets.py`).

## The mental model

A PyCauset matrix is NumPy's matrix (same values, same dtypes, same operator semantics)
with four things NumPy does not have:

1. **A structure class.** `DenseMatrix`, `TriangularMatrix`, `SymmetricMatrix`,
   `AntiSymmetricMatrix`, `HermitianMatrix`, `SkewHermitianMatrix`, `DiagonalMatrix`,
   `IdentityMatrix`. The class says what is stored and which algorithm applies. A causal
   matrix is a `TriangularMatrix` of dtype `bool`, stored one bit per element.
2. **Views.** Transposition, conjugation, scaling and windows are O(1) views over shared
   storage; operations read them directly instead of copying.
3. **Lazy results where they pay.** `C + C.T` of a causal matrix is a view over `C`'s bits
   until something needs its values ([[guides/Lazy Evaluation|Lazy Evaluation]]).
4. **Storage that scales.** Small matrices live in RAM; matrices above a threshold live in
   memory-mapped files, and operations that do not fit run tile by tile.

NumPy is the reference: every operation is checked against it for every dtype, structure
and view (the operation atlas). Where PyCauset deliberately differs, the
[[internals/DType System|DType System]] page says so.

## Make matrices

```python
import pycauset as pc
import numpy as np

A = pc.matrix([[4.0, 1.0], [1.0, 3.0]])          # from data
B = pc.matrix(np.arange(6.0).reshape(2, 3))      # from NumPy (dtype kept)
Z = pc.zeros((1000, 1000), dtype="float32")      # by shape
v = pc.vector([1.0, 2.0])                        # 1-D, like NumPy
A.shape, A.dtype                                 # ((2, 2), 'float64')
```

## Causal matrices

```python
order = np.triu(np.ones((6, 6), dtype=bool), 1)     # a 6-element chain
C = pc.causal_matrix(order)
C.dtype, type(C).__name__                            # ('bool', 'TriangularMatrix')

paths = C @ C          # paths[i, j]: number of 2-step chains from i to j (int32)
R = C + C.T            # comparability matrix, lazy
K = pc.compute_k(C, 2.0)   # C (2 I + C)^-1, a TriangularMatrix
```

Writing below the diagonal of a triangular matrix raises `ValueError`: the class only
stores the upper part.

## Structure is kept when the result has it

```python
D = pc.diagonal([1.0, 2.0, 3.0])
type(D @ D).__name__                                 # 'DiagonalMatrix'
S = pc.symmetric(np.array([[2.0, 1.0], [1.0, 2.0]]))
type(S + S).__name__                                 # 'SymmetricMatrix'
T = pc.TriangularMatrix(3, dtype="float64", has_diagonal=True)
type(T * pc.matrix(np.ones((3, 3)))).__name__        # 'TriangularMatrix'
```

Values and dtypes are NumPy's either way; only the storage and the work shrink.

## Views

```python
At = A.T                 # shares A's storage
Ac = (A * 2j).conj()     # scaled and conjugated, still no copy
W = A[0:1, :]            # a window
At @ A                   # kernels read the views directly
```

Writing into a view folds the view into storage first, so the write means what it says.

## Linear algebra

```python
x = pc.solve(A, v)                  # like np.linalg.solve
w, V = pc.eigh(A)                   # symmetric eigen decomposition
U, s, Vt = pc.svd(A)
pc.determinant(A), pc.matrix_rank(A), pc.norm(A, 2)
```

Factorisations and eigen problems run LAPACK in NumPy's precision (single stays single).
The full list: [[guides/Linear Algebra Operations|Linear Algebra Operations]].

## NumPy in and out

```python
np.asarray(A)            # one copy into NumPy
A + np.eye(2)            # NumPy arrays are operands
np.ones(2) @ A           # on either side
```

No PyCauset operation calls NumPy to compute its result. See
[[guides/Numpy Integration|NumPy Integration]] for the export guard on large and
file-backed matrices.

## Scale up

```python
pc.set_memory_threshold(100 * 1024 * 1024)  # matrices above 100 MB are file-backed
pc.set_num_threads(4)                       # cap parallel work
pc.save(A, "a.pycauset")                    # one file; load memory-maps it
```

See [[guides/Storage and Memory|Storage and Memory]],
[[guides/Performance Guide|Performance]] and [[guides/Advanced Usage|Advanced Usage]].

## Where next

- [[guides/Examples|Examples]]: copy-paste recipes.
- [[guides/Matrix Guide|Matrix Guide]] and [[guides/Vector Guide|Vector Guide]].
- [[docs/index|API Reference]]: exact signatures.
