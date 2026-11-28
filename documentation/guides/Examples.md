# Examples

Short, copy-paste recipes. Every block on this page runs as written, top to bottom
(`tests/engine/test_doc_snippets.py`).

```python
import pycauset as pc
import numpy as np
```

## Build matrices and vectors

```python
A = pc.matrix([[4.0, 1.0], [1.0, 3.0]])          # from nested lists: float64
B = pc.matrix(np.arange(6).reshape(2, 3))        # from NumPy: keeps int64
Z = pc.zeros((3, 3), dtype="int16")              # allocate by shape
v = pc.vector([1.0, 2.0])                        # vectors are 1-D, like NumPy's
I = pc.identity(4)                               # O(1) storage
```

## Causal matrices

```python
order = np.triu(np.random.default_rng(1).random((200, 200)) < 0.05, 1)
C = pc.causal_matrix(order)            # TriangularMatrix, dtype bool, 1 bit per entry
R = pc.causal_matrix(200, populate=True, seed=3)   # random, p = 0.5
two_step = C @ C                       # path counts (int32), on the packed bits
relations = C + C.T                    # comparability matrix, a lazy view
dense = np.asarray(C)                  # bool array, C[i, j] is True when i precedes j
```

## Structured matrices

```python
S = pc.symmetric(np.array([[2.0, 1.0], [1.0, 3.0]]))   # stores one triangle
D = pc.diagonal([1.0, 2.0, 3.0])                       # stores n values
T = pc.TriangularMatrix(3, dtype="float64", has_diagonal=True)
T[0, 1] = 5.0                                          # writes below the diagonal raise
type(T + T).__name__                                   # 'TriangularMatrix'
```

## Linear algebra

```python
x = pc.solve(A, v)                     # solve A x = v
P, L, U = pc.lu(A)                     # LU decomposition
Lc = pc.cholesky(A)                    # lower-triangular Cholesky factor
w, V = pc.eigh(A)                      # symmetric eigenvalues / eigenvectors
U_, s, Vt = pc.svd(A)                  # singular value decomposition
pc.inv(A), pc.cond(A), pc.slogdet(A), pc.norm(A, "fro")
```

Results come back as PyCauset objects in NumPy's dtypes; `np.asarray(...)` or
`pc.to_numpy(...)` converts them. See [[guides/Linear Algebra Operations|Linear Algebra
Operations]] for every operation.

## NumPy interop

```python
m = pc.matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
m + np.ones((2, 2))                    # NumPy arrays are operands too
np.ones(2) @ m                         # either side
back = np.asarray(m)                   # one copy into NumPy
```

## Save and load

```python
pc.save(A, "a.pycauset")
again = pc.load("a.pycauset")          # memory-mapped
pc.save_npy(A, "a.npy")                # NumPy's format
```

## Configuration

```python
pc.set_memory_threshold(100 * 1024 * 1024)  # matrices above 100 MB are file-backed
pc.set_num_threads(4)                       # cap parallel work
```

See [[guides/Advanced Usage|Advanced Usage]] for the full set of knobs.
