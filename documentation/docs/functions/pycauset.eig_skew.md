# pycauset.eig_skew

~~~python
pycauset.eig_skew(a, k)
~~~

Compute the top-$k$ eigenvalues (by magnitude) and eigenvectors of a real
skew-symmetric matrix.

A real skew-symmetric matrix satisfies $A = -A^\top$. Its eigenvalues are purely
imaginary and come in $\pm i\lambda$ pairs, with one zero eigenvalue when the
dimension is odd. This is the diagonalization used to build the Sorkin-Johnston
vacuum $W$ from the Pauli-Jordan matrix $i\Delta$.

## Parameters

* **a**: Input square matrix (real, skew-symmetric).
* **k**: Number of eigenpairs to return.

## Returns

A tuple `(w, v)`:

* **w**: Complex eigenvalues sorted by descending magnitude, of length `min(k, n)`.
* **v**: Complex `n x min(k, n)` matrix whose columns are the matching right
  eigenvectors, in the same order. Each column satisfies $A v_j = w_j v_j$.

## Notes

* Uses the native backend when available, otherwise falls back to NumPy's general
  eigensolver.
* Requesting `k > n` returns at most `n` eigenpairs.
* The native path uses the general LAPACK `dgeev` solver. A dedicated skew
  tridiagonalization is planned but not yet implemented.

## Example

~~~python
import numpy as np
import pycauset as pc

rng = np.random.default_rng(0)
M = rng.random((8, 8))
A = M - M.T          # skew-symmetric

w, v = pc.eig_skew(pc.matrix(A), 3)
print(w.get(0))      # e.g. 0+1.23j
~~~
