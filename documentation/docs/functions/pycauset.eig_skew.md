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

## Raises

* `TypeError` for a complex matrix (real dtypes only: bool, integer, float).
* `ValueError` if `a` is not square, `k < 1`, or `a` is not exactly skew-symmetric (every
  `A[i, j] == -A[j, i]` and a zero diagonal, checked element by element; the message names the
  first pair that differs).

## Notes

* The eigenpairs come from LAPACK's Hermitian solver (`zheevd`) on `i A`, which is Hermitian
  for a real skew-symmetric `A`: the eigenvalues are exactly imaginary and come in `±` pairs,
  and the eigenvectors are orthonormal.
* Requesting `k > n` returns at most `n` eigenpairs; a 0 x 0 matrix gives empty results.

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
