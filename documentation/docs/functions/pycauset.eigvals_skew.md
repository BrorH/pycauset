# pycauset.eigvals_skew

~~~python
pycauset.eigvals_skew(a, k)
~~~

Compute the top-$k$ eigenvalues (by magnitude) of a real skew-symmetric matrix.

A real skew-symmetric matrix satisfies $A = -A^\top$. Its eigenvalues are purely
imaginary and come in $\pm i\lambda$ pairs, with one zero eigenvalue when the
dimension is odd.

## Parameters

* **a**: Input square matrix (real, skew-symmetric).
* **k**: Number of eigenvalues to return.

## Returns

* **VectorBase**: Complex eigenvalues sorted by descending magnitude, of length
  `min(k, n)`. Each entry has a negligible real part (machine precision) and an
  imaginary part of the form $\pm\lambda$.

## Notes

* Uses the native backend when available, otherwise falls back to NumPy's general
  eigensolver.
* Requesting `k > n` returns at most `n` eigenvalues.
* A dedicated Paige–Van Loan tridiagonalization (the asymptotically faster
  $O(n^2)$-ish algorithm for skew matrices) is planned but not yet implemented;
  the current native path uses the general LAPACK `dgeev` solver.

## Example

~~~python
import numpy as np
import pycauset as pc

rng = np.random.default_rng(0)
M = rng.random((8, 8))
A = M - M.T          # skew-symmetric

evals = pc.eigvals_skew(pc.matrix(A), 3)   # top-3 imaginary eigenvalues
for i in range(evals.size()):
    print(evals.get(i))                     # e.g. 0+1.23j, 0-1.23j, 0+0.91j
~~~
