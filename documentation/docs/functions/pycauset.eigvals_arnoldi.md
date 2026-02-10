# pycauset.eigvals_arnoldi

~~~python
pycauset.eigvals_arnoldi(a, k, m, tol=1e-6)
~~~

Compute the top-$k$ eigenvalues (by magnitude) using an Arnoldi/Lanczos-style iteration.

## Parameters

* **a**: Input square matrix.
* **k**: Number of eigenvalues to return.
* **m**: Arnoldi subspace dimension.
* **tol**: Convergence tolerance used by the fallback path.

## Returns

* **VectorBase**: Eigenvalues sorted by descending magnitude.

## Notes

Uses the native backend when available, otherwise falls back to NumPy.
Only real eigenvalues are supported in the current implementation; complex eigenvalues
raise `NotImplementedError`.
