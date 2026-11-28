# pycauset.eigvals_arnoldi

~~~python
pycauset.eigvals_arnoldi(a, k, m, tol=1e-6)
~~~

The `k` eigenvalues of largest magnitude, computed with a restarted Krylov method
(implicitly restarted Lanczos for symmetric input, implicitly restarted Arnoldi otherwise).

## Parameters

* **a**: Square real matrix of any structure or dtype, in RAM or disk-backed. It is only
  accessed through matrix-vector products, so it is never copied or materialized.
* **k**: Number of eigenvalues to return (`1 <= k <= n`).
* **m**: Krylov subspace size. Raised to at least `2k + 1` (capped at `n`); larger values
  converge in fewer restarts at the cost of `m` vectors of length `n` in RAM.
* **tol**: Relative accuracy of each returned eigenvalue.

## Returns

* **DenseVector (float64)** of length `k` if `a` is symmetric, otherwise **DenseVector (complex_float64)**.
  Sorted by descending magnitude (ties by real part, then imaginary part).

## Raises

* `RuntimeError` if the iteration does not converge to `tol`. Increase `m` (or `tol`).
  The function never returns unconverged values.
* `ValueError` for a non-square matrix or `k` / `m` below 1; `TypeError` for a complex
  matrix (real dtypes only).

## Notes

* Results are deterministic: the starting vector is generated from a fixed seed.
* Symmetry is detected exactly (structure, then an element-by-element check).
* Implementation: [Spectra](https://spectralib.org) (MPL-2.0), driving PyCauset's own
  matrix-vector kernels. Problems too small for a Krylov method use a dense eigensolver.
* For real skew-symmetric matrices (e.g. Pauli–Jordan), `pycauset.eigvals_skew` exploits
  the ±iλ pairing.

## Example

```python
import pycauset as pc
import numpy as np

rng = np.random.default_rng(0)
a = rng.standard_normal((300, 300))
s = a + a.T
w = pc.to_numpy(pc.eigvals_arnoldi(pc.matrix(s), k=5, m=40, tol=1e-10))

ref = np.linalg.eigvalsh(s)
ref = ref[np.argsort(-np.abs(ref))][:5]          # the 5 largest in magnitude
assert np.allclose(w, ref)
```
