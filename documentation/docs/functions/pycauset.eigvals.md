# pycauset.eigvals

```python
pycauset.eigvals(a)
```

Eigenvalues of a general (non-symmetric) square matrix.

## Returns

A complex vector of eigenvalues.

## Backend

- Prefers the native backend, which routes through the AutoSolver cost model:
  the operation runs on the GPU when the GPU is active **and** the model predicts
  it wins, otherwise on the CPU.
- On CUDA builds the GPU path uses cuSOLVER `geev`.
- Falls back to NumPy (`numpy.linalg.eigvals`) if the native backend cannot
  handle the input.

## Notes

- For symmetric/Hermitian inputs prefer `pycauset.eigvalsh` / `pycauset.eigh`.
