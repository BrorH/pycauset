# pycauset.ComplexMatrix

This page is kept for historical/backwards-reference purposes.

PyCauset's complex support is now expressed as first-class dtypes and classes:

- Use `pycauset.Matrix(..., dtype="complex_float16")` -> [[pycauset.ComplexFloat16Matrix]]
- Use `pycauset.Matrix(..., dtype="complex_float32")` (aka `complex64`) -> [[pycauset.ComplexFloat32Matrix]]
- Use `pycauset.Matrix(..., dtype="complex_float64")` (aka `complex128`) -> [[pycauset.ComplexFloat64Matrix]]

For the authoritative dtype rules, see `documentation/internals/DType System.md`.
