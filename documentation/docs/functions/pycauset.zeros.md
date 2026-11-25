# pycauset.zeros

```python
pycauset.zeros(shape, *, dtype="float64", **kwargs)
```

A new vector (`shape` an int) or matrix (`shape` a `(rows, cols)` pair) filled with zeros.

## Parameters

* **shape** (*int or tuple*): `n` for a `DenseVector`, `(rows, cols)` for a `DenseMatrix`.
* **dtype** (*str or type*): element type, `"float64"` by default as in NumPy. Any pycauset
  dtype: `"bool"`, `"int8"`…`"uint64"`, `"float16"`…`"float64"`, `"complex_float16"`…
  `"complex_float64"`; Python `int`, `float`, `bool` and NumPy dtypes are accepted too.

## Returns

* **DenseVector** or **DenseMatrix** of the given dtype.

## Example

```python
import pycauset as pc

m = pc.zeros((3, 4))
assert m.dtype == "float64" and m[2, 3] == 0.0
assert pc.zeros(5, dtype="int16").dtype == "int16"
```
