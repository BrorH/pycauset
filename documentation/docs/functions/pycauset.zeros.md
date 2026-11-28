# pycauset.zeros

```python
pycauset.zeros(shape, *, dtype="float64", storage=None, max_in_ram_bytes=None)
```

A new vector (`shape` an int) or matrix (`shape` a `(rows, cols)` pair) filled with zeros.

## Parameters

* **shape** (*int or tuple*): `n` for a `DenseVector`, `(rows, cols)` for a `DenseMatrix`.
* **dtype** (*str or type*): element type, `"float64"` by default as in NumPy. Any pycauset
  dtype: `"bool"`, `"int8"`…`"uint64"`, `"float16"`…`"float64"`, `"complex_float16"`…
  `"complex_float64"`; Python `int`, `float`, `bool` and NumPy dtypes are accepted too.
* **storage** (*"ram" or "disk", optional*): `"disk"` places the object in a file whatever its
  size; `"ram"` (the default) keeps it in RAM unless it is above the memory threshold or does
  not fit. Any other value raises `ValueError`.
* **max_in_ram_bytes** (*int, optional*): an object larger than this goes to a file.
* Other keywords raise `TypeError`, as does a dtype PyCauset does not store (`"junk"`,
  `np.longdouble`).

## Returns

* **DenseVector** or **DenseMatrix** of the given dtype.

## Example

```python
import pycauset as pc

m = pc.zeros((3, 4))
assert m.dtype == "float64" and m[2, 3] == 0.0
assert pc.zeros(5, dtype="int16").dtype == "int16"
```
