# pycauset.vector

```python
pycauset.vector(source, dtype=None, *, storage=None, max_in_ram_bytes=None)
```

Create a 1D vector from vector-like input.

This is a data constructor (aligned with 1D `np.array([...])`). 

## Parameters

*   **source** (*sequence or numpy.ndarray*): 1D data (e.g. list) or a 1D NumPy array.
*   **dtype** (*str or type, optional*): Coerce storage dtype (e.g. `"float64"`, `"int32"`, `float`, `int`). Without it, a NumPy array keeps its dtype, and Python integers become **int32** (NumPy: int64), as do `int` and `"int"`; see [[internals/DType System|DType System]] §3.2.
*   **storage** (*"ram" or "disk", optional*): `"disk"` places the vector in a file whatever its size; any other value than `"ram"` / `"disk"` raises `ValueError`.
*   **max_in_ram_bytes** (*int or None, optional*): a vector larger than this goes to a file.
*   Other keywords raise `TypeError`, as does a dtype PyCauset does not store. A vector is built by the same rules as `matrix` (one import path).

## Returns

*   **VectorBase**: An instance of a concrete vector class (see [[pycauset.VectorBase]] and [[docs/classes/vector/index.md|vector classes]]).

## Examples

```python
import pycauset

v = pycauset.vector([1, 2, 3])
v_c = pycauset.vector([1+2j, 3-4j], dtype="complex_float32")
```
