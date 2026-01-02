# pycauset.vector

```python
pycauset.vector(source, dtype=None, *, max_in_ram_bytes=None, **kwargs)
```

Create a 1D vector from vector-like input.

This is a data constructor (aligned with 1D `np.array([...])`). 

## Parameters

*   **source** (*sequence or numpy.ndarray*): 1D data (e.g. list) or a 1D NumPy array.
*   **dtype** (*str or type, optional*): Coerce storage dtype (e.g. `"float64"`, `"int32"`, `float`, `int`).
*   **max_in_ram_bytes** (*int or None, optional*): When constructing from a NumPy array, route through the internal `native.asarray` import path if the estimated materialized size exceeds this cap to avoid overcommitting RAM (falls back to native regardless when supported dtypes are used).
*   **kwargs**: Passed through to the backend constructor.

## Returns

*   **VectorBase**: An instance of a concrete vector class (see [[pycauset.VectorBase]] and [[docs/classes/vector/index.md|vector classes]]).

## Examples

```python
import pycauset

v = pycauset.vector([1, 2, 3])
v_c = pycauset.vector([1+2j, 3-4j], dtype="complex_float32")
```
