# pycauset.vector

```python
pycauset.vector(source, dtype=None, **kwargs)
```

Create a 1D vector from vector-like input.

This is a data constructor (aligned with 1D `np.array([...])`). 

## Parameters

*   **source** (*sequence or numpy.ndarray*): 1D data (e.g. list) or a 1D NumPy array.
*   **dtype** (*str or type, optional*): Coerce storage dtype (e.g. `"float64"`, `"int32"`, `float`, `int`).
*   **kwargs**: Passed through to the backend constructor.

## Returns

*   **VectorBase**: An instance of a concrete vector class (see [[pycauset.VectorBase]] and [[docs/classes/vector/index.md|vector classes]]).

## Examples

```python
import pycauset

v = pycauset.vector([1, 2, 3])
v_c = pycauset.vector([1+2j, 3-4j], dtype="complex_float32")
```
