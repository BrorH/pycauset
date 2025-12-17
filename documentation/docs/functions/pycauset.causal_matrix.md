# pycauset.causal_matrix

```python
pycauset.causal_matrix(n, populate=True, **kwargs)
```

Factory function for creating a [[pycauset.TriangularBitMatrix]] suitable for representing a causal relation.

## Parameters

*   **n** (*int*): The dimension of the causal set (`n×n`).
*   **populate** (*bool*): If `True` (default), fills with random bits (`p=0.5`). If `False`, returns an all-zeros matrix.
*   **kwargs**: Passed through to the backend constructor.

## Returns

*   **TriangularBitMatrix**: A new instance of [[pycauset.TriangularBitMatrix]].

## Example

```python
import pycauset

c = pycauset.causal_matrix(1000)
c_empty = pycauset.causal_matrix(1000, populate=False)
```
