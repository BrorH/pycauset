# pycauset.CausalMatrix

```python
pycauset.CausalMatrix(n, populate=True, **kwargs)
```

Factory function to create a `TriangularBitMatrix`. By default, it populates the matrix with random bits ($p=0.5$), simulating a random causal set.

## Parameters

*   **n** (*int*): The dimension of the causal set ($N \times N$).
*   **populate** (*bool*): If `True` (default), fills the matrix with random bits. If `False`, returns an empty (all-zeros) matrix.
*   **kwargs**: Additional arguments passed to the `TriangularBitMatrix` constructor.

## Returns

*   **TriangularBitMatrix**: A new instance of [[pycauset.TriangularBitMatrix]].

## Example

```python
# Create a random causal matrix (p=0.5)
C = pycauset.CausalMatrix(1000)

# Create an empty causal matrix
C_empty = pycauset.CausalMatrix(1000, populate=False)
```

