# pycauset.solve_triangular

Solve a linear system $A x = b$ when $A$ is claimed to be diagonal or triangular via A.properties.

## Signature

```python
pycauset.solve_triangular(A, b)
```

## Behavior (R1_PROPERTIES)

- Uses gospel properties, not truth validation. If A.properties["is_diagonal"] or is_upper_triangular / is_lower_triangular is set, the solver treats off-triangle entries as zero.
- Diagonal path: elementwise divide.
- Triangular path: converts to the native TriangularFloatMatrix and solves via the native inverse then matmul.
- Shape must be square; raises ValueError otherwise.
- If no triangular/diagonal claim is present, falls back to pycauset.solve or raises.

## Examples

```python
A = pycauset.identity(3)
A.properties["is_upper_triangular"] = True  # gospel assertion
b = pycauset.vector([1, 2, 3])
x = pycauset.solve_triangular(A, b)
```

## Notes

- Properties are authoritative; the solver does not scan payloads to verify structure.
- Cached-derived values and other properties are maintained via the properties health-check system.