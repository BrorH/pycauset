# pycauset.matrix_power

```python
pycauset.matrix_power(a, n)
```

Returns the integer power `A^n` of a square matrix, via binary exponentiation.

## Parameters

*   **a** (matrix): A square matrix.
*   **n** (int): The integer exponent. Negative exponents use the inverse.

## Returns

A matrix equal to `A^n`.

## Notes

*   `n == 0` returns an identity matrix of `a`'s shape and dtype.
*   `n == 1` returns `a` itself, as NumPy does.
*   `n < 0` inverts first; a singular `a` raises `numpy.linalg.LinAlgError` (a `ValueError`),
    as NumPy.
*   There are no value shortcuts: an identity or zero matrix is powered by matmul like any
    other (identity x identity is already O(1)).

## Examples

```python
import pycauset as pc

A = pc.matrix([ [1.0, 2.0], [3.0, 4.0] ])
pc.matrix_power(A, 2)   # A @ A
```

## See also

* [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
* [[docs/functions/pycauset.inv.md|pycauset.inv]]
