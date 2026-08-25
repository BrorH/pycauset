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

Structural shortcuts:

*   `n == 0` returns an identity matrix.
*   `n == 1` returns `a` unchanged.
*   `is_identity` returns `a` for any `n`.
*   `is_zero` returns `a` for `n > 0`, and raises `ValueError` for `n < 0`.

## Examples

```python
import pycauset as pc

A = pc.matrix([[1.0, 2.0], [3.0, 4.0]])
pc.matrix_power(A, 2)   # A @ A
```

## See also

* [[docs/functions/pycauset.matmul.md|pycauset.matmul]]
* [[docs/functions/pycauset.invert.md|pycauset.invert]]
