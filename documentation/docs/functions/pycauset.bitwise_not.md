# pycauset.bitwise_not

```python
pycauset.bitwise_not(a)    # ~a
```

Element-wise NOT (NumPy's `invert`): logical not for dtype `bool`, two's-complement not for
integer dtypes (`~5 == -6` in `int8`). Works for every structure and view; the result is
dense (the complement of a triangular matrix is not triangular). Float and complex operands
raise `TypeError`.

## Example

```python
import pycauset as pc

C = pc.causal_matrix(4)
not_c = ~C          # dense bool: True below the diagonal and on it
```
