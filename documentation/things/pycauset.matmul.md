```
pycauset.matmul(a, b, saveas=None)
```
Perform matrix multiplication.

If both inputs are [[pycauset.CausalMatrix]] instances, an optimized C++ implementation is used, returning an [[pycauset.IntegerMatrix]].
Otherwise, a generic multiplication is performed, returning a [[pycauset.Matrix]].

### Parameters:
- a: [[pycauset.Matrix]]. Left operand.
- b: [[pycauset.Matrix]]. Right operand.
- saveas: str (_optional_). Path to save the resulting matrix. Only used if the optimized C++ path is taken.

### Returns:
[[pycauset.IntegerMatrix]] or [[pycauset.Matrix]] instance
