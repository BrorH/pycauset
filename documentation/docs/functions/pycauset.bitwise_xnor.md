# pycauset.bitwise_xnor

```python
pycauset.bitwise_xnor(a, b)
```

Elementwise bitwise XNOR (NOT XOR) of two bit matrices or bit vectors.

Equivalent to `bitwise_not(bitwise_xor(a, b))`.

## Parameters

*   **a** (matrix or vector): First operand.
*   **b** (matrix or vector): Second operand.

## Returns

A bit matrix or bit vector.

## See also

* [[docs/functions/pycauset.bitwise_xor.md|pycauset.bitwise_xor]]
* [[docs/functions/pycauset.bitwise_and.md|pycauset.bitwise_and]]
