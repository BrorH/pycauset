# pycauset.bitwise_nor

```python
pycauset.bitwise_nor(a, b)
```

Elementwise bitwise NOR (NOT OR) of two bit matrices or bit vectors.

Equivalent to `bitwise_not(bitwise_or(a, b))`.

## Parameters

*   **a** (matrix or vector): First operand.
*   **b** (matrix or vector): Second operand.

## Returns

A bit matrix or bit vector.

## See also

* [[docs/functions/pycauset.bitwise_or.md|pycauset.bitwise_or]]
* [[docs/functions/pycauset.bitwise_nand.md|pycauset.bitwise_nand]]
