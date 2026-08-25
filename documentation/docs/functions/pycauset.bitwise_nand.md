# pycauset.bitwise_nand

```python
pycauset.bitwise_nand(a, b)
```

Elementwise bitwise NAND (NOT AND) of two bit matrices or bit vectors.

Equivalent to `bitwise_not(bitwise_and(a, b))`.

## Parameters

*   **a** (matrix or vector): First operand.
*   **b** (matrix or vector): Second operand.

## Returns

A bit matrix or bit vector.

## See also

* [[docs/functions/pycauset.bitwise_and.md|pycauset.bitwise_and]]
* [[docs/functions/pycauset.bitwise_nor.md|pycauset.bitwise_nor]]
