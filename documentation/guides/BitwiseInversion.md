# Bitwise Inversion

PyCauset supports bitwise inversion (NOT-operation) for `TriangularBitMatrix`, `IntegerMatrix`, and `TriangularFloatMatrix`.

## Usage

You can use the Python bitwise NOT operator `~` or the `pycauset.bitwise_not()` function.

```python
import pycauset

# Bit Matrix
m = pycauset.TriangularBitMatrix(5)
m[0, 1] = 1
inv = ~m
# OR
inv = pycauset.bitwise_not(m)

# inv[0, 1] will be 0
# inv[0, 2] will be 1 (if m[0, 2] was 0)

# Integer Matrix
im = pycauset.IntegerMatrix(5)
im[0, 1] = 5
inv_im = ~im
# inv_im[0, 1] will be ~5 (bitwise NOT of the stored integer)

# Float Matrix
fm = pycauset.TriangularFloatMatrix(5)
fm[0, 1] = 0.5
inv_fm = ~fm
# inv_fm[0, 1] will be the double value corresponding to the bitwise NOT of the IEEE 754 representation of 0.5
```

## Implementation Details

- **TriangularBitMatrix**: Inverts all bits in the upper triangle. Padding bits in the last word of each row are masked to 0 to maintain consistency.
- **IntegerMatrix**: Performs bitwise NOT on the stored 32-bit integers.
- **TriangularFloatMatrix**: Performs bitwise NOT on the 64-bit floating point representation (IEEE 754). This is primarily useful for specific cryptographic or encoding applications and does not correspond to arithmetic inversion ($1/x$) or matrix inversion ($A^{-1}$).

## Performance

The inversion is performed efficiently using block operations on the memory-mapped data.
