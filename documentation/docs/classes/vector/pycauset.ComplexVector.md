# pycauset.ComplexVector

A vector of complex numbers, stored as two separate memory-mapped files (real and imaginary parts).

## Constructor

```python
pycauset.ComplexVector(n: int, backing_file_real: str = "", backing_file_imag: str = "")
```

*   `n`: Size of the vector.
*   `backing_file_real`: Path to store the real part.
*   `backing_file_imag`: Path to store the imaginary part.

## Properties

### `real`
Returns the `FloatVector` representing the real part.

### `imag`
Returns the `FloatVector` representing the imaginary part.

### `shape`
Returns `(n,)`.

## Methods

### `get(i: int) -> complex`
Returns the complex value at index `i`.

### `set(i: int, value: complex)`
Sets the value at index `i`.

### `conjugate() -> ComplexVector`
Returns the complex conjugate of the vector.

### `dot(other: ComplexVector) -> complex`
Computes the dot product with another vector.

### `cross(other: ComplexVector) -> ComplexVector`
Computes the cross product (only for 3D vectors).

## Operators

*   `+` (Addition): Element-wise addition with another `ComplexVector` or scalar.
*   `-` (Subtraction): Element-wise subtraction.
*   `*` (Multiplication): Element-wise multiplication by a scalar.
