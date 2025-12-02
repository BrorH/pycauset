# pycauset.ComplexMatrix

A matrix storing complex numbers, backed by two separate memory-mapped files for real and imaginary parts.

## Constructor

```python
pycauset.ComplexMatrix(n: int, backing_file_real: str = "", backing_file_imag: str = "")
```

## Properties

### `real`
Returns the `FloatMatrix` representing the real part.

### `imag`
Returns the `FloatMatrix` representing the imaginary part.

### `T`
Returns the transpose of the matrix.

### `H`
Returns the Hermitian conjugate (conjugate transpose) of the matrix.

## Methods

### `get(i: int, j: int) -> complex`
Get the complex value at row `i` and column `j`.

### `set(i: int, j: int, value: complex)`
Set the complex value at row `i` and column `j`.

### `size() -> int`
Get the dimension of the matrix.

### `close()`
Release the memory-mapped backing files.

### `conjugate() -> ComplexMatrix`
Returns the complex conjugate of the matrix.

### `__add__(other: ComplexMatrix) -> ComplexMatrix`
Add two complex matrices.

### `__mul__(other: ComplexMatrix) -> ComplexMatrix`
Multiply two complex matrices.

### `__repr__() -> str`
String representation of the matrix.
