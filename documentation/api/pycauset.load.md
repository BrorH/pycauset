# pycauset.load

```python
pycauset.load(path)
```

Loads a matrix from a binary file created by PyCauset.

## Parameters

*   **path** (*str*): The path to the file to load.

## Returns

*   **Matrix**: An instance of `CausalMatrix`, `IntegerMatrix`, `TriangularFloatMatrix`, or `FloatMatrix`, depending on the file content.

## Description

This function reads the 4096-byte header of the specified file to determine the matrix type and data format. It then memory-maps the file and returns the appropriate matrix object.

The returned object is backed by the file on disk. Changes to the matrix (if mutable) are written directly to the file.

## Example

```python
import pycauset

# Load a matrix
matrix = pycauset.load("my_matrix.pycauset")

if isinstance(matrix, pycauset.CausalMatrix):
    print("Loaded a CausalMatrix")
    print(matrix.get(0, 1))
elif isinstance(matrix, pycauset.IntegerMatrix):
    print("Loaded an IntegerMatrix")
```

## See Also

*   [[File Format]]
*   [[api/pycauset.save|pycauset.save]]
