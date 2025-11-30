# pycauset.load

```python
pycauset.load(path: str) -> PersistentObject
```

Loads a matrix or vector from a binary file created by PyCauset.

## Parameters

*   **path** (*str*): The path to the file to load.

## Returns

*   **PersistentObject**: An instance of the appropriate class (e.g., `TriangularBitMatrix`, `IntegerMatrix`, `FloatVector`, etc.) depending on the file content.

## Description

This function reads the header of the specified file to determine the type and data format. It then memory-maps the file and returns the appropriate object.
