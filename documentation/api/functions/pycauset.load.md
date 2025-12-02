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

## See Also

*   [[pycauset.CausalSet.load]]: For loading `CausalSet` objects from `.causet` archives.

