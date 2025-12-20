# pycauset.precision_mode

A context manager that temporarily overrides the current thread-local promotion precision mode.

## Syntax

```python
with pycauset.precision_mode(mode):
    ...
```

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `mode` | `str` | Either `"lowest"` or `"highest"`. |

## Description

This is the recommended way to override promotion policy for a specific block of code,
including expressions that use Python operators like `a + b` or `a @ b`.

On exit, the previous precision mode is restored.

Notes:

- This controls **storage dtype selection** (the dtype of the resulting matrix/vector).
- It does not directly control accelerator internal compute dtype.

## Example

```python
import pycauset as pc

a = pc.Float32Matrix(2)
b = pc.FloatMatrix(2)  # float64

with pc.precision_mode("highest"):
    c = a @ b
```

## See also

- [[docs/functions/pycauset.set_precision_mode.md|pycauset.set_precision_mode]]
- [[docs/functions/pycauset.get_precision_mode.md|pycauset.get_precision_mode]]
- [[docs/functions/pycauset.divide.md|pycauset.divide]]
