# pycauset.get_num_threads

Gets the current number of threads used for parallel operations.

## Syntax

```python
n = pycauset.get_num_threads()
```

## Returns

| Type | Description |
| :--- | :--- |
| `int` | The current number of threads. |

## Description

Returns the number of threads currently configured for the global thread pool. If `set_num_threads` has not been called, this returns the default value (usually the number of hardware threads).

## Example

```python
import pycauset

n = pycauset.get_num_threads()
print(f"PyCauset is using {n} threads.")
```
