# pycauset.set_num_threads

Sets the number of threads to use for parallel operations.

## Syntax

```python
pycauset.set_num_threads(n)
```

## Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `n` | `int` | The number of threads to use. Must be $\ge 1$. |

## Description

PyCauset uses a custom thread pool for parallelizing heavy operations like matrix multiplication and eigenvalue solving. By default, it uses the number of hardware threads available on the system.

Use this function to manually control the parallelism level. This is useful for benchmarking or when running in a shared environment where you want to limit resource usage.

## Example

```python
import pycauset
import os

# Use half the available cores
cores = os.cpu_count()
pycauset.set_num_threads(cores // 2)
```
