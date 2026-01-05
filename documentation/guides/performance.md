# Performance Guide

PyCauset is designed for high-performance matrix operations, particularly for large-scale causal analysis. This guide highlights areas where PyCauset outperforms standard libraries like NumPy and explains the architectural decisions behind these performance gains.

## NumPy Interoperability

PyCauset achieves **3x to 8x faster** data transfer rates compared to standard NumPy file I/O for supported types.

| Operation | Speedup vs NumPy | Notes |
| :--- | :--- | :--- |
| **Float64 / Int32 Read/Write** | **3x - 8x** | Uses zero-copy memory mapping and direct `memcpy` paths. |
| **Complex128 Read/Write** | **3.5x - 4x** | Optimized C++ loops bypass Python iteration overhead. |
| **BitMatrix (Boolean) Write** | **5.0x** | SIMD-accelerated packing (SSE2) converts booleans to bits instantly. |
| **BitMatrix (Boolean) Read** | **1.9x** | SIMD-accelerated unpacking. |

### Why is it faster?
1.  **Zero-Copy Primitives**: For `float64` and `int32`, PyCauset exposes raw memory pointers directly to NumPy, allowing the OS to handle data transfer without CPU intervention.
2.  **SIMD Acceleration**: Boolean arrays are packed/unpacked using SSE2 intrinsics (`_mm_movemask_epi8`, `_mm_packus_epi16`), processing 16 elements per CPU cycle.
3.  **Bypassing Python Overhead**: All heavy lifting is done in C++, avoiding the overhead of Python's interpreter loops for element-wise access.

## BitMatrix Operations

PyCauset's `DenseBitMatrix` is a specialized container for boolean matrices that stores data as packed bits (1 bit per element), offering **8x memory savings** compared to NumPy's `bool` (1 byte per element).

Beyond storage, operations are significantly faster:

| Operation | Speedup vs NumPy | Notes |
| :--- | :--- | :--- |
| **Bitwise XOR (`^`)** | **~5.0x** | Operates on 64 bits at once (machine word) instead of byte-by-byte. |
| **Bitwise AND (`&`)** | **~5.0x** | (Estimated) Uses same optimized path. |
| **Bitwise OR (`|`)** | **~5.0x** | (Estimated) Uses same optimized path. |

### Example
```python
import pycauset
import numpy as np

# Create large boolean matrices
N = 5000
a_np = np.random.randint(0, 2, (N, N)).astype(bool)
b_np = np.random.randint(0, 2, (N, N)).astype(bool)

a_pc = pycauset.asarray(a_np)
b_pc = pycauset.asarray(b_np)

# PyCauset is ~5x faster
c_pc = a_pc ^ b_pc 
```

## Memory Efficiency

For boolean matrices, PyCauset is strictly superior in memory usage:

*   **NumPy**: 1 Byte per element.
*   **PyCauset**: 1 Bit per element.

A $10,000 \times 10,000$ boolean matrix takes:
*   **NumPy**: ~100 MB
*   **PyCauset**: ~12.5 MB

This allows you to work with much larger datasets in RAM.
