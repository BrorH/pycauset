# PyCauset

[![Documentation](https://img.shields.io/badge/docs-live-blue)](https://brorh.github.io/pycauset/)

**PyCauset** is a high-performance Python module designed for numerical work with [Causal Sets](https://en.wikipedia.org/wiki/Causal_sets). It is built to handle massive matrices that exceed available RAM by leveraging memory-mapped files and efficient C++ backends.

**[Explore the Full Documentation / Wiki »](https://brorh.github.io/pycauset/)**

## Why PyCauset?

For a causal set of size $N$, the relevant mathematical objects are typically of order $\mathcal O(N^2)$ and operations are $\mathcal O(N^3)$. For even moderate sizes like $N=10,000$, standard in-memory libraries like NumPy can struggle with memory limits.

**PyCauset solves this by:**
*   **Hybrid Storage**: Automatically keeping small matrices in RAM for speed, while seamlessly spilling large matrices to disk.
*   **Memory Mapping**: Storing massive matrices on disk and loading only necessary chunks into RAM.
*   **Bit Packing**: Storing boolean matrices (causal relations) as individual bits, reducing storage requirements by 8x-64x compared to standard types.
*   **C++ Efficiency**: Core operations are implemented in optimized C++.

## Installation

### From PyPI (Recommended)
The easiest way to install PyCauset is via pip:

```bash
pip install pycauset
```

We provide pre-compiled binary wheels for Windows, macOS, and Linux. No C++ compiler is required for installation.

### From Source
If you want to build from source or contribute:

1.  Clone the repository.
2.  Install build dependencies: `pip install scikit-build-core pybind11`.
3.  Build and install:
    ```bash
    pip install .
    ```

## GPU Acceleration

PyCauset supports GPU acceleration for matrix operations using NVIDIA CUDA.

**Requirements:**
*   NVIDIA GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper, Blackwell).
*   **Note:** Pascal (GTX 10 series) and older GPUs are **not supported** by the bundled CUDA 13.0 backend.
*   Drivers: NVIDIA Driver 520.00 or later.

The GPU backend is automatically detected and used if available. If initialization fails (e.g., unsupported hardware), PyCauset seamlessly falls back to the CPU backend.

## GPU Acceleration

PyCauset supports GPU acceleration for matrix operations using NVIDIA CUDA.

**Requirements:**
*   NVIDIA GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper, Blackwell).
*   **Note:** Pascal (GTX 10 series) and older GPUs are **not supported** by the bundled CUDA 13.0 backend.
*   Drivers: NVIDIA Driver 520.00 or later.

**Automatic Fix for Older GPUs:**
If you have a Pascal GPU (e.g., GTX 1060, 1070, 1080) and CUDA 13 installed, the build system will detect the incompatibility and offer to automatically install CUDA 12.6 via `winget`. This allows PyCauset to compile a compatible backend for your hardware.

The GPU backend is automatically detected and used if available. If initialization fails (e.g., unsupported hardware), PyCauset seamlessly falls back to the CPU backend.

## Quick Start

### 1. Creating Matrices
The primary structure for causal sets is the `TriangularBitMatrix` (aliased as `CausalMatrix`).

```python
import pycauset

# Create a random 1000x1000 causal matrix (Bernoulli p=0.5)
C = pycauset.CausalMatrix.random(1000, density=0.5)

# Create an empty matrix (initialized to zeros)
C_empty = pycauset.CausalMatrix(1000)

# Create from a NumPy array
import numpy as np
arr = np.triu(np.random.randint(0, 2, (10, 10)), k=1).astype(bool)
C_from_np = pycauset.CausalMatrix(arr)
```

### 2. Matrix Operations
PyCauset supports standard arithmetic and specialized causal set operations.

```python
# Matrix Multiplication (Counting paths of length 2)
# Returns an IntegerMatrix
M = pycauset.matmul(C, C)

# Elementwise Multiplication
E = C * C

# Bitwise Inversion (NOT)
C_inv = ~C

# Linear Algebra Inversion (for dense float matrices)
# Returns a FloatMatrix
F = pycauset.FloatMatrix(100)
F_inv = pycauset.invert(F)
```

### 3. Computing the K-Matrix
A common operation in causal set theory is computing $K = C(aI + C)^{-1}$.

```python
# Compute K with scalar a=1.0
# Returns a TriangularFloatMatrix
K = pycauset.compute_k(C, a=1.0)
```

## Matrix Types

PyCauset uses a template-based architecture to support efficient storage for different data types:

| Class | Description | Storage |
|-------|-------------|---------|
| `TriangularBitMatrix` | Strictly upper-triangular boolean matrix. | 1 bit / element |
| `IntegerMatrix` | Dense matrix of 32-bit integers. | 4 bytes / element |
| `FloatMatrix` | Dense matrix of 64-bit floats. | 8 bytes / element |
| `TriangularFloatMatrix` | Strictly upper-triangular float matrix. | 8 bytes / element |

## Storage Management

PyCauset manages disk storage automatically to keep your workspace clean.

*   **Temporary Files**: All matrices are created as temporary files in a `.pycauset/` directory. These files are **automatically deleted** when your Python script exits (even if it crashes or is interrupted).
*   **Permanent Storage**: To keep a matrix, you **must** use the `save()` function.

```python
# This matrix is temporary and will be deleted at exit
temp = pycauset.CausalMatrix(500)

# Save it permanently to a specific path
pycauset.save(temp, "my_causet.pycauset")
```
