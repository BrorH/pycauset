# PyCauset

[![Documentation](https://img.shields.io/badge/docs-live-blue)](https://brorh.github.io/pycauset/)
[![PyPI version](https://badge.fury.io/py/pycauset.svg)](https://badge.fury.io/py/pycauset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyCauset** is a high-performance Python library for **Causal Set Theory**. It bridges the gap between abstract mathematical models and large-scale numerical simulations, allowing researchers to work with causal sets of millions of elements on consumer hardware.

**Core philosophy:** PyCauset is **NumPy for causal sets**. Users should interact only with Python objects and a NumPy-like API, while storage, hardware dispatch (CPU/GPU), and performance optimizations happen automatically behind the scenes.

Precision and overflow are policy-driven:

*   Mixed-float ops underpromote by default (warn-once) rather than silently widening.
*   Integer overflow is a hard error; large integer matmul may emit a heuristic risk warning.

The authoritative dtype rules live in `documentation/internals/DType System.md`.

**[Explore the Full Documentation »](https://brorh.github.io/pycauset/)**

## Key Features

*   **Hybrid Storage Architecture**: PyCauset automatically manages memory. Small matrices live in RAM for speed, while massive datasets spill seamlessly to **memory-mapped disk storage** (ZIP-compressed archives).
*   **GPU Acceleration**: Built-in NVIDIA CUDA backend for matrix multiplication, inversion, and eigenvalue problems. Includes custom kernels for **accelerated bit-matrix operations**.
*   **Smart Precision**: Automatically selects `Float64` or `Float32` based on matrix size and hardware capabilities to maximize throughput.
*   **Physics Engines**:
    *   **Spacetimes**: Minkowski Diamond, Cylinder, and Box manifolds.
    *   **Fields**: Scalar field propagators ($K_R$) and path integrals.
*   **Visualization**: Interactive 3D visualization of embeddings and causal structures using Plotly.

## Installation

## Roadmap (DType System)

*   Expand integer dtype coverage (additional widths + unsigned) end-to-end.
*   Keep dtype support explicit and enforced via the support-matrix tests/tools.

### From PyPI (Recommended)
```bash
pip install pycauset
```
We provide pre-compiled binary wheels for Windows, macOS, and Linux.

### From Source
```bash
git clone https://github.com/BrorH/pycauset.git
cd pycauset
pip install .
```

### Development Install
For contributors, an editable install builds the native extension via `scikit-build-core`:

```bash
pip install -e .
```

## Quick Start

### 1. Simulating Spacetime
The `CausalSet` class is the main entry point for physics simulations.

```python
import pycauset as pc
from pycauset.vis import plot_embedding

# 1. Sprinkle 5000 points into a 2D Minkowski Diamond
c = pc.CausalSet(n=5000, density=100, seed=42)

# 2. Access the Causal Matrix (TriangularBitMatrix)
# Stored efficiently (1 bit per element)
C = c.C

# 3. Visualize the embedding
fig = plot_embedding(c)
fig.show()
```

### 2. Quantum Field Theory
Compute the Retarded Propagator ($K_R$) for a scalar field.

```python
from pycauset.field import ScalarField

# Define a massive scalar field (m=1.5) on the causal set
field = ScalarField(c, mass=1.5)

# Compute the propagator K = aC(I - b aC)^-1
# This uses GPU acceleration if available
K = field.propagator()
```

### 3. Pure Linear Algebra
You can use PyCauset as a high-performance sparse/dense matrix library.

```python
# Create a random boolean matrix (10k x 10k)
A = pc.CausalMatrix(10000, populate=True)
B = pc.CausalMatrix(10000, populate=True)

# Fast GPU-accelerated BitMatrix multiplication
# Returns an IntegerMatrix of path counts
Paths = A @ B 

# Invert a dense float matrix
M = pc.Matrix(2000, dtype=pc.float32)  # also accepts np.float32 or "float32" (case-insensitive)
M_inv = ~M # or M.inverse()
```

## GPU Acceleration

PyCauset automatically detects compatible NVIDIA GPUs.

*   **Requirements**: NVIDIA GPU (Compute Capability 6.0+ recommended).
*   **Drivers**: Recent NVIDIA Drivers.
*   **Operations**: `matmul`, `inverse`, `popcount`.

If no GPU is found, PyCauset falls back to a highly optimized multi-threaded CPU backend (OpenMP + AVX-512).

## Storage Format

Large datasets are stored in the **`.pycauset`** format, which is a standard ZIP archive containing:
1.  `metadata.json`: Dimensions, types, and generation parameters.
2.  `data.bin`: Raw binary data (memory-mapped directly from the archive).

```python
# Save your simulation
c.save("simulation_run_1.pycauset")

# Load it later (instantaneous, no RAM overhead)
c_loaded = pc.CausalSet.load("simulation_run_1.pycauset")
```

## Citation

If you use PyCauset in your research, please cite:
[https://github.com/BrorH/pycauset](https://github.com/BrorH/pycauset)
