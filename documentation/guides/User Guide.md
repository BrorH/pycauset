# PyCauset User Guide

## Overview
**PyCauset** is a high-performance library for Causal Set Theory. It bridges the gap between abstract mathematical models and large-scale numerical simulations by using a hybrid storage model: small objects live in RAM, while massive datasets automatically spill over to memory-mapped files on disk.

## Getting Started

### 1. Installation
Ensure you have PyCauset installed. See the [[Installation]] guide for details.

```python
import pycauset as pc
```

### 2. Creating a Causal Set
The primary workflow involves defining a spacetime region and "sprinkling" points into it to generate a Causal Set.

```python
# 1. Define a Spacetime (e.g., 4D Minkowski Diamond)
# The library handles the geometry and causal relations.
spacetime = pc.spacetime.MinkowskiDiamond(dimension=4)

# 2. Generate a Causal Set
# Sprinkle 1000 points into the spacetime.
c = pc.CausalSet(1000, spacetime=spacetime)

# Access properties
print(f"Size: {c.N}")
print(f"Dimension: {c.spacetime.dimension()}")
```

### 3. Visualization
PyCauset provides interactive 3D visualizations powered by Plotly.

```python
from pycauset.vis import plot_embedding

# Generate an interactive 3D plot of the causal set
fig = plot_embedding(c)
fig.show()
```
See the [[Visualization]] guide for more options.

### 4. Saving and Loading
You can save your entire causal set (including the causal matrix, coordinates, and metadata) to a portable archive file.

```python
# Save to disk
c.save("my_universe.pycauset")

# Load it back later
c_loaded = pc.load("my_universe.pycauset")
```

## Physics & Analysis

### Field Theory
You can define quantum fields on your causal set background.

```python
from pycauset.field import ScalarField

# Define a massive scalar field on a causal set "c"
field = ScalarField(c, mass=0.5)

# Compute the Retarded Propagator (K)
# This uses the highly optimized matrix engine under the hood.
K = field.propagator()
```
See the [[Field Theory]] guide for details.

## Advanced Usage

### The Matrix Engine
Under the hood, PyCauset uses a powerful matrix engine that handles data larger than RAM. While `CausalSet` abstracts this away, you can use the matrix classes directly for linear algebra.

*   **[[pycauset.Matrix]]**: Factory for creating dense or sparse matrices.
*   **[[pycauset.matmul]]**: Optimized matrix multiplication.

For a deep dive into matrix operations, see the **[[Matrix Guide]]**.

### Vectors
PyCauset supports efficient, disk-backed vectors that interoperate with its matrices.
See the **[[Vector Guide]]**.

### NumPy Integration
PyCauset is designed to work seamlessly with the scientific Python ecosystem.
*   Convert PyCauset objects to NumPy arrays: `np.array(matrix)`
*   Create PyCauset objects from NumPy arrays: `pc.Matrix(array)` or `pc.Vector(array)`

See the **[[Numpy Integration]]** guide.

## Configuration

### Memory Management
PyCauset automatically manages memory. Small objects stay in RAM; large ones go to disk. You can control the threshold:

```python
# Set threshold to 100 MB (default is 1 GB)
pc.set_memory_threshold(100 * 1024 * 1024)
```

### Storage Location
By default, temporary files are stored in a `.pycauset` folder in your working directory. You can change this by setting the `PYCAUSET_STORAGE_DIR` environment variable.


