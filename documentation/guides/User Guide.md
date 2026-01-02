# PyCauset User Guide

## Overview
**PyCauset** is a high-performance library for Causal Set Theory. It bridges the gap between abstract mathematical models and large-scale numerical simulations by using a hybrid storage model: small objects live in RAM, while massive datasets may automatically **spill** by switching to temporary memory-mapped backing files on disk.

## Getting Started

### 1. Installation
Ensure you have PyCauset installed. See [[guides/Installation|guides/Installation]] for details.

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
You can save your entire causal set (including the causal matrix, coordinates, and metadata) to a portable, single-file `.pycauset` container.

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

*   **[[docs/functions/pycauset.matrix.md|pycauset.matrix]]**: Construct a matrix from data.
*   **[[docs/functions/pycauset.zeros.md|pycauset.zeros]]** / **[[docs/functions/pycauset.empty.md|pycauset.empty]]**: Allocate with an explicit `dtype`.
*   **[[docs/functions/pycauset.matmul.md|pycauset.matmul]]**: Matrix multiplication.

For a deep dive into matrix operations, see the **[[Matrix Guide]]**.

### Vectors
PyCauset supports efficient, disk-backed vectors that interoperate with its matrices.
See the **[[Vector Guide]]**.

### NumPy Integration
PyCauset is designed to work seamlessly with the scientific Python ecosystem.
*   Convert PyCauset objects to NumPy arrays: `np.array(matrix)`
*   Create PyCauset objects from NumPy arrays: `pc.matrix(array)` or `pc.vector(array)`

See the **[[Numpy Integration]]** guide.

## Configuration

### Memory Management
PyCauset automatically manages memory. Small objects stay in RAM; large ones go to disk. You can control the threshold:

```python
# Set threshold to 100 MB (default is 1 GB)
pc.set_memory_threshold(100 * 1024 * 1024)
```

### Storage Location
PyCauset may create temporary and/or backing files for disk-backed objects.

By default, these files are stored in a `.pycauset` directory under your current working directory. To override this, call `pycauset.set_backing_dir(...)` once after import (and before allocating large matrices).

Example (cross-platform):

```python
from pathlib import Path
import pycauset as pc
pc.set_backing_dir(Path.cwd() / "pycauset_storage")
```

For details on what gets stored, when cleanup happens, and how persistence works, see [[Storage and Memory]].

Rule of thumb:

- Spill/working storage uses temporary session files (for example `.tmp` under the backing directory).
- Portable persistence is explicit: `save()` writes a `.pycauset` snapshot.

Related knobs:

- [[docs/functions/pycauset.set_memory_threshold.md|pycauset.set_memory_threshold]]
- [[docs/parameters/pycauset.keep_temp_files.md|pycauset.keep_temp_files]]

---

## See also

- [[docs/index|API Reference]]
- [[internals/index|Internals]] (especially [[internals/DType System|DType System]])


