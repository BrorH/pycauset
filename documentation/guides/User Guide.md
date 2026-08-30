# User Guide

A walk through the main things PyCauset does.

## Getting started

### 1. Install

See [[guides/Installation|Installation]].

```python
import pycauset as pc
```

### 2. Make a causal set

Pick a spacetime, then sprinkle points into it.

```python
spacetime = pc.spacetime.MinkowskiDiamond(dimension=4)

c = pc.CausalSet(1000, spacetime=spacetime)

print(f"Size: {c.N}")
print(f"Dimension: {c.spacetime.dimension()}")
```

`CausalSet` builds the causal matrix and the coordinates for you.

### 3. Plot it

```python
from pycauset.vis import plot_embedding

fig = plot_embedding(c)
fig.show()
```

See [[Visualization]] for more.

### 4. Lazy evaluation

Matrix operations like `A + B` and `A * scalar` return a lightweight expression
instead of computing immediately. The work happens when you materialize the result
(into a matrix, or into NumPy).

```python
A = pc.matrix([ [1, 2], [3, 4] ], dtype="float64")
B = pc.matrix([ [5, 6], [7, 8] ], dtype="float64")

expr = A + B                 # lazy: no computation yet
C_np = pc.to_numpy(expr)     # materialized here
```

Materialization also happens on element access and on `pc.save(...)`; see
[[guides/Matrix Guide|Matrix Guide]] for the triggers.

### 5. Save and load

Save a causal set (matrix, coordinates, metadata) to a single `.pycauset` file, and
load it back later.

```python
c.save("my_universe.pycauset")
c_loaded = pc.load("my_universe.pycauset")
```

## Physics

### Fields

```python
from pycauset.field import ScalarField

field = ScalarField(c, mass=0.5)
K = field.propagator()   # retarded propagator
```

See [[Field Theory]].

## Matrices and vectors

You can use the matrix engine directly, without a `CausalSet`. It is the same
storage and dispatch the rest of the library uses.

- [[docs/functions/pycauset.matrix.md|pycauset.matrix]]: build from data.
- [[docs/functions/pycauset.zeros.md|pycauset.zeros]] / [[docs/functions/pycauset.empty.md|pycauset.empty]]: allocate with a `dtype`.
- [[docs/functions/pycauset.matmul.md|pycauset.matmul]]: multiply.

Deeper coverage is in the [[Matrix Guide]] and [[Vector Guide]].

### NumPy

- To NumPy: `np.array(matrix)`.
- From NumPy: `pc.matrix(array)` or `pc.vector(array)`.

See [[Numpy Integration]].

## Configuration

### Memory threshold

Small objects stay in RAM; large ones go to disk. The cutoff is 1 GB by default.

```python
pc.set_memory_threshold(100 * 1024 * 1024)   # 100 MB
```

### Where the files go

Disk-backed objects use a `.pycauset` directory under your working directory by
default. Change it once after import, before allocating anything large:

```python
from pathlib import Path
import pycauset as pc
pc.set_backing_dir(Path.cwd() / "pycauset_storage")
```

Two kinds of files:

- Session temp files (`.tmp`) for spill/working storage.
- `.pycauset` snapshots written explicitly by `save()`.

See [[Storage and Memory]] for the details.

## See also

- [[docs/index|API Reference]]
- [[internals/index|Internals]]
- [[internals/DType System|DType System]]
