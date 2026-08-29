<div align="center">
  <img src="https://raw.githubusercontent.com/BrorH/pycauset/main/documentation/docs/assets/logo/logo.png" width="150" alt="PyCauset Logo" style="vertical-align: middle; margin-right: 20px;">
  <img src="https://raw.githubusercontent.com/BrorH/pycauset/main/documentation/docs/assets/logo/logo-text-colour.png" width="300" alt="PyCauset Text" style="vertical-align: middle;">

  <br><br>

[![Documentation](https://img.shields.io/badge/docs-live-blue)](https://brorh.github.io/pycauset/)
[![PyPI version](https://badge.fury.io/py/pycauset.svg)](https://badge.fury.io/py/pycauset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## **A High-Performance Toolset for Causal Set Theory in Python.**

[Causal set theory](https://en.wikipedia.org/wiki/Causal_sets) is a discrete proposal for [quantum gravity](https://en.wikipedia.org/wiki/Quantum_gravity). PyCauset is a low-compromise numerical tool for causal sets ("causets"), built from the ground up to be fast, flexible, and easy to use. 

PyCauset is made of two main components:

- The **PyCauset Engine** is built to be [NumPy](https://numpy.org/) for causal sets. If you know NumPy, you already know the engine: the same shapes, dtypes, operators, and conventions, backed by a C++ core for speed. It is specifically made for causal sets: bit-packed causal matrices, metadata that lets the math skip unnecessary work, storage that spills to disk when RAM runs out, and CPU dispatch handled behind the scenes.

- The **PyCauset Physics Tool Suite** is an extensive collection of tools for working with causal sets: a [library](https://brorh.github.io/pycauset/guides/Spacetime/) of spacetimes, sprinkling routines, [field-theoretic machinery](https://brorh.github.io/pycauset/guides/Field%20Theory/) and visualizations. 

Read the [documentation](https://brorh.github.io/pycauset/).

## Quick start

Sprinkle a causal set into a 2D [Minkowski](https://en.wikipedia.org/wiki/Minkowski_space) diamond:

```python
import pycauset as pc
from pycauset.vis import plot_embedding

c = pc.CausalSet(n=3000, seed=42)   # 3000 points in a 2D diamond (default geometry)

fig = plot_embedding(c)             # interactive Plotly figure
fig.show()
```

<img src="https://raw.githubusercontent.com/BrorH/pycauset/main/documentation/docs/assets/gallery/diamond_embedding.png" width="520" alt="3000 points in a 2D Minkowski diamond">



Define a field and compute a propagator:

```python
from pycauset.field import ScalarField

field = ScalarField(c, mass=1.5)    # a massive scalar field on the same causet
K = field.propagator()              # retarded propagator K_R = aC(I - baC)^-1
```

Or use the engine on its own:

```python
A = pc.causal_matrix(10000, populate=True)
B = pc.causal_matrix(10000, populate=True)
Paths = pc.dot(A, B)                       # alternatively, use `A @ B` 

M = pc.zeros((2000, 2000), dtype=pc.float32)
M_inv = M.invert()                          # inversion
```

## Features

**Numerical Engine**

PyCauset has an optimized C++ core with Python bindings, which is built specifically for work with causal sets:

- **Work with matrices bigger than your RAM.** Large matrices stream to disk and back, so the only limit to computation is your storage and time.
- **It speaks [NumPy](https://numpy.org/).** Same shapes, dtypes, operators, and conventions. It is also compatible with NumPy arrays, so you can mix and match.
- **Bit-wise causal relations.** The causal matrix elements are individual bits, which allows for 8x more efficient storage than a byte-based representation.
- **CPU (and soon GPU) optimized.** *It just works.* 
- **Storage and precision, handled.** Memory, precision, and hardware are automatically chosen.

**Physics Tool Suite** 

- **Spacetimes**: [Minkowski](https://en.wikipedia.org/wiki/Minkowski_space) diamond, cylinder, box. Arbitrary dimensions, signatures, and curved geometries ([de Sitter](https://en.wikipedia.org/wiki/De_Sitter_space), [anti-de Sitter](https://en.wikipedia.org/wiki/Anti-de_Sitter_space), [FLRW](https://en.wikipedia.org/wiki/Friedmann%E2%80%93Lema%C3%AEtre%E2%80%93Robertson%E2%80%93Walker_metric)) are on the R2 roadmap.
- **Sprinkling**: fixed-N or Poisson density, seeded and reproducible.
- **Fields**: scalar fields with [propagator](https://en.wikipedia.org/wiki/Propagator) and Pauli-Jordan functions.
- **Visualization**: interactive 2D/3D embeddings and [Hasse diagrams](https://en.wikipedia.org/wiki/Hasse_diagram).

## Performance

PyCauset's dense kernels use the same OpenBLAS/LAPACK backend as NumPy, so the goal is parity rather than a large speedup, with a goal of all PyCauset operations being at least 0.90x the speed of NumPy for in-memory operations, (see [BENCHMARKS.md](BENCHMARKS.md)).
However, the biggest reason to use PyCauset is that it memory-maps past RAM, where NumPy raises MemoryError.

## Gallery

Both images come straight from the public API. Reproduce them with `scripts/make_r1_gallery.py`.

<img src="https://raw.githubusercontent.com/BrorH/pycauset/main/documentation/docs/assets/gallery/diamond_hasse.png" width="420" alt="Hasse diagram of an 80-point diamond">
*The causal links of an 80-point diamond (Hasse diagram).*

<img src="https://raw.githubusercontent.com/BrorH/pycauset/main/documentation/docs/assets/gallery/cylinder_embedding.png" width="420" alt="3000 points on a Minkowski cylinder">
*3000 points sprinkled onto a Minkowski cylinder, rendered as a 3D tube.*

## Installation

```bash
pip install pycauset
```

Pre-compiled wheels for Windows, macOS, and Linux. From source:

```bash
git clone https://github.com/BrorH/pycauset.git
cd pycauset
pip install .
```

### GPU acceleration (optional)

The default install is CPU-only. To add GPU support:

```bash
pip install "pycauset[gpu]"
```

This also downloads the CUDA runtime, about 500 MB. It needs an NVIDIA GPU with
Compute Capability 6.0 or newer, on Linux or Windows. macOS is not supported.

If you have an NVIDIA GPU but are on the CPU-only build, the first `import pycauset`
prints a short note with the install command. Set `PYCAUSET_GPU_HINT=0` to silence it.

## Status

Pre-alpha. Release 1 (v0.6.1, the Foundation Release) shipped the numerical core, matrices, storage, and linear algebra, CPU-only (GPU parity is deferred). Release 2 (the Physics Release) adds the physics suite, arbitrary-dimension and curved spacetimes, fields, and the Sorkin-Johnston vacuum, and folds in the deferred optimization program. The current plan is tracked in the documentation.

## Documentation

Guides, API reference, and the roadmap: [brorh.github.io/pycauset](https://brorh.github.io/pycauset/)

## License

[MIT](https://opensource.org/licenses/MIT). If you use PyCauset in your research, please cite [the repository](https://github.com/BrorH/pycauset).

Questions and ideas: bror dot hjemgaard at gmail dot com

This repository contains AI-generated code.
