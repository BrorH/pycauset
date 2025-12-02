# Welcome to PyCauset

**PyCauset** is a high-performance Python library for numerical Causal Set Theory. It is designed to bridge the gap between abstract mathematical models and large-scale numerical simulations.

## Key Features

*   **Hybrid Architecture**: A user-friendly Python interface backed by a highly optimized C++ core.
*   **Scalability**: Uses memory-mapped disk storage for large matrices, allowing you to work with causal sets of millions of elements on consumer hardware.
*  **Numpy Integration**: Seamless interoperability with the scientific Python ecosystem.
*  **Field Theory**: Tools for defining fields and computing propagators on causal sets backgrounds.
*   **Rich Spacetimes**: Built-in support for various spacetime manifolds
*   **Visualization**: Interactive 2D and 3D visualizations powered by Plotly


## Getting Started

If you are new to PyCauset, we recommend starting with the following resources:

1.  **[[Installation]]**: Get PyCauset running on your machine.
2.  **[[User Guide]]**: A step-by-step introduction to the core concepts.
3.  **[[Causal Sets]]**: Learn about the fundamental object of the library.
4.  **[[Visualization]]**: Explore your causal sets with interactive plots.

## Documentation Structure

*   **[[guides/index|Guides]]**: In-depth tutorials and conceptual explanations.
*   **[[docs/index|API Reference]]**: Detailed documentation of classes and functions.
*   **[[internals/index|Internals]]**: Information about the C++ core, file formats, and algorithms.
*   **[[project/index|Project]]**: Philosophy, roadmap, and contribution guidelines.

## Why PyCauset?

Causal sets are computationally demanding. For a set of size $N$, the causal matrix is $O(N^2)$ and transitive closure is $O(N^3)$. For $N=10,000$, a dense matrix of integers requires hundreds of megabytes. For $N=100,000$, it requires gigabytes.

PyCauset solves this by:
1.  **Bit-packing**: Using bitsets where possible to reduce memory usage by 32x or 64x.
2.  **Memory Mapping**: Transparently offloading large matrices to disk, so your RAM is never the bottleneck.
3.  **Lazy Evaluation**: Computing properties only when needed.

## Citation

If you use PyCauset in your research, please cite the repository:
[https://github.com/BrorH/pycauset](https://github.com/BrorH/pycauset)
