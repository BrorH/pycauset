---
title: PyCauset
---

<div class="pycauset-hero" markdown>

<img class="pycauset-hero__logo" src="docs/assets/logo/logo.png" alt="PyCauset logo">

# PyCauset

**A high-performance Python library for numerical Causal Set Theory** — a discrete
proposal for quantum gravity. A NumPy-compatible engine with a C++ core,
disk-backed matrices that spill past RAM, and a physics toolset for spacetimes,
fields, and visualization.

[Get started](guides/index/){ .md-button .md-button--primary }
[API reference](docs/index/){ .md-button }

</div>

## The Philosophy: Tiered Storage

Causal sets are computationally demanding. For a set of size $N$, the causal matrix is $O(N^2)$. For $N=100{,}000$, a dense matrix requires gigabytes of memory.

PyCauset solves this with a **Hybrid Architecture**:

1.  **RAM-First**: Small matrices behave exactly like NumPy arrays.
2.  **Disk-Backed**: Large matrices can automatically spill by switching to temporary memory-mapped backing files (for example `.tmp` files under the backing directory). Saving a portable `.pycauset` snapshot is explicit.
3.  **Bit-Packing**: Causal relations are stored as single bits, reducing memory usage by 64x compared to standard integers.

## Documentation

<div class="grid cards" markdown>

-   :material-compass-rose: **Guides**

    ---

    Practical tutorials and conceptual explanations.

    - [[guides/Installation|Installation]]
    - [[guides/User Guide|User Guide]]
    - [[guides/Causal Sets|Causal Sets]]
    - [[guides/Field Theory|Field Theory]]
    - [[guides/Visualization|Visualization]]
    - [[guides/Performance Guide|Performance]]
    - [[guides/Storage and Memory|Storage]]

-   :material-code-braces: **API Reference**

    ---

    Detailed documentation of classes and functions.

    - [[docs/classes/index|Classes]]: `CausalSet`, `Matrix`, `Vector`, `Spacetime`
    - [[docs/functions/index|Functions]]: `matmul`, `inverse`, and more

-   :material-brain: **Internals**

    ---

    Deep dive into the C++ core for contributors.

    - [[internals/Compute Architecture|Compute Architecture]]
    - [[internals/MemoryArchitecture|Memory Architecture]]
    - [[internals/Memory and Data|Memory & Data]]
    - [[internals/Algorithms|Algorithms]]

-   :material-rocket-launch: **Project**

    ---

    Design philosophy and the roadmap.

    - [[project/Philosophy|Philosophy]]
    - [[project/Contributing|Contributing]]
    - [[internals/plans/TODO|Roadmap]]

-   :material-wrench: **Dev Handbook**

    ---

    High-signal onboarding for contributors.

    - [[dev/Restructure Plan|Restructure Plan]]

</div>

## Citation

If you use PyCauset in your research, please cite the repository:
[https://github.com/BrorH/pycauset](https://github.com/BrorH/pycauset)
