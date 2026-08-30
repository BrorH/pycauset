---
title: PyCauset
---

<div class="pycauset-hero" markdown>

<div class="pycauset-hero__logos">

<img class="pycauset-hero__logo" src="docs/assets/logo/logo.png" alt="PyCauset logo">

<img class="pycauset-hero__text" src="docs/assets/logo/logo-text-colour.png" alt="PyCauset">

</div>

<h1 class="visually-hidden">PyCauset</h1>

**A high-performance Python library for numerical Causal Set Theory**

[Get started](guides/index/){ .md-button .md-button--primary }
[API reference](docs/index/){ .md-button }

</div>

## Welcome to PyCauset

PyCauset is a numerical tool for causal set theory. Causal sets are computationally
demanding: a set of size $N$ has an $O(N^2)$ causal matrix, and
[NumPy](https://numpy.org/) or [SciPy](https://scipy.org/) are not built for very
large $N$.

PyCauset handles that by spilling large matrices to disk automatically, so your
matrix size is limited only by disk and time. Small matrices behave like NumPy
arrays. See the [philosophy](project/Philosophy/) page.


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

    Deep dive into the C++ core and the design philosophy.

    - [[project/Philosophy|Philosophy]]
    - [[internals/Compute Architecture|Compute Architecture]]
    - [[internals/Memory and Data|Memory & Data]]
    - [[internals/Algorithms|Algorithms]]

</div>

## Citation

If you use PyCauset in your research, please cite the repository:
[https://github.com/BrorH/pycauset](https://github.com/BrorH/pycauset)
