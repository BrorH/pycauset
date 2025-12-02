This is the documentation for the pycauset python module, created by Bror Hjemgaard.

## What is pycauset?
_Pycauset_ is a python module to aid in numerical work with [causal sets](https://en.wikipedia.org/wiki/Causal_sets) (often shortened to "causets"). 

## Why is pycauset?
Causal sets are _huge_, and for a causal set of size $N$, the relevant mathematical objects are typically of order $\mathcal O(N^2)$ and the operations of order $\mathcal O(N^3)$. For a relatively small causet with $N=1000$ this quickly becomes unwieldly and impossible to do on consumer hardware without proper care. Pycauset provides that care. 

## How is pycauset?
Pycauset is built on a high-performance C++ core that handles the heavy lifting of matrix operations. It employs a hybrid storage model: small objects live in RAM for speed, while large datasets are automatically memory-mapped to disk. This allows you to work with causal sets of millions of elements without exhausting your system memory.

While the core engine is matrix-based, PyCauset provides a high-level **Causal Set** abstraction. You can generate causal sets by sprinkling points into various spacetime manifolds (like Minkowski space), analyze their structure, and save them to portable archives (`.causet` files) that contain both the causal structure and the spacetime metadata.

Pycauset is written to behave similarly to [numpy](https://numpy.org/) for the user, offering a familiar API for array-like operations.

## Guides
*   [[Causal Sets]] (Start Here)
*   [[Field Theory]] (New!)
*   [[Spacetime]]
*   [[User Guide]]
*   [[Installation]]
*   [[Matrix Guide]]
*   [[Vector Guide]]
*   [[Numpy Integration]]
*   [[Matrix Multiplication]]
*   [[Inversion]]
*   [[BitwiseInversion]]

## Project
*   [[Philosophy]]