This is the documentation for the pycauset python module, created by Bror Hjemgaard.

## What is pycauset?
_Pycauset_ is a python module to aid in numerical work with [causal sets](https://en.wikipedia.org/wiki/Causal_sets) (often shortened to "causets"). 

## Why is pycauset?
Causal sets are _huge_, and for a causal set of size $N$, the relevant mathematical objects are typically of order $\mathcal O(N^2)$ and the operations of order $\mathcal O(N^3)$. For a relatively small causet with $N=1000$ this quickly becomes unwieldly and impossible to do on consumer hardware without proper care. Pycauset provides that care. 

## How is pycauset?
Pycauset is written in C++ and stores all binary matrix entries as individual bits - saving orders of magnitude from using, say, [numpy](https://numpy.org/) for the same task. By default, each matrix is stored as a binary file, and only chunks of the full file is loaded into memory at one time. This means there is no upper limit to how large $N$ can be - as long as you have the time and storage. For example, an external storage medium can be used for this purpose.

Pycauset is written to behave similarly to [numpy](https://numpy.org/) for the user.

## Guides
*   [Installation](guides/Installation.md)
*   [User Guide](guides/User%20Guide.md)
*   [Matrix Guide](guides/Matrix%20Guide.md)
*   [Vector Guide](guides/Vector%20Guide.md)
*   [Matrix Multiplication](guides/Matrix%20Multiplication.md)
*   [Inversion](guides/Inversion.md)
*   [Bitwise Inversion](guides/BitwiseInversion.md)

## Project
*   [Philosophy & Design Principles](project/Philosophy.md)