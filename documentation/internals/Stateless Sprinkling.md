# Stateless Sprinkling and Memory Management

One of the key features of `pycauset` is its ability to handle extremely large causal sets—potentially billions of elements—without exhausting system RAM. This is achieved through a technique we call **Stateless Sprinkling**.

## The Problem

A standard implementation of a causal set simulation typically follows these steps:

1.  **Generate Coordinates**: Create $N$ random points in a $D$-dimensional spacetime. This requires storing $N \times D$ floating-point numbers.
2.  **Compute Causality**: Compare every pair of points to determine if they are causally connected.
3.  **Store Matrix**: Save the resulting adjacency matrix (causal matrix).

For $N = 10^9$ (1 billion) in $D=4$ dimensions, storing the coordinates alone would require:
$$ 10^9 \times 4 \times 8 \text{ bytes} \approx 32 \text{ GB} $$
This is already pushing the limits of consumer hardware, before we even consider the memory needed for the matrix itself or the overhead of the Python interpreter.

## The Solution: Stateless Sprinkling

`pycauset` avoids storing the coordinates entirely. Instead of keeping the positions of all $N$ points in memory, we only store the **seed** used to generate them.

### Deterministic Re-generation

Because we use a deterministic pseudo-random number generator (PRNG), we can re-generate any point's coordinates on demand if we know:
1.  The global seed.
2.  The index of the point.

However, re-generating points one by one would be slow. Instead, we use a block-based approach during the matrix generation phase.

### Block-Based Matrix Generation

When generating the causal matrix (which is stored on disk as a memory-mapped file), we process the points in blocks that fit comfortably in the CPU cache (e.g., 1024 points at a time).

1.  **Generate Block A**: We generate the coordinates for a small block of points (Row Block).
2.  **Generate Block B**: We generate the coordinates for another small block (Column Block).
3.  **Compute Sub-matrix**: We compute the causality relations between points in Block A and Block B.
4.  **Discard Coordinates**: Once the sub-matrix is computed and written to disk, the coordinates for Block A and Block B are discarded.

This ensures that at any given moment, we only hold a tiny fraction of the coordinates in RAM (kilobytes, not gigabytes).

### Implications

*   **O(1) Memory for Coordinates**: The memory usage for coordinates is constant, regardless of $N$.
*   **Disk-Backed Matrices**: The resulting causal matrix is stored in a `TriangularBitMatrix` which is backed by a file on disk. This allows the OS to manage memory paging, so even the matrix doesn't need to be fully loaded into RAM.
*   **Reproducibility**: The entire causal set is defined by its size $N$ and the `seed`. This makes it easy to save and share "datasets" by just sharing the metadata.

## Summary

By trading a small amount of CPU time (re-generating coordinates) for massive memory savings, `pycauset` enables the study of causal sets orders of magnitude larger than traditional in-memory implementations.
