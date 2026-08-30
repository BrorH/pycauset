# Pycauset Philosophy & Design Principles

PyCauset is for causal sets big enough that $O(N^2)$ storage is the bottleneck. A few mantras guide every decision:


## Core Mantras

### Build for Scale
* We assume every matrix *might* be 10TB, meaning we design for the worst case (disk-backed, out-of-core) first, then optimize the best case (RAM-only) second. While NumPy crashes if you allocate 200GB, PyCauset should handle it just fine.

### Be Lazy
*   Never compute what you can describe. For example, scalar multiplication (`A * 3.5`) is just a metadata update, taking $O(1)$ time and 0 bytes. 
*   Never write to disk what you can keep in RAM. For example, matrices stay in RAM until they grow too large or the user explicitly saves them. 

### Numpy Compatibility, C++ Engine
*   The API should feel like home to numpy-users and be intimately compatible. The engine is pure C++ optimized for our specific storage formats and causal set operations. Every operation aims to benchmark at  >0.90x the speed of NumPy for in-memory operations.

### Anti-Promotion 
*   Data types must remain as small as possible.

### Fun and Easy
*  PyCauset should be intuitive and fun to use. Users should be able to get started with a few lines of code, and the API should be easy to learn and remember.


---
