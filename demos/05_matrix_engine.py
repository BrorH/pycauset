"""The matrix engine: solve a linear system directly."""

import pycauset as pc

A = pc.matrix([[4.0, 1.0], [1.0, 3.0]], dtype="float64")
b = pc.matrix([[1.0], [2.0]], dtype="float64")

print(pc.to_numpy(pc.solve(A, b)))   # solve A x = b
