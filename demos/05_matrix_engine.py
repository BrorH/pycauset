"""The matrix/vector engine on its own, plus synthetic orders."""

import numpy as np
import pycauset as pc

# Linear algebra without the physics.
A = pc.matrix([[4.0, 1.0], [1.0, 3.0]], dtype="float64")
b = pc.matrix([[1.0], [2.0]], dtype="float64")

x = pc.solve(A, b)          # solve A x = b
vals, _ = pc.eigh(A)        # symmetric eigenvalues/vectors
U, s, Vh = pc.svd(A)        # singular value decomposition

print(f"solve:           {pc.to_numpy(x).ravel()}")
print(f"eigenvalues:     {np.asarray(vals)}")
print(f"singular values: {np.asarray(s)}")

# Synthetic orders, no geometry needed.
import pycauset.synthetic as syn

chain = syn.chain(20)
anti = syn.antichain(20)
tp = syn.transitive_percolation(0.2, 50)

print(f"chain is_chain:         {chain.is_chain(list(range(20)))}")
print(f"antichain is_antichain: {anti.is_antichain(list(range(20)))}")
