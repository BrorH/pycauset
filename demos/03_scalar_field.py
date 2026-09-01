"""A scalar field on a causal set: propagators, Sorkin-Johnston vacuum, entropy."""

import numpy as np
import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=11)

phi = pc.field("scalar", mass=1.0)   # a set-independent field
Q = phi.on(c)                        # correlated on this causet

# K = retarded propagator, iD = i*Delta = K - K^T, W = Sorkin-Johnston vacuum.
K = Q.retarded()
iD = Q.pauli_jordan()
W = Q.wightman()

# Entanglement entropy of a small region (list of element indices).
S = Q.entanglement_entropy([0, 1, 2, 3, 4, 5])
print(f"entanglement entropy: {S:.6f}")

# The vacuum and commutator functions must be Hermitian.
print(f"W hermitian:        {np.allclose(W, W.conj().T)}")
print(f"i*Delta hermitian:  {np.allclose(iD, iD.conj().T)}")
