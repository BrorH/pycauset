"""A scalar field on a causal set: propagators, Sorkin-Johnston vacuum, entropy."""

import numpy as np
import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=11)

phi = pc.field("scalar", mass=1.0)   # a set-independent field
Q = phi.on(c)                        # correlated on this causet

K = Q.retarded()       # retarded propagator K_R
iD = Q.pauli_jordan()  # i*Delta = K_R - K_A
W = Q.wightman()       # Sorkin-Johnston vacuum two-point function
S = Q.entanglement_entropy([0, 1, 2, 3, 4, 5])

print(f"retarded propagator shape: {K.shape}")
print(f"i*Delta shape:            {iD.shape}")
print(f"Wightman shape:           {W.shape}")
print(f"entanglement entropy:     {S:.6f}")
print(f"W is Hermitian:           {np.allclose(W, W.conj().T)}")
print(f"i*Delta is Hermitian:     {np.allclose(iD, iD.conj().T)}")
