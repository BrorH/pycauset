"""A scalar field on a causal set: propagators and entanglement."""

import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=11)

# A field is set-independent; .on(causet) correlates it with the geometry.
Q = pc.field("scalar", mass=1.0).on(c)

K = Q.retarded()        # retarded propagator
iD = Q.pauli_jordan()   # i*Delta = K - K^T
W = Q.wightman()        # Sorkin-Johnston vacuum two-point function

print(Q.entanglement_entropy([0, 1, 2, 3, 4, 5]))
