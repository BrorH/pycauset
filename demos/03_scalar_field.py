"""A scalar field on a causal set, and its entanglement entropy."""

import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=11)

# A field is set-independent; .on(causet) gives the correlated field.
Q = pc.field("scalar", mass=1.0).on(c)

# Entanglement entropy of a small region.
print(Q.entanglement_entropy([0, 1, 2, 3, 4, 5]))
