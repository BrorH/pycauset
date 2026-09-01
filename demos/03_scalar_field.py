"""A scalar field: the Sorkin-Johnston vacuum and entanglement entropy."""

import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=11)

# A field is set-independent; applying it to a causet gives the correlated field.
Q = pc.field("scalar", mass=1.0).on(c)

# The Sorkin-Johnston vacuum two-point function, and the entanglement entropy
# of a small region.
W = Q.wightman()
S = Q.entanglement_entropy([0, 1, 2, 3, 4, 5])
print(f"entanglement entropy: {S:.4f}")
