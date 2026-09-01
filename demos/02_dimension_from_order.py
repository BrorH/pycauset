"""Recover the spacetime dimension from the order alone."""

import pycauset as pc

# Sprinkle into a 3D diamond — we never tell PyCauset the dimension.
c = pc.causet(n=400, spacetime=pc.MinkowskiDiamond(3), seed=42)

# Myrheim-Meyer reads the causal order and recovers ~3.
print(c.myrheim_meyer_dimension())
