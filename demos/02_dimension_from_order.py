"""The dimension comes out of the order alone (Myrheim-Meyer)."""

import pycauset as pc

# Sprinkle into a 3D diamond. We never tell PyCauset the dimension.
c = pc.causet(n=400, spacetime=pc.MinkowskiDiamond(3), seed=42)

# Myrheim-Meyer reads nothing but the causal order and recovers ~3.
print(f"estimated dimension: {c.myrheim_meyer_dimension():.2f}")
