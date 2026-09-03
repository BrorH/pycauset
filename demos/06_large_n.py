"""150,000 points — the causal matrix spills to disk on its own."""

import pycauset as pc

pc.debug_mode = True
# This causal matrix spills to disk.
c = pc.causet(n=100_000, spacetime=pc.spacetime.MinkowskiDiamond(3), seed=42)

pc.plot_embedding(c).show()
