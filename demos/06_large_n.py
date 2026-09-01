"""150,000 points — the causal matrix spills to disk on its own."""

import pycauset as pc

# The ~1.4 GB causal matrix exceeds the 1 GB threshold, so it spills to disk.
# (The native C++ spacetime keeps the sprinkle fast.)
c = pc.causet(n=150_000, spacetime=pc._native.MinkowskiDiamond(2), seed=42)

pc.plot_embedding(c).show()
