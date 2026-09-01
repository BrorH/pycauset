"""Sprinkle a causal set and plot it."""

import pycauset as pc

# 3000 points in a 2D Minkowski diamond.
c = pc.causet(n=3000, seed=42)

# c.n -> 3000, c.C -> the bit-packed causal matrix.
pc.plot_embedding(c).show()
