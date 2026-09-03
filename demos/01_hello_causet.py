"""Sprinkle a causal set and plot it."""

import pycauset as pc

# 1000 points in a 2D Minkowski diamond.
c = pc.causet(n=1000, seed=42)

pc.plot_embedding(c).show()
