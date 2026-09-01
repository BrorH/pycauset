"""Sprinkle a causal set and plot it."""

import pycauset as pc

# 3000 points in a 2D Minkowski diamond (the default geometry).
c = pc.causet(n=3000, seed=42)

pc.plot_embedding(c).show()
