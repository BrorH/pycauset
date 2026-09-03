"""Causal relations drawn on top of the embedding."""

import pycauset as pc

# A small diamond, so the causal relations stay readable.
c = pc.causet(n=40, spacetime=pc.spacetime.MinkowskiDiamond(2), seed=7)

# show_relations draws a faint line for every pair A < B.
pc.plot_embedding(c, show_relations=True).show()
