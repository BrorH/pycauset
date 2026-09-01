import pycauset as pc

# 150k points; the ~1.4 GB causal matrix spills to disk on its own.
c = pc.causet(n=150_000, spacetime=pc._native.MinkowskiDiamond(2), seed=42)
pc.plot_embedding(c).show()
