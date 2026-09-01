import pycauset as pc

c = pc.causet(n=3000, seed=42)   # 3000 points in a 2D diamond
pc.plot_embedding(c).show()
