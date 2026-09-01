import pycauset as pc

c = pc.causet(n=800, spacetime=pc.MinkowskiCylinder(2, height=2.0, circumference=5.0), seed=42)
pc.plot_embedding(c).show()   # renders as a 3D tube
