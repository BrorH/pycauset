import pycauset as pc

c = pc.causet(n=400, spacetime=pc.MinkowskiDiamond(3), seed=42)   # sprinkled in 3D
print(c.myrheim_meyer_dimension())   # recovers ~3 from the order alone
