import pycauset as pc

c = pc.causet(n=500, spacetime=pc.MinkowskiDiamond(2), seed=11)
Q = pc.field("scalar", mass=1.0).on(c)   # a massive scalar field
print(Q.entanglement_entropy([0, 1, 2, 3, 4, 5]))
