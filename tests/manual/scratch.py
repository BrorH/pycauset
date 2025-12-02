import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))
from pycauset import Causet, spacetime

# 1. Standard Diamond (Fixed N)
c1 = Causet(n=1000, spacetime=spacetime.MinkowskiDiamond(2))

# 2. Cylinder (Periodic Boundary) with Density
# This will generate N ~ Poisson(density * volume)
cyl = spacetime.MinkowskiCylinder(dimension=2, height=10.0, circumference=5.0)
c2 = Causet(density=100, spacetime=cyl)



print(f"Generated {c2.n} elements in a cylinder of volume {cyl.volume()}")

print(c1.C)