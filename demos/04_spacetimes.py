"""Different spacetimes: a diamond and a cylinder (periodic space)."""

import os

import pycauset as pc

# The default region is a 2D diamond.
c1 = pc.causet(n=800, spacetime=pc.MinkowskiDiamond(2), seed=42)

# A cylinder wraps space around, so it renders as a 3D tube.
c2 = pc.causet(n=800, spacetime=pc.MinkowskiCylinder(2, height=2.0, circumference=5.0), seed=42)

fig = pc.plot_embedding(c2, title="Minkowski cylinder")
os.makedirs("demos/output", exist_ok=True)
try:
    fig.write_image("demos/output/04_cylinder.png", scale=2)
    print("saved demos/output/04_cylinder.png")
except Exception as exc:  # kaleido not installed
    print(f"(image skipped: {exc})")
