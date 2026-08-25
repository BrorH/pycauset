"""Generate the R1 gallery images used in the README.

Usage:  python scripts/make_r1_gallery.py
Output: PNGs in documentation/docs/assets/gallery/

This script only uses the public API. If it stops working,
the README images are out of date.
"""
from pathlib import Path

import pycauset as pc
from pycauset.vis import plot_embedding, plot_hasse

OUT = Path(__file__).resolve().parents[1] / "documentation" / "docs" / "assets" / "gallery"
OUT.mkdir(parents=True, exist_ok=True)

# 1. A 2D Minkowski diamond, 3000 points
c = pc.CausalSet(n=3000, seed=42)
fig = plot_embedding(c, title="Minkowski diamond, 3000 points")
fig.write_image(OUT / "diamond_embedding.png", scale=2)

# 2. A small diamond with its causal links (Hasse diagram)
c_small = pc.CausalSet(n=80, seed=7)
fig = plot_hasse(c_small, title="Hasse diagram of a small diamond, 80 points")
fig.write_image(OUT / "diamond_hasse.png", scale=2)

# 3. A Minkowski cylinder, 3000 points (3D)
st = pc.spacetime.MinkowskiCylinder(2, height=10, circumference=5)
c_cyl = pc.CausalSet(n=3000, density=60, spacetime=st, seed=11)
fig = plot_embedding(c_cyl, title="Minkowski cylinder, 3000 points")
fig.write_image(OUT / "cylinder_embedding.png", scale=2)

print("wrote:", sorted(p.name for p in OUT.glob("*.png")))
