"""The spacetime library, plus a custom spacetime."""

import os

import numpy as np
import pycauset as pc
from pycauset import spacetime

# Each built-in spacetime gives a causet when sprinkled into.
for name, st in [
    ("MinkowskiDiamond", spacetime.MinkowskiDiamond(2)),
    ("MinkowskiCylinder", spacetime.MinkowskiCylinder(2, height=2.0, circumference=5.0)),
    ("MinkowskiBox", spacetime.MinkowskiBox(2, time_extent=2.0, space_extent=1.0)),
    ("DeSitter", spacetime.DeSitter(2)),
    ("FLRW", spacetime.FLRW(2)),
    ("Schwarzschild", spacetime.Schwarzschild(2)),
]:
    c = pc.causet(n=500, spacetime=st, seed=42)
    # c.n and c.coordinates().shape confirm each one sprinkled.

# AntiDeSitter ships as a sampled parametrization but has no causal order (the
# naive hyperboloid has closed timelike curves), so it can't be sprinkled.
ads = spacetime.AntiDeSitter(2)
# ads.dimension() -> 2, ads.sample(...) -> points, but no causal order.

# A custom spacetime is just four methods: dimension, volume, sample, is_causal.
@spacetime.register("my_diamond")
class MyDiamond(spacetime.Spacetime):
    def dimension(self):
        return 2

    def volume(self):
        return 1.0

    def sample(self, rng, n):
        return rng.uniform(0.0, 1.0, size=(n, 2))

    def is_causal(self, u, v):
        return u[0] < v[0] and u[1] < v[1]


c = pc.causet(n=500, spacetime=MyDiamond(), seed=42)
c.validate()

# Save a 3D picture of the cylinder.
fig = pc.plot_embedding(
    pc.causet(n=800, spacetime=spacetime.MinkowskiCylinder(2, height=2.0, circumference=5.0), seed=42),
    title="Minkowski cylinder",
)
os.makedirs("demos/output", exist_ok=True)
try:
    fig.write_image("demos/output/04_cylinder.png", scale=2)
    print("saved demos/output/04_cylinder.png")
except Exception as exc:  # kaleido not installed
    print(f"(static image skipped: {exc})")
