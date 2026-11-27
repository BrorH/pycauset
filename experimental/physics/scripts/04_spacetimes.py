"""Spacetimes: a cylinder, and how to make your own."""

import pycauset as pc
from pycauset import spacetime

# A cylinder wraps space around, so it renders as a 3D tube.
c = pc.causet(n=800, spacetime=spacetime.MinkowskiCylinder(2, height=2.0, circumference=5.0), seed=42)
pc.plot_embedding(c).show()

# Making your own spacetime is four methods.
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
