"""Hello causal set: sprinkle, inspect, plot, save/load."""

import os

import pycauset as pc

# 1. Sprinkle 3000 points into the default 2D Minkowski diamond.
c = pc.causet(n=3000, seed=42)
print(f"sprinkled n = {c.n}")
print(f"coordinates shape = {c.coordinates().shape}")
print(f"causal matrix type = {type(c.C).__name__}")

# 2. The order is a strict partial order; the constructor already checked it.
c.validate()
print("validate() passed")

# 3. Plot it and save a static image.
fig = pc.plot_embedding(c, title="3000 points in a 2D diamond")
os.makedirs("demos/output", exist_ok=True)
try:
    fig.write_image("demos/output/01_hello_causet.png", scale=2)
    print("saved demos/output/01_hello_causet.png")
except Exception as exc:  # kaleido not installed
    print(f"(static image skipped: {exc})")

# 4. Save the whole causet to one file and load it back.
c.save("demos/output/universe.pycauset")
c2 = pc.load("demos/output/universe.pycauset")
print(f"saved and reloaded: n = {c2.n}")
