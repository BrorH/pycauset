"""100,000 points whose causal matrix lives in a file, not in RAM (about a minute).

The causal matrix of 100,000 points is a 625 MB bit triangle. New objects above the memory
threshold (1 GiB by default) are file-backed; the demo lowers it to 256 MiB so this one is.
"""

import pycauset as pc

pc.set_memory_threshold(256 * 2**20)
c = pc.causet(n=100_000, spacetime=pc.spacetime.MinkowskiDiamond(3), seed=42)
print("causal matrix stored in:", c.C.get_backing_file())  # a .pycauset-tmp file
print("relations:", pc.sum(c.C))

pc.plot_embedding(c).show()  # a seeded random 50,000-point subset, with a warning
