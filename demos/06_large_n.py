"""Large N: a causet whose causal matrix is bigger than RAM, spilled to disk.

Run it early in its own terminal and come back to it. It prints a progress
marker, then the spill result, then a DONE line when the plot is saved.

    python demos/06_large_n.py            # N = 150,000 (~1-2 min)
    python demos/06_large_n.py 80000      # smaller, faster
"""

import importlib
import os
import sys
import time

import pycauset as pc

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000

# The native (C++) spacetime triggers the fast C++ sprinkler.
ntv = importlib.import_module("pycauset._pycauset")

os.makedirs("demos/output", exist_ok=True)
pc.set_backing_dir("demos/output")

print(f"sprinkling N = {N:,} ...", flush=True)   # progress marker
t0 = time.time()
c = pc.causet(n=N, spacetime=ntv.MinkowskiDiamond(2), seed=42)

# The causal matrix is ~N^2/2 bits (~1.4 GB for N=150k), over the 1 GB default
# threshold, so it spilled to disk automatically. The .tmp file is the proof.
tmp_files = sorted(f for f in os.listdir("demos/output") if f.endswith(".tmp"))
print(f"spilled to disk: {tmp_files}", flush=True)

# Plot the (subsampled) embedding as the thing that "pops" when done.
fig = pc.plot_embedding(c, title=f"{N:,} points in a 2D diamond")
try:
    fig.write_image("demos/output/06_large_n.png", scale=2)
    print(f"DONE -> demos/output/06_large_n.png (took {time.time() - t0:.0f}s)", flush=True)
except Exception as exc:  # kaleido not installed
    print(f"(image skipped: {exc})", flush=True)
