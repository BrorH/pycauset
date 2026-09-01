"""Large N: a causet whose causal matrix is bigger than RAM, spilled to disk.

Run it early in a separate terminal and come back to it: it prints progress and,
when it finishes, saves a plot and announces itself.

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

threshold = pc.get_memory_threshold() or 0
print(f"memory threshold: {threshold / 1e6:.0f} MB (above this, objects go to disk)", flush=True)
print(f"sprinkling N = {N:,} points ...", flush=True)

t0 = time.time()
c = pc.causet(n=N, spacetime=ntv.MinkowskiDiamond(2), seed=42)
dt = time.time() - t0

bits = c.n * (c.n - 1) / 2
mb = bits / 8 / 1e6
print(f"done in {dt:.1f}s", flush=True)
print(f"causal matrix: {type(c.C).__name__}, ~{mb:.0f} MB", flush=True)
if mb > threshold / 1e6:
    print("-> exceeds the threshold, so it was spilled to disk", flush=True)
else:
    print("-> fits in RAM (under the threshold)", flush=True)

tmp_files = sorted(f for f in os.listdir("demos/output") if f.endswith(".tmp"))
print(f"disk backing files: {tmp_files}", flush=True)

# Plot the (subsampled) embedding as the thing that "pops" when done.
fig = pc.plot_embedding(c, title=f"{N:,} points in a 2D diamond")
try:
    fig.write_image("demos/output/06_large_n.png", scale=2)
    print("\n" + "=" * 52, flush=True)
    print(">>> DONE - plot saved to demos/output/06_large_n.png <<<", flush=True)
    print("=" * 52, flush=True)
except Exception as exc:  # kaleido not installed
    print(f"(image skipped: {exc})", flush=True)
