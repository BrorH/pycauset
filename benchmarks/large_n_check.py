"""Large-N correctness and timing check.

Runs causal-set operations at large N and verifies correctness on a random
sample of entries (direct O(N) reference) plus a structural invariant (the
square of a strictly upper triangular matrix is strictly upper triangular).

Run with:

    .venv\\Scripts\\python.exe benchmarks\\large_n_check.py
"""

import random
import sys
import time

import pycauset as pc


def check(n, *, seed=2024, samples=40):
    t0 = time.perf_counter()
    C = pc.CausalSet(n, spacetime=pc.MinkowskiDiamond(3), seed=seed).C
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    C2 = C @ C
    t_square = time.perf_counter() - t0

    rng = random.Random(seed)
    sample_ok = True
    for _ in range(samples):
        i = rng.randrange(n)
        j = rng.randrange(i + 1, n)  # strictly upper triangle
        s = 0
        for k in range(i + 1, j):
            s += (1 if C.get(i, k) else 0) * (1 if C.get(k, j) else 0)
        if C2.get(i, j) != s:
            sample_ok = False
            print(f"  MISMATCH at ({i},{j}): got {C2.get(i, j)}, want {s}")
            break

    diag_ok = all(C2.get(i, i) == 0 for i in range(min(n, 64)))

    print(f"N={n:>6}  build={t_build:8.3f}s  C@C={t_square:8.3f}s  "
          f"sample_ok={sample_ok}  diag_ok={diag_ok}")


def main():
    if len(sys.argv) > 1:
        ns = [int(x) for x in sys.argv[1:]]
    else:
        ns = [2000, 5000, 10000]
    for n in ns:
        check(n)


if __name__ == "__main__":
    main()
