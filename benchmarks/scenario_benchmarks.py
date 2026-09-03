"""Everyday-use scenario benchmarks with NumPy comparison.

Times representative causal-set operations in pycauset and, where a dense
NumPy equivalent exists, compares wall-clock time. Run with:

    .venv\\Scripts\\python.exe benchmarks\\scenario_benchmarks.py

These are informal wall-clock measurements (best of N), not a rigorous
statistical benchmark.
"""

import time

import numpy as np

import pycauset as pc


def _best(fn, *, repeat, number):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        dt = (time.perf_counter() - t0) / number
        best = min(best, dt)
    return best


def _dense(mat, n):
    out = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            out[i, j] = 1 if mat.get(i, j) else 0
    return out


def main():
    n = 256
    sp = pc.spacetime.MinkowskiDiamond(3)
    C = pc.CausalSet(n, spacetime=sp, seed=1234).C
    B = pc.ones((n, n), dtype=pc.bool_)
    Cn = _dense(C, n)                 # int64 dense reference
    Bn = np.ones((n, n), dtype=bool)

    print(f"n = {n} (3D diamond causal matrix + a bool matrix)\n")
    print(f"{'operation':<40} {'pycauset':>10} {'numpy':>10} {'ratio':>8}")

    def row(name, pc_fn, np_fn, *, repeat=5, number=10):
        pc_t = _best(pc_fn, repeat=repeat, number=number)
        np_t = _best(np_fn, repeat=repeat, number=number)
        print(f"{name:<40} {pc_t*1e3:>9.3f} {np_t*1e3:>9.3f} {pc_t/np_t:>7.2f}x")

    row("C @ C (triangular bit x bit)",
        lambda: C @ C, lambda: Cn @ Cn)
    row("B @ C (dense bool x triangular bit)",
        lambda: B @ C, lambda: Bn.astype(int) @ Cn)
    row("(C @ C) @ C (cube)",
        lambda: (C @ C) @ C, lambda: Cn @ Cn @ Cn)
    row("matrix_power(C, 3)",
        lambda: pc.matrix_power(C, 3), lambda: np.linalg.matrix_power(Cn, 3))
    row("dot(C, C)",
        lambda: pc.dot(C, C), lambda: Cn @ Cn)
    ones_pc = pc.ones(n, dtype="int32")
    ones_np = np.ones(n, dtype=int)
    row("C @ ones (matvec)",
        lambda: C @ ones_pc,
        lambda: Cn @ ones_np)

    # Float64 dense matmul: NumPy uses BLAS here, so this is the realistic
    # "NumPy fast path" comparison.
    F = pc.ones((n, n), dtype="float64")
    Fn = np.ones((n, n), dtype=float)
    row("float64 dense matmul (BLAS)",
        lambda: F @ F, lambda: Fn @ Fn)

    print("\nratio < 1 means pycauset is faster; > 1 means slower than NumPy.")
    print("NumPy int reference uses int64 dense arrays (NumPy's int matmul is a naive")
    print("loop, not BLAS); the float64 row is NumPy's BLAS-backed fast path.")


if __name__ == "__main__":
    main()
