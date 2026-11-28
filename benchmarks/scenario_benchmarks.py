"""Causal-set workloads, PyCauset against NumPy.

    python benchmarks/scenario_benchmarks.py            # n = 256
    python benchmarks/scenario_benchmarks.py --n 4000   # larger (the former large_n_check.py)

Ratios are ``numpy_time / pycauset_time`` (above 1: PyCauset is faster), as in BENCHMARKS.md,
against two NumPy baselines: int64 (NumPy's exact integer matmul, a plain loop) and float32
(NumPy's BLAS route for 0/1 matrices, exact while counts stay below 2^24). Best of N; informal
wall-clock measurements, not a statistical benchmark.
"""

from __future__ import annotations

import argparse
import time
import warnings

# pycauset first (its BLAS thread defaults must be set before NumPy's OpenBLAS loads; NumPy
# imported first made NumPy's own timings 10-100x slow, review 3.15 C-8).
import pycauset as pc  # noqa: I001

if pc._numpy_imported_first:
    raise SystemExit("scenario_benchmarks.py: NumPy was imported before pycauset; run it as a script")
pc._native._set_derived_cache(False)  # time the computation, not a cached answer

import numpy as np


def _best(fn, *, repeat: int, number: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    n = ap.parse_args().n
    number = 10 if n <= 512 else 1
    repeat = 5 if n <= 512 else 3

    C = pc.causet(n=n, spacetime=pc.spacetime.MinkowskiDiamond(3), seed=1234).C
    B = pc.ones((n, n), dtype="bool")
    # The NumPy operands are prepared outside the timed calls (an astype was timed).
    Ci, Cf = np.asarray(C).astype(np.int64), np.asarray(C).astype(np.float32)
    Bi, Bf = np.ones((n, n), dtype=np.int64), np.ones((n, n), dtype=np.float32)
    ones_pc, ones_i, ones_f = pc.ones(n, dtype="int32"), np.ones(n, dtype=np.int64), np.ones(n, dtype=np.float32)

    print(f"n = {n} (a 3D diamond causal matrix C and an all-ones bool matrix B)\n")
    print(f"{'operation (ms)':<34} {'pycauset':>10} {'np int64':>10} {'np f32':>10} {'vs int64':>9} {'vs f32':>8}")

    def row(name, pc_fn, int_fn, f32_fn):
        pc_t = _best(pc_fn, repeat=repeat, number=number)
        it = _best(int_fn, repeat=repeat, number=number) if n <= 1024 else float("nan")  # int64 is O(n^3) Python-speed
        ft = _best(f32_fn, repeat=repeat, number=number)
        print(f"{name:<34} {pc_t * 1e3:>10.3f} {it * 1e3:>10.3f} {ft * 1e3:>10.3f} {it / pc_t:>8.2f}x {ft / pc_t:>7.2f}x")

    with warnings.catch_warnings():
        # bit @ bit counts in int32: the notice is expected here, not news.
        warnings.simplefilter("ignore", pc.PyCausetDTypeWarning)
        row("C @ C (path counts)", lambda: C @ C, lambda: Ci @ Ci, lambda: Cf @ Cf)
        row("B @ C (dense bool x triangular)", lambda: B @ C, lambda: Bi @ Ci, lambda: Bf @ Cf)
        row("(C @ C) @ C", lambda: (C @ C) @ C, lambda: Ci @ Ci @ Ci, lambda: Cf @ Cf @ Cf)
        row("matrix_power(C, 3)", lambda: pc.matrix_power(C, 3), lambda: np.linalg.matrix_power(Ci, 3),
            lambda: np.linalg.matrix_power(Cf, 3))
        row("C @ ones (matvec)", lambda: C @ ones_pc, lambda: Ci @ ones_i, lambda: Cf @ ones_f)

    print("\nRatios are numpy / pycauset: above 1, PyCauset is faster.")


if __name__ == "__main__":
    main()
